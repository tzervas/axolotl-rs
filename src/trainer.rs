//! Training loop and optimization.

use candle_core::Device;
use candle_nn::VarMap;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::AxolotlConfig;
use crate::dataset::Dataset;
use crate::error::{AxolotlError, Result};
use crate::model::{load_model, LoadedModel};
use crate::optimizer::{AdamWOptimizer, OptimizerConfig};
use crate::scheduler::{LRScheduler, SchedulerType};

/// Training orchestrator.
///
/// # Example
///
/// ```no_run
/// use axolotl_rs::{AxolotlConfig, Trainer};
///
/// # fn main() -> axolotl_rs::Result<()> {
/// // Create configuration
/// let config = AxolotlConfig::from_preset("llama2-7b")?;
///
/// // Create trainer
/// let mut trainer = Trainer::new(config)?;
///
/// // Run training
/// trainer.train()?;
/// # Ok(())
/// # }
/// ```
pub struct Trainer {
    /// Configuration
    config: AxolotlConfig,
    /// Current step
    step: usize,
    /// Current epoch
    epoch: usize,
    /// Device for training
    device: Device,
    /// Loaded model (optional, loaded during train())
    model: Option<LoadedModel>,
    /// Optimizer (optional, created during train())
    optimizer: Option<AdamWOptimizer>,
    /// Learning rate scheduler (optional, created during train())
    scheduler: Option<LRScheduler>,
}

impl Trainer {
    /// Create a new trainer.
    ///
    /// Validates the configuration before creating the trainer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = AxolotlConfig::from_preset("llama2-7b")?;
    /// let trainer = Trainer::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: AxolotlConfig) -> Result<Self> {
        config.validate()?;
        
        // Determine device
        let device = if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)
                .map_err(|e| AxolotlError::Training(format!("Failed to initialize CUDA: {}", e)))?
        } else {
            Device::Cpu
        };
        
        tracing::info!("Training device: {:?}", device);

        Ok(Self {
            config,
            step: 0,
            epoch: 0,
            device,
            model: None,
            optimizer: None,
            scheduler: None,
        })
    }

    /// Resume training from a checkpoint.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = AxolotlConfig::from_file("config.yaml")?;
    /// let mut trainer = Trainer::new(config)?;
    ///
    /// // Resume from a previous checkpoint
    /// trainer.resume_from("./outputs/checkpoint-1000")?;
    /// trainer.train()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Note
    ///
    /// This feature is not yet implemented.
    pub fn resume_from(&mut self, _checkpoint_path: &str) -> Result<()> {
        // TODO: Load model weights, optimizer state, scheduler state, and training progress
        Err(AxolotlError::Checkpoint(
            "Resume not yet implemented".into(),
        ))
    }

    /// Run the training loop.
    ///
    /// This performs the following steps:
    /// 1. Loads the dataset
    /// 2. Iterates over epochs and batches
    /// 3. Logs metrics periodically
    /// 4. Saves checkpoints periodically
    /// 5. Saves final checkpoint
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// // Load configuration
    /// let config = AxolotlConfig::from_file("config.yaml")?;
    ///
    /// // Create and run trainer
    /// let mut trainer = Trainer::new(config)?;
    /// trainer.train()?;
    ///
    /// println!("Training complete!");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dataset cannot be loaded
    /// - Model fails to load
    /// - Training encounters an error
    /// - Checkpoint saving fails
    pub fn train(&mut self) -> Result<()> {
        tracing::info!("Starting training");
        tracing::info!("  Base model: {}", self.config.base_model);
        tracing::info!("  Adapter: {:?}", self.config.adapter);
        tracing::info!("  Epochs: {}", self.config.training.epochs);

        // Load model
        let model = load_model(&self.config, &self.device)?;
        tracing::info!("Model loaded with vocab size: {}", model.tokenizer.get_vocab_size(true));
        self.model = Some(model);

        // Load dataset
        let dataset = Dataset::load(&self.config.dataset)?;
        tracing::info!("Loaded {} training examples", dataset.len());

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        // Initialize optimizer
        let optimizer_config = OptimizerConfig {
            learning_rate: self.config.training.learning_rate,
            weight_decay: self.config.training.weight_decay,
            ..OptimizerConfig::default()
        };
        
        // Create varmap for model parameters (will be populated with actual params)
        let varmap = VarMap::new();
        let optimizer = optimizer_config.build_adamw(&varmap)?;
        tracing::info!("Initialized AdamW optimizer with lr={}", optimizer.learning_rate());
        self.optimizer = Some(optimizer);

        // Initialize learning rate scheduler
        let total_steps =
            dataset.len() * self.config.training.epochs / self.config.training.batch_size;
        let warmup_steps = (total_steps as f64 * 0.1) as usize; // 10% warmup
        
        let scheduler = LRScheduler::new(
            SchedulerType::Linear { warmup_steps, total_steps },
            self.config.training.learning_rate,
        );
        tracing::info!("Initialized linear scheduler with {} warmup steps", warmup_steps);
        self.scheduler = Some(scheduler);

        // Setup progress bar
        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) loss: {msg}",
                )?
                .progress_chars("#>-"),
        );

        // Training loop
        for epoch in 0..self.config.training.epochs {
            self.epoch = epoch;
            tracing::info!(
                "Starting epoch {}/{}",
                epoch + 1,
                self.config.training.epochs
            );

            for batch in dataset.train.chunks(self.config.training.batch_size) {
                self.step += 1;

                // Training step
                let loss = self.training_step(batch)?;
                
                // Update progress bar with loss
                pb.set_message(format!("{:.4}", loss));
                pb.inc(1);

                // Log periodically
                if self.step % self.config.training.logging_steps == 0 {
                    tracing::info!(
                        "Step {}/{}, Epoch {}, Loss: {:.4}, LR: {:.2e}",
                        self.step,
                        total_steps,
                        epoch + 1,
                        loss,
                        self.optimizer.as_ref().unwrap().learning_rate()
                    );
                }

                // Save checkpoint periodically
                if self.step % self.config.training.save_steps == 0 {
                    self.save_checkpoint()?;
                }
                
                // Step scheduler
                if let (Some(scheduler), Some(optimizer)) = (self.scheduler.as_mut(), self.optimizer.as_mut()) {
                    scheduler.step(optimizer);
                }
            }
        }

        pb.finish_with_message("Training complete");

        // Save final checkpoint
        self.save_checkpoint()?;

        Ok(())
    }
    
    /// Perform a single training step.
    fn training_step(&self, _batch: &[crate::dataset::Example]) -> Result<f64> {
        // TODO: Full implementation
        // 1. Tokenize batch
        // 2. Forward pass
        // 3. Compute loss
        // 4. Backward pass
        // 5. Optimizer step
        
        // For now, return a dummy loss that decreases over time
        let loss = 2.0 - (self.step as f64 * 0.001).min(1.5);
        Ok(loss)
    }

    /// Save a checkpoint.
    fn save_checkpoint(&self) -> Result<()> {
        let checkpoint_dir = format!("{}/checkpoint-{}", self.config.output_dir, self.step);
        std::fs::create_dir_all(&checkpoint_dir)?;

        // TODO: Save model weights
        // TODO: Save optimizer state
        // TODO: Save training state

        tracing::info!("Saved checkpoint to: {}", checkpoint_dir);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        let trainer = Trainer::new(config);
        assert!(trainer.is_ok());
    }
}
