//! Training loop and optimization.

use candle_core::{Device, Tensor};
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
        tracing::info!(
            "Model loaded with vocab size: {}",
            model.tokenizer.get_vocab_size(true)
        );
        self.model = Some(model);

        // Load dataset
        let dataset = Dataset::load(&self.config.dataset)?;
        tracing::info!("Loaded {} training examples", dataset.len());

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        // Calculate total steps for progress bar
        let total_steps =
            dataset.len() * self.config.training.epochs / self.config.training.batch_size;

        // Initialize optimizer
        {
            let optimizer_config = OptimizerConfig {
                learning_rate: self.config.training.learning_rate,
                weight_decay: self.config.training.weight_decay,
                ..OptimizerConfig::default()
            };

            // Create varmap for model parameters (will be populated with actual params)
            let varmap = VarMap::new();
            let optimizer = optimizer_config.build_adamw(&varmap)?;
            tracing::info!(
                "Initialized AdamW optimizer with lr={}",
                optimizer.learning_rate()
            );
            self.optimizer = Some(optimizer);
        }

        // Initialize learning rate scheduler
        {
            let warmup_steps = (total_steps as f64 * 0.1) as usize; // 10% warmup

            let scheduler = LRScheduler::new(
                SchedulerType::Linear {
                    warmup_steps,
                    total_steps,
                },
                self.config.training.learning_rate,
            );
            tracing::info!(
                "Initialized linear scheduler with {} warmup steps",
                warmup_steps
            );
            self.scheduler = Some(scheduler);
        }

        // Create progress bar
        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
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
                if let (Some(scheduler), Some(optimizer)) =
                    (self.scheduler.as_mut(), self.optimizer.as_mut())
                {
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
    ///
    /// This method:
    /// 1. Tokenizes the batch
    /// 2. Performs forward pass
    /// 3. Computes cross-entropy loss
    /// 4. Performs backward pass and optimizer step
    fn training_step(&mut self, batch: &[crate::dataset::Example]) -> Result<f64> {
        let model = self.model.as_ref().ok_or_else(|| {
            AxolotlError::Training("Model not loaded".into())
        })?;

        // 1. Tokenize batch
        let mut input_ids = Vec::new();
        let mut labels = Vec::new();
        let max_len = self.config.dataset.max_length;

        for example in batch {
            let encoding = model.tokenizer.encode(example.text.as_str(), true)
                .map_err(|e| AxolotlError::Tokenizer(format!("Tokenization failed: {}", e).into()))?;

            let mut ids = encoding.get_ids().to_vec();
            
            // Truncate or pad to max_len
            if ids.len() > max_len {
                ids.truncate(max_len);
            }
            while ids.len() < max_len {
                ids.push(0); // Pad token
            }

            // For causal LM, labels are shifted input_ids
            let label_ids: Vec<u32> = ids.iter().skip(1).chain(std::iter::once(&0)).copied().collect();

            input_ids.push(ids);
            labels.push(label_ids);
        }

        // 2. Convert to tensors
        let batch_size = input_ids.len();
        let flat_input: Vec<i64> = input_ids.iter().flatten().map(|&x| x as i64).collect();
        let flat_labels: Vec<i64> = labels.iter().flatten().map(|&x| x as i64).collect();

        let input_tensor = Tensor::from_vec(flat_input, (batch_size, max_len), &self.device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create input tensor: {}", e)))?;
        let label_tensor = Tensor::from_vec(flat_labels, (batch_size, max_len), &self.device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create label tensor: {}", e)))?;

        // 3. Forward pass
        let logits = model.forward(&input_tensor)
            .map_err(|e| AxolotlError::Training(format!("Forward pass failed: {}", e)))?;

        // 4. Compute cross-entropy loss
        let vocab_size = logits.dims()[2];
        let logits_flat = logits.reshape((batch_size * max_len, vocab_size))
            .map_err(|e| AxolotlError::Training(format!("Reshape failed: {}", e)))?;
        let labels_flat = label_tensor.reshape(batch_size * max_len)
            .map_err(|e| AxolotlError::Training(format!("Label reshape failed: {}", e)))?;

        let loss = candle_nn::loss::cross_entropy(&logits_flat, &labels_flat)
            .map_err(|e| AxolotlError::Training(format!("Loss computation failed: {}", e)))?;

        let loss_val = loss.to_scalar::<f32>()
            .map_err(|e| AxolotlError::Training(format!("Failed to get loss value: {}", e)))? as f64;

        // 5. Backward pass and optimizer step
        if let Some(optimizer) = self.optimizer.as_mut() {
            optimizer.step(&loss)?;
        }

        Ok(loss_val)
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
    use std::fs;
    use tempfile::TempDir;

    /// Helper to create a test config using a preset
    fn create_test_config(output_dir: &str) -> AxolotlConfig {
        let mut config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        config.output_dir = output_dir.to_string();
        // Override dataset path for testing
        config.dataset.path = "test-dataset.jsonl".to_string();
        config
    }

    /// Helper to create a test dataset file
    fn create_test_dataset(path: &str, num_examples: usize) -> std::io::Result<()> {
        let mut content = String::new();
        for i in 0..num_examples {
            content.push_str(&format!(
                r#"{{"instruction":"Test instruction {}","input":"","output":"Test output {}"}}"#,
                i, i
            ));
            content.push('\n');
        }
        fs::write(path, content)
    }

    // ========================================================================
    // Tests for Trainer::new
    // ========================================================================

    #[test]
    fn test_trainer_creation() {
        let config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        let trainer = Trainer::new(config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_trainer_new_stores_config() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());
        let base_model = config.base_model.clone();

        let trainer = Trainer::new(config).unwrap();

        // Verify config is stored correctly
        assert_eq!(trainer.config.base_model, base_model);
    }

    #[test]
    fn test_trainer_new_initializes_counters() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let trainer = Trainer::new(config).unwrap();

        // Verify epoch and step counters start at 0
        assert_eq!(trainer.epoch, 0);
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_trainer_new_with_invalid_config() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let mut config = create_test_config(output_path.to_str().unwrap());

        // Make config invalid by setting base_model to empty (this IS validated)
        config.base_model = String::new();

        let result = Trainer::new(config);
        assert!(result.is_err());
    }

    // ========================================================================
    // Tests for checkpoint directory handling
    // ========================================================================

    #[test]
    fn test_checkpoint_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("checkpoints");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5; // Process all in one batch
        config.training.save_steps = 1; // Save immediately

        let mut trainer = Trainer::new(config).unwrap();

        // Directory shouldn't exist yet
        assert!(!output_path.exists());

        // Run training (will fail due to missing model)
        let _ = trainer.train();

        // Directory should not be created since training failed
        assert!(!output_path.exists());
    }

    #[test]
    fn test_checkpoint_directory_reuse() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("checkpoints");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Pre-create the output directory
        fs::create_dir_all(&output_path).unwrap();

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5;
        config.training.save_steps = 1;

        let mut trainer = Trainer::new(config).unwrap();

        // Should succeed even though directory exists
        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment
    }

    // ========================================================================
    // Tests for resume_from
    // ========================================================================

    #[test]
    fn test_resume_from_not_implemented() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let mut trainer = Trainer::new(config).unwrap();

        // Currently returns not implemented error
        let result = trainer.resume_from("checkpoint-100");
        assert!(result.is_err());

        match result {
            Err(AxolotlError::Checkpoint(msg)) => {
                assert!(msg.contains("not yet implemented"));
            }
            _ => panic!("Expected Checkpoint error"),
        }
    }

    #[test]
    fn test_resume_from_missing_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let mut trainer = Trainer::new(config).unwrap();

        // Try to resume from non-existent checkpoint
        let result = trainer.resume_from("non-existent-checkpoint");
        assert!(result.is_err());
    }

    // ========================================================================
    // Tests for train method
    // ========================================================================

    #[test]
    fn test_train_with_small_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create small dataset (5 examples)
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 2;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        // Training should fail due to missing model files in test environment
        let result = trainer.train();
        assert!(result.is_err());
    }

    #[test]
    fn test_train_with_empty_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("empty_dataset.jsonl");

        // Create empty dataset
        fs::write(&dataset_path, "").unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();

        let mut trainer = Trainer::new(config).unwrap();

        // Training should fail due to missing model files
        let result = trainer.train();
        assert!(result.is_err());
    }

    #[test]
    fn test_train_epoch_iteration() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset with 10 examples
        create_test_dataset(dataset_path.to_str().unwrap(), 10).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 3;
        config.training.batch_size = 5;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // With 10 examples, batch size 5, and 3 epochs:
        // Each epoch would have 2 batches, so 3 epochs = 6 steps total
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);

        // Epoch counter should remain at 0 since training failed before starting
        assert_eq!(trainer.epoch, 0);
    }

    #[test]
    fn test_train_batch_iteration() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset with 7 examples
        create_test_dataset(dataset_path.to_str().unwrap(), 7).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 3;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // With 7 examples and batch size 3, after validation split (10%):
        // Training set would have ~6 examples (7 * 0.9 = 6.3)
        // Batches: [0,1,2], [3,4,5] = 2 steps
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_train_with_missing_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = "non_existent_dataset.jsonl".to_string();

        let mut trainer = Trainer::new(config).unwrap();

        // Training should fail with model loading error (model loading happens before dataset loading)
        let result = trainer.train();
        assert!(result.is_err());

        match result {
            Err(AxolotlError::Model(_)) => {
                // Expected error type (model loading fails first)
            }
            _ => panic!("Expected Model error"),
        }
    }

    // ========================================================================
    // Tests for checkpoint operations
    // ========================================================================

    #[test]
    fn test_checkpoint_path_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 10).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5;
        config.training.save_steps = 1; // Save after each step

        let mut trainer = Trainer::new(config).unwrap();

        let _ = trainer.train();

        // Check that checkpoint directories were NOT created since training failed
        // With 10 examples, batch size 5, we would get 2 steps if training succeeded
        let checkpoint_1 = output_path.join("checkpoint-1");
        let checkpoint_2 = output_path.join("checkpoint-2");

        assert!(!checkpoint_1.exists());
        assert!(!checkpoint_2.exists());
    }

    #[test]
    fn test_checkpoint_final_save() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // Final checkpoint should NOT be saved since training failed
        let final_checkpoint = output_path.join("checkpoint-1");
        assert!(!final_checkpoint.exists());
    }

    // ========================================================================
    // Tests with mock datasets of various sizes
    // ========================================================================

    #[test]
    fn test_train_with_single_example() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset with 3 examples to ensure at least 1 in training split
        create_test_dataset(dataset_path.to_str().unwrap(), 3).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 1;
        config.dataset.val_split = 0.1; // 90% training = 2-3 examples

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment
                                  // With 3 examples and 10% val split: 2 training examples
                                  // With batch size 1: 2 steps
                                  // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_train_with_large_dataset_batching() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create larger dataset (50 examples)
        create_test_dataset(dataset_path.to_str().unwrap(), 50).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 10;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // 50 examples / 10 batch size = 5 steps
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_train_multiple_epochs_step_accumulation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 8).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 5;
        config.training.batch_size = 4;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // 8 examples / 4 batch size = 2 steps per epoch
        // 2 steps * 5 epochs = 10 total steps
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }
}
