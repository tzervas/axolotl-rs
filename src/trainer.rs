//! Training loop and optimization.

use indicatif::{ProgressBar, ProgressStyle};

use crate::config::AxolotlConfig;
use crate::dataset::Dataset;
use crate::error::{AxolotlError, Result};

/// Training orchestrator.
pub struct Trainer {
    /// Configuration
    config: AxolotlConfig,
    /// Current step
    step: usize,
    /// Current epoch
    epoch: usize,
}

impl Trainer {
    /// Create a new trainer.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: AxolotlConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            step: 0,
            epoch: 0,
        })
    }

    /// Resume training from a checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be loaded.
    pub fn resume_from(&mut self, _checkpoint_path: &str) -> Result<()> {
        // TODO: Load checkpoint state
        Err(AxolotlError::Checkpoint(
            "Resume not yet implemented".into(),
        ))
    }

    /// Run the training loop.
    ///
    /// # Errors
    ///
    /// Returns an error if training fails or the dataset cannot be loaded.
    pub fn train(&mut self) -> Result<()> {
        tracing::info!("Starting training");
        tracing::info!("  Base model: {}", self.config.base_model);
        tracing::info!("  Adapter: {:?}", self.config.adapter);
        tracing::info!("  Epochs: {}", self.config.training.epochs);

        // Load dataset
        let dataset = Dataset::load(&self.config.dataset)?;
        tracing::info!("Loaded {} training examples", dataset.len());

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        // Setup progress bar
        let total_steps =
            dataset.len() * self.config.training.epochs / self.config.training.batch_size;
        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .map_err(|e| AxolotlError::Training(format!("Progress bar template error: {e}")))?
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

            for _batch in dataset.train.chunks(self.config.training.batch_size) {
                self.step += 1;

                // TODO: Actual training step
                // 1. Tokenize batch
                // 2. Forward pass
                // 3. Compute loss
                // 4. Backward pass
                // 5. Optimizer step

                pb.inc(1);

                // Log periodically
                if self.step % self.config.training.logging_steps == 0 {
                    // TODO: Log metrics
                }

                // Save checkpoint periodically
                if self.step % self.config.training.save_steps == 0 {
                    self.save_checkpoint()?;
                }
            }
        }

        pb.finish_with_message("Training complete");

        // Save final checkpoint
        self.save_checkpoint()?;

        Ok(())
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

    /// Helper to create a minimal valid config for testing
    fn create_test_config(output_dir: &str) -> AxolotlConfig {
        use crate::config::{AdapterType, DatasetConfig, TrainingConfig};

        AxolotlConfig {
            base_model: "test-model".to_string(),
            adapter: AdapterType::Lora,
            lora: Default::default(),
            quantization: None,
            dataset: DatasetConfig {
                path: "test-dataset.jsonl".to_string(),
                format: Default::default(),
                input_field: "instruction".to_string(),
                output_field: "output".to_string(),
                max_length: 512,
                val_split: 0.1,
            },
            training: TrainingConfig {
                batch_size: 4,
                epochs: 2,
                learning_rate: 1e-4,
                warmup_ratio: 0.03,
                save_steps: 500,
                logging_steps: 10,
                gradient_accumulation_steps: 1,
                max_grad_norm: 1.0,
                lr_scheduler: Default::default(),
                weight_decay: 0.0,
                gradient_checkpointing: false,
                mixed_precision: true,
            },
            output_dir: output_dir.to_string(),
            seed: 42,
        }
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

        // Run training (will create directory)
        let _ = trainer.train();

        // Directory should now exist
        assert!(output_path.exists());
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
        assert!(result.is_ok());
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

        // Training should complete without errors
        let result = trainer.train();
        assert!(result.is_ok());

        // Step counter should have advanced
        assert!(trainer.step > 0);
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

        // Training with empty dataset should complete (no iterations)
        let result = trainer.train();
        assert!(result.is_ok());

        // Step counter should remain at 0
        assert_eq!(trainer.step, 0);
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
        assert!(result.is_ok());

        // With 10 examples, batch size 5, and 3 epochs:
        // Each epoch has 2 batches, so 3 epochs = 6 steps total
        assert_eq!(trainer.step, 6);

        // Epoch counter should be at the last epoch (0-indexed)
        assert_eq!(trainer.epoch, 2);
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
        assert!(result.is_ok());

        // With 7 examples and batch size 3, after validation split (10%):
        // Training set has ~6 examples (7 * 0.9 = 6.3)
        // Batches: [0,1,2], [3,4,5] = 2 steps
        assert_eq!(trainer.step, 2);
    }

    #[test]
    fn test_train_with_missing_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = "non_existent_dataset.jsonl".to_string();

        let mut trainer = Trainer::new(config).unwrap();

        // Training should fail with dataset error
        let result = trainer.train();
        assert!(result.is_err());

        match result {
            Err(AxolotlError::Dataset(_)) => {
                // Expected error type
            }
            _ => panic!("Expected Dataset error"),
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

        // Check that checkpoint directories were created
        // With 10 examples, batch size 5, we get 2 steps
        let checkpoint_1 = output_path.join("checkpoint-1");
        let checkpoint_2 = output_path.join("checkpoint-2");

        assert!(checkpoint_1.exists());
        assert!(checkpoint_2.exists());
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
        assert!(result.is_ok());

        // Final checkpoint should be saved
        let final_checkpoint = output_path.join("checkpoint-1");
        assert!(final_checkpoint.exists());
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
        assert!(result.is_ok());
        // With 3 examples and 10% val split: 2 training examples
        // With batch size 1: 2 steps
        assert_eq!(trainer.step, 2);
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
        assert!(result.is_ok());

        // 50 examples / 10 batch size = 5 steps
        assert_eq!(trainer.step, 5);
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
        assert!(result.is_ok());

        // 8 examples / 4 batch size = 2 steps per epoch
        // 2 steps * 5 epochs = 10 total steps
        assert_eq!(trainer.step, 10);
    }
}
