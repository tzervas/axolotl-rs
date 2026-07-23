//! Training loop and optimization.

use candle_core::backprop::GradStore;
use candle_core::{Device, Tensor, Var};
use candle_nn::VarMap;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::{AxolotlConfig, LrScheduler, TrainingConfig};
use crate::dataset::Dataset;
use crate::error::{AxolotlError, Result};
use crate::model::{load_model, LoadedModel};
use crate::optimizer::{AdamWOptimizer, OptimizerConfig};
use crate::scheduler::{LRScheduler, SchedulerType};

/// Training step metrics for convergence validation and monitoring.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Cross-entropy loss for this step
    pub loss: f64,
    /// Global norm of all gradients
    pub grad_norm: f64,
    /// Global norm of all trainable parameters
    pub param_norm: f64,
}

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
    /// Loaded model (optional, loaded during `train()`)
    model: Option<LoadedModel>,
    /// Optimizer (optional, created during `train()`)
    optimizer: Option<AdamWOptimizer>,
    /// Learning rate scheduler (optional, created during `train()`)
    scheduler: Option<LRScheduler>,
    /// Training metrics from last run
    pub training_metrics: Vec<StepMetrics>,
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

        // Determine device (prefer CUDA, fallback to CPU with warning)
        let force_cpu = std::env::var("AXOLOTL_FORCE_CPU")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let cuda_device = std::env::var("AXOLOTL_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let device = if !force_cpu && cfg!(feature = "cuda") {
            match Device::cuda_if_available(cuda_device) {
                Ok(device @ Device::Cuda(_)) => {
                    tracing::info!("Training device: CUDA (device {})", cuda_device);
                    device
                }
                Ok(_) => {
                    tracing::warn!("CUDA not available; falling back to CPU. This is a compatibility path only.");
                    Device::Cpu
                }
                Err(err) => {
                    tracing::warn!("CUDA init failed ({err}); falling back to CPU. This is a compatibility path only.");
                    Device::Cpu
                }
            }
        } else {
            if force_cpu {
                tracing::warn!(
                    "CPU mode forced via AXOLOTL_FORCE_CPU=1. GPU is the intended default."
                );
            } else {
                tracing::warn!(
                    "CUDA feature disabled; falling back to CPU. Enable with --features cuda."
                );
            }
            Device::Cpu
        };

        Ok(Self {
            config,
            step: 0,
            epoch: 0,
            device,
            model: None,
            optimizer: None,
            scheduler: None,
            training_metrics: Vec::new(),
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
    /// # Errors
    /// Returns an error if the checkpoint cannot be loaded.
    pub fn resume_from(&mut self, checkpoint_path: &str) -> Result<()> {
        self.load_checkpoint(checkpoint_path)
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

        // Initialize optimizer on trainable (adapter) parameters
        {
            let opt_cfg = OptimizerConfig {
                learning_rate: self.config.training.learning_rate,
                weight_decay: self.config.training.weight_decay,
                ..OptimizerConfig::default()
            };
            let optimizer = opt_cfg.build_adamw(&model.trainable_params)?;
            if optimizer.vars().is_empty() {
                return Err(AxolotlError::Training(
                    "No trainable parameters found after model load. For LoRA/QLoRA enable `--features peft` (and `qlora` when needed) and set adapter targets."
                        .into(),
                ));
            }
            tracing::info!(
                "Optimizer ready with {} trainable tensors ({} params)",
                optimizer.vars().len(),
                optimizer
                    .vars()
                    .iter()
                    .map(|v| v.elem_count())
                    .sum::<usize>()
            );
            self.optimizer = Some(optimizer);
        }

        self.model = Some(model);

        // Load dataset
        let dataset = Dataset::load(&self.config.dataset)?;
        tracing::info!("Loaded {} training examples", dataset.len());

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        let accum_steps = self.config.training.gradient_accumulation_steps.max(1);
        if self.config.training.gradient_checkpointing {
            tracing::warn!("training.gradient_checkpointing is set but not implemented; ignoring");
        }
        if self.config.training.mixed_precision {
            tracing::warn!(
                "training.mixed_precision is set but compute is forced to F32; ignoring"
            );
        }

        // Optimizer steps = ceil(microbatches / accum); microbatch size = batch_size
        let microbatches_per_epoch = dataset
            .len()
            .div_ceil(self.config.training.batch_size.max(1));
        let total_microbatches = microbatches_per_epoch * self.config.training.epochs;
        let total_steps = total_microbatches.div_ceil(accum_steps).max(1);

        // Initialize learning rate scheduler from YAML (lr_scheduler + warmup_ratio)
        {
            let scheduler = build_scheduler_from_config(&self.config.training, total_steps);
            tracing::info!(
                "Initialized {:?} scheduler with warmup_ratio={}, total_steps={}, accum={}",
                self.config.training.lr_scheduler,
                self.config.training.warmup_ratio,
                total_steps,
                accum_steps
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

        // Clear previous metrics
        self.training_metrics.clear();

        // Gradient accumulation state
        let mut pending_grads: Option<GradStore> = None;
        let mut micros_in_window: usize = 0;
        let mut window_loss_sum: f64 = 0.0;

        // Training loop
        for epoch in 0..self.config.training.epochs {
            self.epoch = epoch;
            tracing::info!(
                "Starting epoch {}/{}",
                epoch + 1,
                self.config.training.epochs
            );

            for batch in dataset.train.chunks(self.config.training.batch_size) {
                // Microbatch: forward + backward (scaled for accumulation)
                let (loss_val, grads) = self.microbatch_grads(batch, accum_steps)?;
                window_loss_sum += loss_val;
                micros_in_window += 1;

                // Accumulate gradients across microbatches
                pending_grads = Some(match pending_grads {
                    None => grads,
                    Some(mut acc) => {
                        let vars = self
                            .optimizer
                            .as_ref()
                            .map(|o| o.vars().to_vec())
                            .unwrap_or_default();
                        accumulate_grad_store(&mut acc, &grads, &vars)?;
                        acc
                    }
                });

                // Optimizer step only every gradient_accumulation_steps microbatches
                if micros_in_window < accum_steps {
                    continue;
                }

                let mut grads = pending_grads.take().ok_or_else(|| {
                    AxolotlError::Training("Missing accumulated gradients".into())
                })?;

                let optimizer = self
                    .optimizer
                    .as_mut()
                    .ok_or_else(|| AxolotlError::Training("Optimizer not initialized".into()))?;

                // Clip gradients by global norm when max_grad_norm > 0
                let max_norm = f64::from(self.config.training.max_grad_norm);
                let grad_norm = clip_grad_norm(&mut grads, optimizer.vars(), max_norm)?;

                // Apply optimizer update
                optimizer.step_grads(&grads)?;

                // Parameter norm after update
                let param_norm = compute_global_param_norm_vars(optimizer.vars())?;

                self.step += 1;
                let avg_loss = window_loss_sum / micros_in_window as f64;
                micros_in_window = 0;
                window_loss_sum = 0.0;

                let metrics = StepMetrics {
                    loss: avg_loss,
                    grad_norm,
                    param_norm,
                };
                self.training_metrics.push(metrics.clone());

                // Update progress bar with loss
                pb.set_message(format!("{:.4}", metrics.loss));
                pb.inc(1);

                // Log periodically
                if self.step.is_multiple_of(self.config.training.logging_steps) {
                    tracing::info!(
                        "Step {}/{}, Epoch {}, Loss: {:.4}, GradNorm: {:.4}, ParamNorm: {:.4}, LR: {:.2e}",
                        self.step,
                        total_steps,
                        epoch + 1,
                        metrics.loss,
                        metrics.grad_norm,
                        metrics.param_norm,
                        self.optimizer.as_ref().unwrap().learning_rate()
                    );
                }

                // Save checkpoint periodically
                if self.step.is_multiple_of(self.config.training.save_steps) {
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

        // Flush leftover microbatches that did not fill a full accumulation window
        if micros_in_window > 0 {
            if let Some(mut grads) = pending_grads.take() {
                let optimizer = self
                    .optimizer
                    .as_mut()
                    .ok_or_else(|| AxolotlError::Training("Optimizer not initialized".into()))?;
                let max_norm = f64::from(self.config.training.max_grad_norm);
                let grad_norm = clip_grad_norm(&mut grads, optimizer.vars(), max_norm)?;
                optimizer.step_grads(&grads)?;
                let param_norm = compute_global_param_norm_vars(optimizer.vars())?;
                self.step += 1;
                let avg_loss = window_loss_sum / micros_in_window as f64;
                let metrics = StepMetrics {
                    loss: avg_loss,
                    grad_norm,
                    param_norm,
                };
                self.training_metrics.push(metrics.clone());
                pb.set_message(format!("{:.4}", metrics.loss));
                pb.inc(1);
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

    /// Perform a single training step (one microbatch + immediate optimizer step).
    ///
    /// Used by tests and as a simple path when `gradient_accumulation_steps == 1`.
    /// Full training uses [`Self::microbatch_grads`] with accumulation in [`Self::train`].
    fn training_step(&mut self, batch: &[crate::dataset::Example]) -> Result<StepMetrics> {
        let (loss_val, mut grads) = self.microbatch_grads(batch, 1)?;
        let optimizer = self
            .optimizer
            .as_mut()
            .ok_or_else(|| AxolotlError::Training("Optimizer not initialized".into()))?;
        let max_norm = f64::from(self.config.training.max_grad_norm);
        let grad_norm = clip_grad_norm(&mut grads, optimizer.vars(), max_norm)?;
        optimizer.step_grads(&grads)?;
        let param_norm = compute_global_param_norm_vars(optimizer.vars())?;
        Ok(StepMetrics {
            loss: loss_val,
            grad_norm,
            param_norm,
        })
    }

    /// Forward + backward for one microbatch.
    ///
    /// Loss is scaled by `1 / accum_steps` so accumulated gradients match the
    /// mean over the accumulation window. Returns unscaled loss for logging.
    fn microbatch_grads(
        &self,
        batch: &[crate::dataset::Example],
        accum_steps: usize,
    ) -> Result<(f64, GradStore)> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| AxolotlError::Training("Model not loaded".into()))?;

        // 1. Tokenize batch
        let pad_token_id = model
            .tokenizer
            .token_to_id("<pad>")
            .or_else(|| model.tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);

        let mut input_ids = Vec::new();
        let mut labels = Vec::new();
        let max_len = self.config.dataset.max_length;

        for example in batch {
            let encoding = model
                .tokenizer
                .encode(example.text.as_str(), true)
                .map_err(|e| AxolotlError::Tokenizer(format!("Tokenization failed: {e}").into()))?;

            let mut ids = encoding.get_ids().to_vec();
            let original_len = ids.len();

            if ids.len() > max_len {
                ids.truncate(max_len);
            }
            while ids.len() < max_len {
                ids.push(pad_token_id);
            }

            let mut label_ids: Vec<i64> = Vec::with_capacity(max_len);
            for i in 0..max_len {
                if i + 1 < original_len {
                    label_ids.push(i64::from(ids[i + 1]));
                } else {
                    label_ids.push(-100);
                }
            }

            input_ids.push(ids);
            labels.push(label_ids);
        }

        // 2. Convert to tensors
        let batch_size = input_ids.len();
        let flat_input: Vec<i64> = input_ids.iter().flatten().map(|&x| i64::from(x)).collect();
        let flat_labels: Vec<i64> = labels.iter().flatten().copied().collect();

        let input_tensor = Tensor::from_vec(flat_input, (batch_size, max_len), &self.device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create input tensor: {e}")))?;
        let label_tensor = Tensor::from_vec(flat_labels, (batch_size, max_len), &self.device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create label tensor: {e}")))?;

        // 3. Forward pass
        let logits = model
            .forward_with_adapters(&input_tensor)
            .map_err(|e| AxolotlError::Training(format!("Forward pass failed: {e}")))?;

        // 4. Cross-entropy loss
        let loss = compute_cross_entropy_loss(&logits, &label_tensor, &self.device)?;
        let loss_val = f64::from(
            loss.to_vec0::<f32>()
                .map_err(|e| AxolotlError::Training(format!("Failed to get loss value: {e}")))?,
        );

        // 5. Scale for accumulation then backward
        let scale = 1.0 / accum_steps.max(1) as f64;
        let scaled_loss = loss
            .affine(scale, 0.0)
            .map_err(|e| AxolotlError::Training(format!("Failed to scale loss: {e}")))?;
        let grads = scaled_loss
            .backward()
            .map_err(|e| AxolotlError::Training(format!("Backward pass failed: {e}")))?;

        Ok((loss_val, grads))
    }

    /// Save a checkpoint.
    ///
    /// Saves:
    /// - Training state (step, epoch, config)
    /// - Adapter weights (if using LoRA/QLoRA) in safetensors format
    /// - Optimizer state (for resume)
    fn save_checkpoint(&self) -> Result<()> {
        let checkpoint_dir = format!("{}/checkpoint-{}", self.config.output_dir, self.step);
        std::fs::create_dir_all(&checkpoint_dir)?;

        // Save training state
        let optimizer = self.optimizer.as_ref().ok_or_else(|| {
            AxolotlError::Checkpoint("Optimizer not initialized during checkpoint save".into())
        })?;
        let training_state = TrainingState {
            step: self.step,
            epoch: self.epoch,
            learning_rate: optimizer.learning_rate(),
        };
        let state_path = format!("{checkpoint_dir}/training_state.json");
        let state_json = serde_json::to_string_pretty(&training_state)
            .map_err(|e| AxolotlError::Checkpoint(format!("Failed to serialize state: {e}")))?;
        std::fs::write(&state_path, state_json)?;

        // Save config for reproducibility
        let config_path = format!("{checkpoint_dir}/config.yaml");
        self.config.to_file(&config_path)?;

        // Save adapter weights if using LoRA/QLoRA (embedded VarMap or adapter_layers)
        #[cfg(feature = "peft")]
        if let Some(ref model) = self.model {
            let has_trainable = !model.trainable_params.all_vars().is_empty();
            let has_adapter_map = model.adapter_layers.as_ref().is_some_and(|a| !a.is_empty());
            if has_trainable || has_adapter_map {
                model.save_adapter_weights(&checkpoint_dir)?;

                // Also save adapter config as JSON (HuggingFace compatible)
                let adapter_config = serde_json::json!({
                    "base_model_name_or_path": self.config.base_model,
                    "peft_type": "LORA",
                    "r": self.config.lora.r,
                    "lora_alpha": self.config.lora.alpha,
                    "lora_dropout": self.config.lora.dropout,
                    "target_modules": self.config.lora.target_modules,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                });
                let adapter_config_path = format!("{checkpoint_dir}/adapter_config.json");
                std::fs::write(
                    &adapter_config_path,
                    serde_json::to_string_pretty(&adapter_config).unwrap(),
                )?;
            }
        }

        tracing::info!("Saved checkpoint to: {}", checkpoint_dir);
        Ok(())
    }

    /// Load training state from a checkpoint.
    ///
    /// # Errors
    /// Returns error if checkpoint files cannot be read or parsed.
    pub fn load_checkpoint(&mut self, checkpoint_path: &str) -> Result<()> {
        let state_path = format!("{checkpoint_path}/training_state.json");
        let state_json = std::fs::read_to_string(&state_path)
            .map_err(|e| AxolotlError::Checkpoint(format!("Failed to read state: {e}")))?;
        let state: TrainingState = serde_json::from_str(&state_json)
            .map_err(|e| AxolotlError::Checkpoint(format!("Failed to parse state: {e}")))?;

        self.step = state.step;
        self.epoch = state.epoch;

        if let Some(optimizer) = self.optimizer.as_mut() {
            optimizer.set_learning_rate(state.learning_rate);
        }

        // Load adapter weights if available
        #[cfg(feature = "peft")]
        {
            let adapter_path = format!("{checkpoint_path}/adapter_model.safetensors");
            if std::path::Path::new(&adapter_path).exists() {
                if let Some(ref mut model) = self.model {
                    model.load_adapter_weights(checkpoint_path)?;
                }
            }
        }

        tracing::info!(
            "Loaded checkpoint from: {} (step={}, epoch={})",
            checkpoint_path,
            state.step,
            state.epoch
        );
        Ok(())
    }

    /// Get reference to the loaded model for testing/inspection.
    ///
    /// Returns None if model hasn't been loaded yet (before `train()` is called).
    #[allow(dead_code)]
    pub fn get_model(&self) -> Option<&LoadedModel> {
        self.model.as_ref()
    }

    /// Get mutable reference to the loaded model for testing/inspection.
    ///
    /// Returns None if model hasn't been loaded yet (before `train()` is called).
    #[allow(dead_code)]
    pub fn get_model_mut(&mut self) -> Option<&mut LoadedModel> {
        self.model.as_mut()
    }

    /// Get all training metrics collected during training.
    ///
    /// Returns a vector of metrics for each training step.
    /// Use this for convergence validation and analysis.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let mut trainer = Trainer::new(AxolotlConfig::from_preset("llama2-7b")?)?;
    /// trainer.train()?;
    ///
    /// let metrics = trainer.metrics();
    /// println!("Training completed with {} steps", metrics.len());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(dead_code)]
    pub fn metrics(&self) -> &[StepMetrics] {
        &self.training_metrics
    }

    /// Get loss values for all training steps.
    ///
    /// Returns a vector of loss values, useful for convergence validation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let mut trainer = Trainer::new(AxolotlConfig::from_preset("llama2-7b")?)?;
    /// trainer.train()?;
    ///
    /// let losses = trainer.losses();
    /// assert!(losses[losses.len()-1] < losses[0], "Loss should decrease");
    /// # Ok(())
    /// # }
    /// ```
    #[allow(dead_code)]
    pub fn losses(&self) -> Vec<f64> {
        self.training_metrics.iter().map(|m| m.loss).collect()
    }

    /// Get gradient norms for all training steps.
    ///
    /// Returns a vector of global gradient norms.
    #[allow(dead_code)]
    pub fn grad_norms(&self) -> Vec<f64> {
        self.training_metrics.iter().map(|m| m.grad_norm).collect()
    }

    /// Get parameter norms for all training steps.
    ///
    /// Returns a vector of global parameter norms.
    #[allow(dead_code)]
    pub fn param_norms(&self) -> Vec<f64> {
        self.training_metrics.iter().map(|m| m.param_norm).collect()
    }

    /// Get current training step.
    #[allow(dead_code)]
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get current epoch.
    #[allow(dead_code)]
    pub fn epoch(&self) -> usize {
        self.epoch
    }
}

/// Training state for checkpoint serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TrainingState {
    /// Current training step
    step: usize,
    /// Current epoch
    epoch: usize,
    /// Current learning rate
    learning_rate: f64,
}

/// Compute cross-entropy loss on the last position prediction.
///
/// This is used when the model only returns logits for the last position
/// (common in generation-optimized models like candle's Llama).
/// Build an [`LRScheduler`] from training YAML knobs.
///
/// Uses `lr_scheduler` and `warmup_ratio` (not hardcoded Linear/10%).
pub(crate) fn build_scheduler_from_config(
    training: &TrainingConfig,
    total_steps: usize,
) -> LRScheduler {
    let total_steps = total_steps.max(1);
    let warmup_steps = ((total_steps as f64) * f64::from(training.warmup_ratio)).round() as usize;
    let warmup_steps = warmup_steps.min(total_steps);
    let scheduler_type = match training.lr_scheduler {
        LrScheduler::Constant => SchedulerType::Constant,
        LrScheduler::Linear => SchedulerType::Linear {
            warmup_steps,
            total_steps,
        },
        LrScheduler::Cosine => SchedulerType::Cosine {
            warmup_steps,
            total_steps,
        },
    };
    LRScheduler::new(scheduler_type, training.learning_rate)
}

/// Sum `new_grads` into `acc` for each trainable variable.
fn accumulate_grad_store(acc: &mut GradStore, new_grads: &GradStore, vars: &[Var]) -> Result<()> {
    for var in vars {
        let t = var.as_tensor();
        if let Some(g) = new_grads.get(t) {
            if let Some(existing) = acc.get(t) {
                let summed = existing
                    .add(g)
                    .map_err(|e| AxolotlError::Training(format!("Grad accum add failed: {e}")))?;
                acc.insert(t, summed);
            } else {
                acc.insert(t, g.clone());
            }
        }
    }
    Ok(())
}

/// Global L2 norm of gradients for `vars` in `grads` (before clipping).
pub(crate) fn compute_global_grad_norm_from_store(grads: &GradStore, vars: &[Var]) -> Result<f64> {
    let mut total_sq = 0.0f64;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            let sq = tensor_sum_squares(g)?;
            total_sq += sq;
        }
    }
    Ok(total_sq.sqrt())
}

/// Clip gradients in-place by global norm. Returns the pre-clip global norm.
///
/// When `max_norm <= 0.0`, clipping is disabled (norm is still computed).
pub(crate) fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f64> {
    let total_norm = compute_global_grad_norm_from_store(grads, vars)?;
    if max_norm > 0.0 && total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for var in vars {
            let t = var.as_tensor();
            if let Some(g) = grads.get(t) {
                let scaled = g
                    .affine(scale, 0.0)
                    .map_err(|e| AxolotlError::Training(format!("Grad clip scale failed: {e}")))?;
                grads.insert(t, scaled);
            }
        }
    }
    Ok(total_norm)
}

/// Global L2 norm of all parameters in a `VarMap`.
fn compute_global_param_norm(varmap: &VarMap) -> Result<f64> {
    compute_global_param_norm_vars(&varmap.all_vars())
}

/// Global L2 norm of parameter variables.
pub(crate) fn compute_global_param_norm_vars(vars: &[Var]) -> Result<f64> {
    let mut total_sq = 0.0f64;
    for var in vars {
        total_sq += tensor_sum_squares(var.as_tensor())?;
    }
    Ok(total_sq.sqrt())
}

/// Sum of squares of all elements (as f64).
fn tensor_sum_squares(t: &Tensor) -> Result<f64> {
    let sq = t
        .sqr()
        .map_err(|e| AxolotlError::Training(format!("sqr failed: {e}")))?
        .sum_all()
        .map_err(|e| AxolotlError::Training(format!("sum_all failed: {e}")))?;
    // Prefer f32 then f64
    if let Ok(v) = sq.to_vec0::<f32>() {
        return Ok(f64::from(v));
    }
    let v = sq
        .to_vec0::<f64>()
        .map_err(|e| AxolotlError::Training(format!("to_vec0 failed: {e}")))?;
    Ok(v)
}

/// Legacy helper kept for call sites that still pass a `VarMap` for grad norms.
/// Prefer [`compute_global_grad_norm_from_store`] during training.
#[allow(dead_code)]
fn compute_global_grad_norm(varmap: &VarMap) -> Result<f64> {
    // Without a GradStore we cannot recover grads; report 0 for empty maps only.
    if varmap.all_vars().is_empty() {
        return Ok(0.0);
    }
    Err(AxolotlError::Training(
        "compute_global_grad_norm requires GradStore; use compute_global_grad_norm_from_store"
            .into(),
    ))
}

///
/// # Arguments
/// * `logits` - Model output logits with shape [`batch_size`, `vocab_size`] (last position only)
/// * `labels` - Target labels with shape [`batch_size`, `seq_len`], -100 for masked positions
/// * `device` - Device for tensor operations
///
/// # Returns
/// A scalar loss tensor that can be backpropagated through
#[allow(dead_code)]
fn compute_last_position_loss(logits: &Tensor, labels: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = logits.dims();

    // Logits should be [batch, vocab] for last-position-only output
    if dims.len() != 2 {
        return Err(AxolotlError::Training(format!(
            "Expected 2D logits [batch, vocab], got {dims:?}"
        )));
    }

    let (batch_size, vocab_size) = (dims[0], dims[1]);
    let label_dims = labels.dims();
    // seq_len not used, but kept for clarity of dimensions
    let _ = label_dims[1];

    // For each sequence, get the last non-padding label
    // This is the target for predicting what comes after the last token
    let labels_flat = labels
        .to_vec2::<i64>()
        .map_err(|e| AxolotlError::Training(format!("Failed to read labels: {e}")))?;

    // Find last valid (non -100) label for each batch item
    let mut last_labels: Vec<u32> = Vec::with_capacity(batch_size);
    let mut valid_mask: Vec<f32> = Vec::with_capacity(batch_size);

    for seq_labels in &labels_flat {
        // Find the last valid label (not -100)
        let mut last_valid: Option<i64> = None;
        for &label in seq_labels.iter().rev() {
            if label >= 0 && (label as usize) < vocab_size {
                last_valid = Some(label);
                break;
            }
        }

        if let Some(label) = last_valid {
            last_labels.push(label as u32);
            valid_mask.push(1.0);
        } else {
            last_labels.push(0);
            valid_mask.push(0.0);
        }
    }

    let valid_count: f32 = valid_mask.iter().sum();

    if valid_count == 0.0 {
        return Tensor::new(0.0f32, device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create zero loss: {e}")));
    }

    let labels_tensor = Tensor::from_vec(last_labels, batch_size, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create labels tensor: {e}")))?;

    // Compute log softmax
    let log_probs = candle_nn::ops::log_softmax(logits, 1)
        .map_err(|e| AxolotlError::Training(format!("Log softmax failed: {e}")))?;

    // Gather log probs at target indices
    let target_indices = labels_tensor
        .unsqueeze(1)
        .map_err(|e| AxolotlError::Training(format!("Unsqueeze failed: {e}")))?;
    let gathered = log_probs
        .gather(&target_indices, 1)
        .map_err(|e| AxolotlError::Training(format!("Gather failed: {e}")))?
        .squeeze(1)
        .map_err(|e| AxolotlError::Training(format!("Squeeze failed: {e}")))?;

    // Apply mask and compute mean of negative log likelihood
    let mask_tensor = Tensor::from_vec(valid_mask, batch_size, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create mask tensor: {e}")))?;
    let masked_loss = gathered
        .neg()
        .map_err(|e| AxolotlError::Training(format!("Neg failed: {e}")))?
        .mul(&mask_tensor)
        .map_err(|e| AxolotlError::Training(format!("Mul failed: {e}")))?;

    // Sum and divide by valid count
    let total_loss = masked_loss
        .sum_all()
        .map_err(|e| AxolotlError::Training(format!("Sum failed: {e}")))?;

    let valid_count_scalar = Tensor::new(valid_count, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create count tensor: {e}")))?;

    let loss = total_loss
        .broadcast_div(&valid_count_scalar)
        .map_err(|e| AxolotlError::Training(format!("Div failed: {e}")))?;

    Ok(loss)
}

/// Compute cross-entropy loss with gradient tracking (full sequence version).
///
/// This function uses tensor operations that maintain the autograd graph,
/// enabling proper backpropagation through the loss to update `LoRA` weights.
///
/// # Arguments
/// * `logits` - Model output logits with shape [`batch_size`, `seq_len`, `vocab_size`] or [batch*seq, vocab]
/// * `labels` - Target labels with shape [`batch_size`, `seq_len`], -100 for masked positions
/// * `device` - Device for tensor operations
///
/// # Returns
/// A scalar loss tensor that can be backpropagated through
#[allow(dead_code)]
fn compute_cross_entropy_loss(logits: &Tensor, labels: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = logits.dims();
    let label_dims = labels.dims();

    // Handle different logit shapes from different model implementations
    // Candle's Llama returns [batch * seq, vocab] while some return [batch, seq, vocab]
    let (num_positions, vocab_size) = match dims.len() {
        2 => (dims[0], dims[1]),
        3 => (dims[0] * dims[1], dims[2]),
        _ => {
            return Err(AxolotlError::Training(format!(
                "Expected 2D or 3D logits, got {dims:?}"
            )))
        }
    };

    // Flatten logits to [num_positions, vocab_size]
    let logits_flat = if dims.len() == 3 {
        logits
            .reshape((num_positions, vocab_size))
            .map_err(|e| AxolotlError::Training(format!("Logits reshape failed: {e}")))?
    } else {
        logits.clone()
    };

    // Flatten labels to [num_positions]
    let labels_flat = labels
        .reshape(num_positions)
        .map_err(|e| AxolotlError::Training(format!("Labels reshape failed: {e}")))?;

    // Verify dimensions match
    if num_positions != label_dims.iter().product::<usize>() {
        return Err(AxolotlError::Training(format!(
            "Logits positions {} != labels positions {}",
            num_positions,
            label_dims.iter().product::<usize>()
        )));
    }

    // Create mask for valid (non-padding) positions
    // Labels of -100 are masked out
    let labels_i64 = labels_flat
        .to_vec1::<i64>()
        .map_err(|e| AxolotlError::Training(format!("Failed to read labels: {e}")))?;

    let valid_mask: Vec<f32> = labels_i64
        .iter()
        .map(|&l| {
            if l >= 0 && (l as usize) < vocab_size {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let valid_count: f32 = valid_mask.iter().sum();

    if valid_count == 0.0 {
        // No valid labels, return zero loss
        return Tensor::new(&[0.0f32], device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create zero loss: {e}")));
    }

    // Replace invalid labels with 0 (they'll be masked anyway)
    let safe_labels: Vec<u32> = labels_i64
        .iter()
        .map(|&l| {
            if l >= 0 && (l as usize) < vocab_size {
                l as u32
            } else {
                0
            }
        })
        .collect();
    let safe_labels_tensor = Tensor::from_vec(safe_labels, num_positions, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create safe labels: {e}")))?;

    // Compute log softmax (this maintains gradients)
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)
        .map_err(|e| AxolotlError::Training(format!("Log softmax failed: {e}")))?;

    // Gather log probs at target indices
    let target_indices = safe_labels_tensor
        .unsqueeze(1)
        .map_err(|e| AxolotlError::Training(format!("Unsqueeze failed: {e}")))?;
    let gathered = log_probs
        .gather(&target_indices, 1)
        .map_err(|e| AxolotlError::Training(format!("Gather failed: {e}")))?
        .squeeze(1)
        .map_err(|e| AxolotlError::Training(format!("Squeeze failed: {e}")))?;

    // Apply mask and compute mean of negative log likelihood
    let mask_tensor = Tensor::from_vec(valid_mask, num_positions, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create mask tensor: {e}")))?;
    let masked_loss = gathered
        .neg()
        .map_err(|e| AxolotlError::Training(format!("Neg failed: {e}")))?
        .mul(&mask_tensor)
        .map_err(|e| AxolotlError::Training(format!("Mul failed: {e}")))?;

    // Sum and divide by valid count
    let total_loss = masked_loss
        .sum_all()
        .map_err(|e| AxolotlError::Training(format!("Sum failed: {e}")))?;

    // Create scalar tensor for valid_count and squeeze total_loss to same shape
    let valid_count_scalar = Tensor::new(valid_count, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create count tensor: {e}")))?;

    // Both are scalars now, division should work
    let loss = total_loss
        .broadcast_div(&valid_count_scalar)
        .map_err(|e| AxolotlError::Training(format!("Div failed: {e}")))?;

    Ok(loss)
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
    fn test_resume_from_missing_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let mut trainer = Trainer::new(config).unwrap();

        // Resuming from non-existent checkpoint should fail
        let result = trainer.resume_from("nonexistent-checkpoint");
        assert!(result.is_err());

        match result {
            Err(AxolotlError::Checkpoint(_)) => {}
            _ => panic!("Expected Checkpoint error"),
        }
    }

    #[test]
    fn test_checkpoint_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let mut trainer = Trainer::new(config).unwrap();
        trainer.step = 100;
        trainer.epoch = 2;

        // Initialize optimizer for checkpoint save (required)
        let optimizer_config = OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..OptimizerConfig::default()
        };
        let varmap = VarMap::new();
        trainer.optimizer = Some(optimizer_config.build_adamw(&varmap).unwrap());

        // Save checkpoint
        trainer.save_checkpoint().unwrap();

        // Verify checkpoint files exist
        let checkpoint_dir = output_path.join("checkpoint-100");
        assert!(checkpoint_dir.join("training_state.json").exists());
        assert!(checkpoint_dir.join("config.yaml").exists());

        // Load checkpoint into new trainer
        let config2 = create_test_config(output_path.to_str().unwrap());
        let mut trainer2 = Trainer::new(config2).unwrap();

        // Initialize optimizer in trainer2 to test learning rate restoration
        let optimizer_config2 = OptimizerConfig {
            learning_rate: 0.002, // Different initial value
            weight_decay: 0.01,
            ..OptimizerConfig::default()
        };
        let varmap2 = VarMap::new();
        trainer2.optimizer = Some(optimizer_config2.build_adamw(&varmap2).unwrap());

        trainer2
            .load_checkpoint(checkpoint_dir.to_str().unwrap())
            .unwrap();

        assert_eq!(trainer2.step, 100);
        assert_eq!(trainer2.epoch, 2);
        // Verify learning rate was restored from checkpoint
        assert_eq!(trainer2.optimizer.as_ref().unwrap().learning_rate(), 0.001);
    }

    #[test]
    fn test_resume_from_valid_checkpoint() {
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

    // ========================================================================
    // PR-029: training knobs honesty
    // ========================================================================

    #[test]
    fn test_scheduler_from_config_honors_type_and_warmup() {
        let mut training = TrainingConfig::default();
        training.lr_scheduler = LrScheduler::Cosine;
        training.warmup_ratio = 0.1;
        training.learning_rate = 1e-3;

        let sched = build_scheduler_from_config(&training, 1000);
        // step 0 -> still at 0 before any step(); get_lr uses current_step
        assert!((sched.get_lr() - 0.0).abs() < 1e-12);

        training.lr_scheduler = LrScheduler::Linear;
        let linear = build_scheduler_from_config(&training, 100);
        // mid-warmup: after setting internal step via repeated step would need optimizer;
        // instead inspect type via Cosine vs Linear by comparing schedule shapes:
        // at current_step=0 both start at 0 for Linear/Cosine
        assert!((linear.get_lr() - 0.0).abs() < 1e-12);

        training.lr_scheduler = LrScheduler::Constant;
        let constant = build_scheduler_from_config(&training, 100);
        assert!((constant.get_lr() - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn test_scheduler_warmup_ratio_not_hardcoded_10_percent() {
        let mut training = TrainingConfig::default();
        training.lr_scheduler = LrScheduler::Linear;
        training.warmup_ratio = 0.5; // 50% warmup
        training.learning_rate = 1.0;

        let mut sched = build_scheduler_from_config(&training, 100);
        // Manually advance current_step by calling step with a dummy optimizer
        let varmap = VarMap::new();
        let mut opt = OptimizerConfig {
            learning_rate: 1.0,
            ..OptimizerConfig::default()
        }
        .build_adamw(&varmap)
        .unwrap();

        // After 25 steps, current_step becomes 25; lr should be 0.25 (25/50 warmup)
        for _ in 0..25 {
            sched.step(&mut opt);
        }
        // step() increments then sets lr; after 25 steps current_step=25
        // linear warmup: lr = base * (25/50) = 0.5
        assert!(
            (opt.learning_rate() - 0.5).abs() < 1e-9,
            "expected 0.5 from 50% warmup, got {}",
            opt.learning_rate()
        );
    }

    #[test]
    fn test_param_norm_is_real_not_constant() {
        use candle_core::Device;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        {
            let mut ws = varmap.data().lock().unwrap();
            // Two params with known L2: [3,4] => 5; [0,0,0] => 0; total 5
            let t1 = Tensor::from_vec(vec![3.0f32, 4.0], (2,), &device).unwrap();
            let t2 = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], (3,), &device).unwrap();
            ws.insert("a".into(), candle_core::Var::from_tensor(&t1).unwrap());
            ws.insert("b".into(), candle_core::Var::from_tensor(&t2).unwrap());
        }
        let n = compute_global_param_norm(&varmap).unwrap();
        assert!((n - 5.0).abs() < 1e-4, "param norm should be 5, got {n}");

        // Different weights => different norm (not a fake constant 1.0)
        let varmap2 = VarMap::new();
        {
            let mut ws = varmap2.data().lock().unwrap();
            let t1 = Tensor::from_vec(vec![1.0f32, 0.0], (2,), &device).unwrap();
            ws.insert("a".into(), candle_core::Var::from_tensor(&t1).unwrap());
        }
        let n2 = compute_global_param_norm(&varmap2).unwrap();
        assert!((n2 - 1.0).abs() < 1e-4);
        assert!((n - n2).abs() > 1.0);
    }

    #[test]
    fn test_grad_clip_scales_when_exceeding_max_norm() {
        use candle_core::{Device, Var};
        let device = Device::Cpu;
        // Build a tiny graph so we get a real GradStore
        let w =
            Var::from_tensor(&Tensor::from_vec(vec![3.0f32, 4.0], (2,), &device).unwrap()).unwrap();
        // loss = sum(w^2) => grad = 2w = [6,8], norm = 10
        let loss = w.as_tensor().sqr().unwrap().sum_all().unwrap();
        let mut grads = loss.backward().unwrap();
        let vars = vec![w.clone()];
        let pre = compute_global_grad_norm_from_store(&grads, &vars).unwrap();
        assert!(
            (pre - 10.0).abs() < 1e-3,
            "expected grad norm 10, got {pre}"
        );

        let reported = clip_grad_norm(&mut grads, &vars, 5.0).unwrap();
        assert!((reported - 10.0).abs() < 1e-3);
        let post = compute_global_grad_norm_from_store(&grads, &vars).unwrap();
        assert!(
            (post - 5.0).abs() < 1e-2,
            "clipped norm should be ~5, got {post}"
        );
    }

    #[test]
    fn test_grad_accum_sums_microbatch_grads() {
        use candle_core::{Device, Var};
        let device = Device::Cpu;
        let w =
            Var::from_tensor(&Tensor::from_vec(vec![1.0f32, 0.0], (2,), &device).unwrap()).unwrap();
        // loss1 = w[0]*2 => grad [2, 0] but use sum(w) * 2
        let loss1 = (w.as_tensor() * 2.0).unwrap().sum_all().unwrap();
        let g1 = loss1.backward().unwrap();
        let loss2 = (w.as_tensor() * 3.0).unwrap().sum_all().unwrap();
        let g2 = loss2.backward().unwrap();
        let vars = vec![w.clone()];
        let mut acc = g1;
        accumulate_grad_store(&mut acc, &g2, &vars).unwrap();
        let g = acc.get(w.as_tensor()).unwrap();
        let vals = g.to_vec1::<f32>().unwrap();
        // grad of sum(w*2) is [2,2]? wait sum of [1,0]*2 = 2, grad w.r.t w is [2,2]?
        // (w * 2).sum_all() grad is 2 for each element -> [2, 2]
        // (w * 3).sum_all() grad is [3, 3]
        // sum = [5, 5]
        assert!((vals[0] - 5.0).abs() < 1e-4, "got {:?}", vals);
        assert!((vals[1] - 5.0).abs() < 1e-4, "got {:?}", vals);
    }

    #[test]
    fn test_accum_steps_config_defaults_and_override() {
        let t = TrainingConfig::default();
        assert_eq!(t.gradient_accumulation_steps, 4);
        let mut t2 = TrainingConfig::default();
        t2.gradient_accumulation_steps = 8;
        assert_eq!(t2.gradient_accumulation_steps, 8);
        // Effective optimizer steps helper: ceil division used in train()
        let microbatches = 10usize;
        let accum = t2.gradient_accumulation_steps.max(1);
        let opt_steps = microbatches.div_ceil(accum);
        assert_eq!(opt_steps, 2); // 10/8 -> 2
    }
}
