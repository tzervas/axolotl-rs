//! Model loading and adapter merging.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::config::{AdapterType, AxolotlConfig};
use crate::error::{AxolotlError, Result};

#[cfg(feature = "peft")]
use peft_rs::{LoraConfig as PeftLoraConfig, LoraLayer, SaveLoad};

#[cfg(feature = "qlora")]
use qlora_rs::{QLoraConfig, QuantizedLinear};

#[cfg(feature = "peft")]
use crate::lora_llama::LoraLlama;

#[cfg(all(feature = "peft", feature = "qlora"))]
use super::qlora_llama::{prepare_for_qlora_training, QLoraLlama};

// Additional imports for tests
#[cfg(test)]
use crate::config::{DatasetConfig, LoraSettings, QuantType, QuantizationSettings, TrainingConfig};

/// Loaded model with configuration.
pub struct LoadedModel {
    /// Model weights and forward pass
    pub model: Box<dyn Module>,
    /// Tokenizer
    pub tokenizer: tokenizers::Tokenizer,
    /// Device where model is loaded
    #[allow(dead_code)]
    pub device: Device,
    /// Model dtype
    #[allow(dead_code)]
    pub dtype: DType,
    /// Adapter layers (if using LoRA/QLoRA)
    #[allow(dead_code)]
    pub adapter_layers: Option<AdapterLayers>,
    /// Trainable parameters (`LoRA` weights)
    pub trainable_params: VarMap,
}

/// Container for adapter layers organized by module name.
#[derive(Default)]
pub struct AdapterLayers {
    /// `LoRA` layers keyed by module path (e.g., "`model.layers.0.self_attn.q_proj`")
    #[cfg(feature = "peft")]
    pub lora_layers: HashMap<String, LoraLayer>,
    /// QLoRA layers keyed by module path
    #[cfg(feature = "qlora")]
    pub qlora_layers: HashMap<String, QuantizedLinear>,
    /// Whether this is a `QLoRA` model (quantized base)
    #[allow(dead_code)]
    pub is_quantized: bool,
}

#[cfg(not(feature = "peft"))]
#[allow(dead_code)]
impl AdapterLayers {
    /// Placeholder when peft feature is disabled
    pub fn lora_layers(&self) -> &HashMap<String, ()> {
        static EMPTY: std::sync::OnceLock<HashMap<String, ()>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }
}

#[allow(dead_code)]
impl AdapterLayers {
    /// Create new adapter layers container.
    #[must_use]
    pub fn new(is_quantized: bool) -> Self {
        Self {
            #[cfg(feature = "peft")]
            lora_layers: HashMap::new(),
            #[cfg(feature = "qlora")]
            qlora_layers: HashMap::new(),
            is_quantized,
        }
    }

    /// Get the number of adapter layers.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(feature = "qlora")]
        if self.is_quantized {
            return self.qlora_layers.len();
        }
        #[cfg(feature = "peft")]
        return self.lora_layers.len();
        #[cfg(not(feature = "peft"))]
        0
    }

    /// Check if there are no adapter layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl LoadedModel {
    /// Run forward pass on input tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model
            .forward(input_ids)
            .map_err(|e| AxolotlError::Model(format!("Forward pass failed: {e}")))
    }

    /// Run forward pass with adapter layers.
    ///
    /// **IMPORTANT**: Current implementation does NOT properly integrate adapters.
    /// `LoRA` adapters need to be injected at each attention/MLP layer, not applied
    /// post-hoc to logits. This requires custom model architecture (`LoraLlama`).
    ///
    /// For now, this returns base model output. Gradient flow is maintained through
    /// the trainable `LoRA` parameters in `trainable_params` `VarMap`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_with_adapters(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Get base model output (logits for all positions)
        let logits = self.forward(input_ids)?;

        // TODO: Implement proper per-layer LoRA injection via LoraLlama
        // Current approach: Return base logits
        // This allows testing of training loop, loss computation, and optimizer
        // even without proper LoRA integration

        tracing::trace!("Forward pass complete (base model only, LoRA not integrated yet)");

        Ok(logits)
    }

    /// Get trainable parameters for optimizer.
    ///
    /// Returns only the `LoRA` A/B matrices, not the frozen base model weights.
    #[must_use]
    #[allow(dead_code)]
    pub fn trainable_tensors(&self) -> Vec<candle_core::Var> {
        self.trainable_params.all_vars()
    }

    /// Count trainable parameters.
    #[must_use]
    #[allow(dead_code)]
    pub fn trainable_param_count(&self) -> usize {
        self.trainable_tensors()
            .iter()
            .map(|v| v.elem_count())
            .sum()
    }

    /// Save adapter weights to safetensors.
    ///
    /// Supports both:
    /// - Separate `adapter_layers` map (post-hoc LoRA wrap)
    /// - Embedded adapters in `LoraLlama`/`QLoraLlama` via `trainable_params` `VarMap`
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails or no adapter tensors exist.
    #[cfg(feature = "peft")]
    pub fn save_adapter_weights<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let dir = path.as_ref();
        std::fs::create_dir_all(dir)?;
        let weights_path = dir.join("adapter_model.safetensors");

        if let Some(adapter_layers) = &self.adapter_layers {
            if !adapter_layers.lora_layers.is_empty() {
                let mut all_tensors: Vec<(String, Tensor)> = Vec::new();
                for (name, layer) in &adapter_layers.lora_layers {
                    let state = layer.state_dict().map_err(|e| {
                        AxolotlError::Checkpoint(format!(
                            "Failed to get state dict for {name}: {e}"
                        ))
                    })?;
                    for (key, tensor) in state {
                        all_tensors.push((format!("{name}.{key}"), tensor));
                    }
                }
                let tensors_ref: Vec<(&str, Tensor)> = all_tensors
                    .iter()
                    .map(|(name, tensor)| (name.as_str(), tensor.clone()))
                    .collect();
                safetensors::tensor::serialize_to_file(tensors_ref, None, &weights_path).map_err(
                    |e| AxolotlError::Checkpoint(format!("Failed to save adapter: {e}")),
                )?;
                tracing::info!(
                    "Saved {} adapter layers to {:?}",
                    adapter_layers.lora_layers.len(),
                    dir
                );
                return Ok(());
            }
        }

        // Embedded path: save LoRA A/B (and any other trainable adapter tensors) from VarMap
        if self.trainable_params.all_vars().is_empty() {
            return Err(AxolotlError::Model(
                "No adapter weights to save (empty trainable VarMap and no adapter_layers)".into(),
            ));
        }
        self.trainable_params
            .save(&weights_path)
            .map_err(|e| AxolotlError::Checkpoint(format!("Failed to save adapter VarMap: {e}")))?;
        tracing::info!(
            "Saved {} trainable adapter tensors from VarMap to {:?}",
            self.trainable_params.all_vars().len(),
            dir
        );
        Ok(())
    }

    /// Load adapter weights from safetensors and **apply** them to the model.
    ///
    /// - Embedded adapters: values are written into `trainable_params` by name
    /// - Separate `adapter_layers`: tensors are applied via `LoraLayer::load_state_dict`
    ///
    /// # Errors
    ///
    /// Returns an error if loading or applying fails.
    #[cfg(feature = "peft")]
    pub fn load_adapter_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let dir = path.as_ref();
        let weights_path = dir.join("adapter_model.safetensors");
        if !weights_path.exists() {
            return Err(AxolotlError::Checkpoint(format!(
                "adapter_model.safetensors not found at {}",
                weights_path.display()
            )));
        }

        let tensors = candle_core::safetensors::load(&weights_path, &self.device)
            .map_err(|e| AxolotlError::Checkpoint(format!("Failed to load adapter: {e}")))?;

        if tensors.is_empty() {
            return Err(AxolotlError::Checkpoint(
                "Loaded adapter file contained zero tensors".into(),
            ));
        }

        if let Some(adapter_layers) = &mut self.adapter_layers {
            if !adapter_layers.lora_layers.is_empty() {
                // Group flat keys `{module}.lora_a.weight` back into per-layer state dicts
                for (name, layer) in adapter_layers.lora_layers.iter_mut() {
                    let prefix = format!("{name}.");
                    let mut state = HashMap::new();
                    for (key, tensor) in &tensors {
                        if let Some(suffix) = key.strip_prefix(&prefix) {
                            state.insert(suffix.to_string(), tensor.clone());
                        }
                    }
                    if !state.is_empty() {
                        layer.load_state_dict(state).map_err(|e| {
                            AxolotlError::Checkpoint(format!(
                                "Failed to apply adapter tensors to {name}: {e}"
                            ))
                        })?;
                    }
                }
                tracing::info!(
                    "Applied adapter tensors to {} layers from {:?}",
                    adapter_layers.lora_layers.len(),
                    dir
                );
                return Ok(());
            }
        }

        // Embedded path: apply into VarMap in-place (only existing keys are updated)
        self.trainable_params.load(&weights_path).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to apply adapter VarMap: {e}"))
        })?;
        tracing::info!(
            "Applied {} adapter tensors into trainable VarMap from {:?}",
            tensors.len(),
            dir
        );
        Ok(())
    }

    /// Capture current `LoRA` weight matrices for gradient flow verification.
    ///
    /// Returns a `HashMap` of module name to (`A_matrix`, `B_matrix`) flattened f32 values.
    /// Reads real values from `trainable_params` (embedded) and/or `adapter_layers`.
    #[cfg(feature = "peft")]
    pub fn capture_lora_weights(
        &self,
    ) -> Result<std::collections::HashMap<String, (Vec<f32>, Vec<f32>)>> {
        use std::collections::HashMap;

        let mut paired: HashMap<String, (Option<Vec<f32>>, Option<Vec<f32>>)> = HashMap::new();

        // From VarMap (embedded LoraLlama / QLoraLlama)
        {
            let data = self
                .trainable_params
                .data()
                .lock()
                .map_err(|e| AxolotlError::Model(format!("VarMap lock poisoned: {e}")))?;
            for (name, var) in data.iter() {
                let (module, is_a) = if let Some(m) = name.strip_suffix(".lora_a.weight") {
                    (m, true)
                } else if let Some(m) = name.strip_suffix(".lora_A.weight") {
                    (m, true)
                } else if let Some(m) = name.strip_suffix(".lora_b.weight") {
                    (m, false)
                } else if let Some(m) = name.strip_suffix(".lora_B.weight") {
                    (m, false)
                } else {
                    continue;
                };
                let vals = var
                    .as_tensor()
                    .flatten_all()
                    .map_err(|e| AxolotlError::Model(format!("flatten {name}: {e}")))?
                    .to_vec1::<f32>()
                    .map_err(|e| AxolotlError::Model(format!("to_vec1 {name}: {e}")))?;
                let entry = paired.entry(module.to_string()).or_insert((None, None));
                if is_a {
                    entry.0 = Some(vals);
                } else {
                    entry.1 = Some(vals);
                }
            }
        }

        // From separate adapter layers
        if let Some(adapter_layers) = &self.adapter_layers {
            for (module_name, layer) in &adapter_layers.lora_layers {
                if let Ok(state) = layer.state_dict() {
                    let a = state
                        .get("lora_a.weight")
                        .and_then(|t| t.flatten_all().ok())
                        .and_then(|t| t.to_vec1::<f32>().ok());
                    let b = state
                        .get("lora_b.weight")
                        .and_then(|t| t.flatten_all().ok())
                        .and_then(|t| t.to_vec1::<f32>().ok());
                    if a.is_some() || b.is_some() {
                        paired.insert(module_name.clone(), (a, b));
                    }
                }
            }
        }

        let mut weights = HashMap::new();
        for (module, (a, b)) in paired {
            weights.insert(module, (a.unwrap_or_default(), b.unwrap_or_default()));
        }
        Ok(weights)
    }

    /// Verify that `LoRA` weights have been updated after a training step.
    ///
    /// Compares captured weights with current weights to detect if gradients
    /// flowed through the `LoRA` layers and were applied by the optimizer.
    #[cfg(feature = "peft")]
    pub fn verify_lora_weight_updates(
        &self,
        initial_weights: &std::collections::HashMap<String, (Vec<f32>, Vec<f32>)>,
    ) -> Result<bool> {
        if initial_weights.is_empty() {
            return Ok(false);
        }

        let current_weights = self.capture_lora_weights()?;

        // Check if any weights changed
        for (module_name, (initial_a, initial_b)) in initial_weights {
            if let Some((current_a, current_b)) = current_weights.get(module_name) {
                // Calculate change magnitude for A matrix
                let a_changed = if !initial_a.is_empty() && !current_a.is_empty() {
                    let diff: f64 = initial_a
                        .iter()
                        .zip(current_a.iter())
                        .map(|(i, c)| f64::from(i - c).abs())
                        .sum();
                    diff > 0.0
                } else {
                    false
                };

                // Calculate change magnitude for B matrix
                let b_changed = if !initial_b.is_empty() && !current_b.is_empty() {
                    let diff: f64 = initial_b
                        .iter()
                        .zip(current_b.iter())
                        .map(|(i, c)| f64::from(i - c).abs())
                        .sum();
                    diff > 0.0
                } else {
                    false
                };

                if a_changed || b_changed {
                    tracing::debug!(
                        "LoRA weights updated in {}: A={}, B={}",
                        module_name,
                        a_changed,
                        b_changed
                    );
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/// Model architecture information extracted from config.json.
///
/// This struct holds the key dimensions needed for creating adapter layers
/// with correct sizes, regardless of the specific model (SmolLM2-135M, `TinyLlama`, LLaMA-7B, etc.).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Hidden size / embedding dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    #[allow(dead_code)]
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Intermediate size (MLP hidden dimension)
    #[allow(dead_code)]
    pub intermediate_size: usize,
}

impl ModelInfo {
    /// Create `ModelInfo` from a `LlamaConfig`.
    #[must_use]
    pub fn from_llama_config(config: &LlamaConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            intermediate_size: config.intermediate_size,
        }
    }

    /// Get the input/output dimensions for a target module.
    ///
    /// Different projection layers have different dimensions:
    /// - `q_proj`: `hidden_size` -> `hidden_size`
    /// - `k_proj`, `v_proj`: `hidden_size` -> `hidden_size` * (`kv_heads` / `attn_heads`)
    /// - `o_proj`: `hidden_size` -> `hidden_size`
    /// - `gate_proj`, `up_proj`: `hidden_size` -> `intermediate_size`
    /// - `down_proj`: `intermediate_size` -> `hidden_size`
    #[allow(dead_code)]
    #[must_use]
    pub fn get_target_dims(&self, target: &str) -> (usize, usize) {
        match target {
            // Attention projections
            "q_proj" | "o_proj" => (self.hidden_size, self.hidden_size),
            "k_proj" | "v_proj" => {
                let kv_dim = self.hidden_size * self.num_kv_heads / self.num_attention_heads;
                (self.hidden_size, kv_dim)
            }
            // MLP projections
            "gate_proj" | "up_proj" => (self.hidden_size, self.intermediate_size),
            "down_proj" => (self.intermediate_size, self.hidden_size),
            // Default to hidden_size for unknown targets
            _ => (self.hidden_size, self.hidden_size),
        }
    }

    /// Create a default ModelInfo for testing (7B-like dimensions).
    #[cfg(test)]
    pub fn default_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 11008,
        }
    }
}

/// Load a model from the configuration.
///
/// # Errors
///
/// Returns an error if model files cannot be found or loaded.
pub fn load_model(config: &AxolotlConfig, device: &Device) -> Result<LoadedModel> {
    tracing::info!("Loading model: {}", config.base_model);

    // Determine model type from config
    let model_path = resolve_model_path(&config.base_model)?;

    // PR-063: refuse unsupported families early (before LlamaConfig parse / stub train)
    if !is_supported_llama_family(config, &model_path)? {
        return Err(AxolotlError::Model(format!(
            "Unsupported model architecture for '{}'. Supported families: {}. Refuse to fall back to a stub model. Convert/export to LLaMA-compatible weights or wait for additional architecture ports.",
            config.base_model,
            SUPPORTED_ARCHITECTURE_FAMILIES.join(", ")
        )));
    }

    // Load tokenizer
    let tokenizer = load_tokenizer(&model_path)?;
    tracing::info!(
        "Loaded tokenizer with vocab size: {}",
        tokenizer.get_vocab_size(true)
    );

    // Load model info from config.json for adapter layer dimensions
    let model_info = load_model_info(&model_path)?;
    tracing::info!(
        "Model info: hidden_size={}, num_layers={}, kv_heads={}",
        model_info.hidden_size,
        model_info.num_layers,
        model_info.num_kv_heads
    );

    // Determine dtype
    // Note: Force F32 for now as candle's RoPE doesn't handle F16 well
    // TODO: Enable F16 once candle fixes the rope dtype handling
    let dtype = DType::F32;

    if config.quantization.is_some() {
        tracing::info!("QLoRA mode: using F32 for model (quantization applied to weights)");
    }

    // Create trainable parameter map for adapters BEFORE loading model
    let trainable_params = VarMap::new();

    // Check adapter type for model loading strategy
    let use_lora_model = config.adapter == AdapterType::Lora;
    let is_qlora = config.adapter == AdapterType::Qlora;

    // Load model weights based on architecture and adapter type
    let (model, adapter_layers) = if is_qlora {
        // QLoraLlama: combines quantized base with trainable LoRA adapters
        #[cfg(all(feature = "peft", feature = "qlora"))]
        {
            let quant_settings = config.quantization.as_ref().ok_or_else(|| {
                AxolotlError::Config("QLoRA requires quantization settings".into())
            })?;

            let qlora_config = qlora_rs::QLoraConfig {
                lora: peft_rs::LoraConfig {
                    r: config.lora.r,
                    alpha: config.lora.alpha,
                    dropout: config.lora.dropout,
                    target_modules: config.lora.target_modules.clone(),
                    ..Default::default()
                },
                quantization: qlora_rs::QuantizationConfig {
                    block_size: quant_settings.block_size,
                    double_quant: quant_settings.double_quant,
                    // Critical for stability: BF16 has improved numerical stability for QLoRA training.
                    // Validation showed FP16 has ~20% failure rate (see PR description and QLoRA paper Section 4.1)
                    compute_dtype: qlora_rs::quantization::ComputeDType::BF16,
                    ..Default::default()
                },
                target_modules: config.lora.target_modules.clone(),
                cache_dequantized: false, // On-the-fly dequant for training (memory optimal)
            };

            let model = load_qlora_model(
                config,
                &model_path,
                device,
                dtype,
                &qlora_config,
                &trainable_params,
            )?;

            // AdapterLayers will be empty since adapters are embedded in QLoraLlama
            (model, None)
        }
        #[cfg(not(all(feature = "peft", feature = "qlora")))]
        {
            return Err(AxolotlError::Model(
                "QLoRA requested but peft and/or qlora features not enabled".into(),
            ));
        }
    } else if use_lora_model {
        // LoraLlama creates its own adapters internally during construction
        // Pass lora_config through model_info
        #[cfg(feature = "peft")]
        {
            let lora_config = PeftLoraConfig {
                r: config.lora.r,
                alpha: config.lora.alpha,
                dropout: config.lora.dropout,
                target_modules: config.lora.target_modules.clone(),
                ..Default::default()
            };

            let model = load_model_architecture(
                config,
                &model_path,
                device,
                dtype,
                None,
                Some((&model_info, &trainable_params, &lora_config)),
            )?;
            // AdapterLayers will be empty since LoRA is embedded in model
            (model, None)
        }
        #[cfg(not(feature = "peft"))]
        {
            return Err(AxolotlError::Model(
                "LoRA requested but peft feature not enabled".into(),
            ));
        }
    } else {
        // Standard model + separate adapter layers
        let model = load_model_architecture(config, &model_path, device, dtype, None, None)?;
        let adapter_layers = create_adapter_layers(config, &model_info, device, &trainable_params)?;
        (model, adapter_layers)
    };

    let adapter_count = adapter_layers.as_ref().map_or(0, AdapterLayers::len);
    let trainable_count: usize = trainable_params
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();

    tracing::info!(
        "Model loaded on {:?} with dtype {:?}, {} adapter layers, {} trainable params",
        device,
        dtype,
        adapter_count,
        trainable_count
    );

    Ok(LoadedModel {
        model,
        tokenizer,
        device: device.clone(),
        dtype,
        adapter_layers,
        trainable_params,
    })
}

/// Create adapter layers based on configuration.
///
/// Uses `VarBuilder` backed by `VarMap` to ensure `LoRA` weights are tracked
/// for gradient computation and optimizer updates.
#[allow(unused_variables)]
fn create_adapter_layers(
    config: &AxolotlConfig,
    model_info: &ModelInfo,
    device: &Device,
    trainable_params: &VarMap,
) -> Result<Option<AdapterLayers>> {
    match config.adapter {
        AdapterType::None => Ok(None),
        AdapterType::Lora => {
            #[cfg(feature = "peft")]
            {
                let mut layers = AdapterLayers::new(false);

                // Create LoRA config from settings
                let lora_config = PeftLoraConfig {
                    r: config.lora.r,
                    alpha: config.lora.alpha,
                    dropout: config.lora.dropout,
                    target_modules: config.lora.target_modules.clone(),
                    ..Default::default()
                };

                // Create VarBuilder from VarMap for gradient tracking
                // This ensures LoRA A/B weights are registered as trainable Vars
                let vb = VarBuilder::from_varmap(trainable_params, DType::F32, device);

                // Create LoRA layers for each target module with correct dimensions
                for target in &config.lora.target_modules {
                    let (in_features, out_features) = model_info.get_target_dims(target);

                    for layer_idx in 0..model_info.num_layers {
                        let layer_name = format!("model.layers.{layer_idx}.self_attn.{target}");

                        // Use VarBuilder with layer-specific prefix for unique variable names
                        let layer_vb = vb.pp(&layer_name);
                        let lora_layer = LoraLayer::new(
                            in_features,
                            out_features,
                            lora_config.clone(),
                            layer_vb,
                        )
                        .map_err(|e| {
                            AxolotlError::Model(format!(
                                "Failed to create LoRA layer {layer_name}: {e}"
                            ))
                        })?;

                        layers.lora_layers.insert(layer_name, lora_layer);
                    }
                }

                tracing::info!(
                    "Created {} LoRA layers with r={}, alpha={}",
                    layers.len(),
                    config.lora.r,
                    config.lora.alpha
                );

                Ok(Some(layers))
            }
            #[cfg(not(feature = "peft"))]
            {
                tracing::warn!("LoRA requested but peft feature not enabled");
                Ok(None)
            }
        }
        AdapterType::Qlora => {
            #[cfg(feature = "qlora")]
            {
                let quant_settings = config.quantization.as_ref().ok_or_else(|| {
                    AxolotlError::Config("QLoRA requires quantization settings".into())
                })?;

                let mut layers = AdapterLayers::new(true);

                // Create QLoRA config
                let qlora_config = QLoraConfig {
                    lora: peft_rs::LoraConfig {
                        r: config.lora.r,
                        alpha: config.lora.alpha,
                        dropout: config.lora.dropout,
                        target_modules: config.lora.target_modules.clone(),
                        ..Default::default()
                    },
                    quantization: qlora_rs::QuantizationConfig {
                        block_size: quant_settings.block_size,
                        double_quant: quant_settings.double_quant,
                        ..Default::default()
                    },
                    target_modules: config.lora.target_modules.clone(),
                    cache_dequantized: false, // On-the-fly dequant for training
                };

                // Create VarBuilder from VarMap for gradient tracking
                let vb = VarBuilder::from_varmap(trainable_params, DType::F32, device);

                // Create QLoRA layers for each target module with correct dimensions
                for target in &config.lora.target_modules {
                    let (in_features, out_features) = model_info.get_target_dims(target);

                    for layer_idx in 0..model_info.num_layers {
                        let layer_name = format!("model.layers.{}.self_attn.{}", layer_idx, target);

                        // Create zero-initialized weight tensor for quantization
                        // In real usage, this should load actual model weights
                        let weight =
                            Tensor::zeros(&[out_features, in_features], DType::F32, device)
                                .map_err(|e| {
                                    AxolotlError::Model(format!(
                                        "Failed to create weight tensor for {}: {}",
                                        layer_name, e
                                    ))
                                })?;

                        // Use VarBuilder for gradient tracking of LoRA weights
                        let layer_vb = vb.pp(&layer_name);
                        let qlora_layer = QuantizedLinear::from_weight_with_varbuilder(
                            &weight,
                            None,
                            &qlora_config,
                            layer_vb,
                        )
                        .map_err(|e| {
                            AxolotlError::Model(format!(
                                "Failed to create QLoRA layer {}: {}",
                                layer_name, e
                            ))
                        })?;

                        layers.qlora_layers.insert(layer_name, qlora_layer);
                    }
                }

                tracing::info!(
                    "Created {} QLoRA layers with r={}, alpha={}, {}bit quantization",
                    layers.len(),
                    config.lora.r,
                    config.lora.alpha,
                    quant_settings.bits
                );

                Ok(Some(layers))
            }
            #[cfg(not(feature = "qlora"))]
            {
                tracing::warn!("QLoRA requested but qlora feature not enabled");
                Ok(None)
            }
        }
    }
}

/// Load model info from config.json file.
fn load_model_info(model_path: &PathBuf) -> Result<ModelInfo> {
    let config_path = model_path.join("config.json");

    if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {e}")))?;
        let llama_config: LlamaConfig = serde_json::from_str(&config_str)
            .map_err(|e| AxolotlError::Model(format!("Failed to parse config.json: {e}")))?;
        Ok(ModelInfo::from_llama_config(&llama_config))
    } else {
        // Return default 7B-like config for testing
        tracing::warn!("config.json not found, using default LLaMA-7B dimensions");
        Ok(ModelInfo {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 11008,
        })
    }
}

/// Resolve model path from `HuggingFace` model ID or local path.
fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    // Check if it's a local path
    let path = PathBuf::from(model_id);
    if path.exists() {
        return Ok(path);
    }

    // Try HuggingFace cache directory
    let cache_dir = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HOME").map(|h| format!("{h}/.cache/huggingface")))
        .unwrap_or_else(|_| "/tmp/huggingface".to_string());

    let hf_path = PathBuf::from(format!(
        "{}/hub/models--{}",
        cache_dir,
        model_id.replace('/', "--")
    ));

    if hf_path.exists() {
        Ok(hf_path)
    } else {
        Err(AxolotlError::Model(format!(
            "Model not found at '{model_id}' or in HF cache at '{}'. Use `huggingface-cli download {model_id}` to download.", hf_path.display()
        )))
    }
}

/// Load tokenizer from model directory.
fn load_tokenizer(model_path: &PathBuf) -> Result<tokenizers::Tokenizer> {
    let tokenizer_file = model_path.join("tokenizer.json");

    if !tokenizer_file.exists() {
        return Err(AxolotlError::Tokenizer(
            format!("tokenizer.json not found in {}", model_path.display()).into(),
        ));
    }

    tokenizers::Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| AxolotlError::Tokenizer(format!("Failed to load tokenizer: {e}").into()))
}

/// Families currently supported for real weight load + train (not stubs).
pub const SUPPORTED_ARCHITECTURE_FAMILIES: &[&str] = &[
    "llama",
    "LlamaForCausalLM",
    "tinyllama (model_type=llama)",
    "smollm / SmolLM2 (model_type=llama)",
];

/// Detect whether a model directory / name is a supported LLaMA-family architecture.
fn is_supported_llama_family(config: &AxolotlConfig, model_path: &PathBuf) -> Result<bool> {
    let config_path = model_path.join("config.json");
    if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {e}")))?;
        // Prefer structured JSON when possible
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&config_str) {
            if let Some(model_type) = v.get("model_type").and_then(|x| x.as_str()) {
                let mt = model_type.to_lowercase();
                // LLaMA family only — refuse mistral/phi/gemma/qwen/etc. explicitly
                if mt == "llama" || mt == "tinyllama" {
                    return Ok(true);
                }
                return Ok(false);
            }
            if let Some(archs) = v.get("architectures").and_then(|x| x.as_array()) {
                for a in archs {
                    if let Some(s) = a.as_str() {
                        if s == "LlamaForCausalLM" {
                            return Ok(true);
                        }
                    }
                }
                // architectures present but not LLaMA
                if !archs.is_empty() {
                    return Ok(false);
                }
            }
        }
        // Fallback string heuristics on config body
        if config_str.contains("LlamaForCausalLM")
            || config_str.contains("\"model_type\": \"llama\"")
            || config_str.contains("\"model_type\":\"llama\"")
        {
            return Ok(true);
        }
        return Ok(false);
    }

    // No config.json: name-based heuristic for local paths / HF ids
    let name_lower = config.base_model.to_lowercase();
    Ok(name_lower.contains("llama")
        || name_lower.contains("smollm")
        || name_lower.contains("tinyllama"))
}

/// Load model architecture based on config.
fn load_model_architecture(
    config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
    _adapter_layers: Option<&AdapterLayers>,
    #[cfg(feature = "peft")] lora_params: Option<(&ModelInfo, &VarMap, &PeftLoraConfig)>,
    #[cfg(not(feature = "peft"))] lora_params: Option<(&ModelInfo, &VarMap)>,
) -> Result<Box<dyn Module>> {
    let is_llama_arch = is_supported_llama_family(config, model_path)?;

    if is_llama_arch {
        load_llama_model(config, model_path, device, dtype, lora_params)
    } else {
        // PR-063: never silently train on a 10×10 stub when the user passed a real model path
        Err(AxolotlError::Model(format!(
            "Unsupported model architecture for '{}'. Supported families: {}. Refuse to fall back to a stub model. Convert/export to LLaMA-compatible weights or wait for additional architecture ports.",
            config.base_model,
            SUPPORTED_ARCHITECTURE_FAMILIES.join(", ")
        )))
    }
}

/// Load weight tensors from single-file or sharded `HuggingFace` `safetensors` layout.
///
/// Supports:
/// - `model.safetensors`
/// - `model.safetensors.index.json` + multi-file shards (`model-00001-of-0000N.safetensors`)
/// - `pytorch_model.bin`
fn load_weight_varbuilder(
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
) -> Result<VarBuilder<'static>> {
    let single = model_path.join("model.safetensors");
    let index = model_path.join("model.safetensors.index.json");
    let pth = model_path.join("pytorch_model.bin");

    if single.exists() {
        let tensors = candle_core::safetensors::load(&single, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load safetensors: {e}")))?;
        return Ok(VarBuilder::from_tensors(tensors, dtype, device));
    }

    if index.exists() {
        let tensors = load_sharded_safetensors(model_path, &index, device)?;
        return Ok(VarBuilder::from_tensors(tensors, dtype, device));
    }

    if pth.exists() {
        return VarBuilder::from_pth(&pth, dtype, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load pytorch model: {e}")));
    }

    Err(AxolotlError::Model(format!(
        "No model weights found in {}. Expected model.safetensors, model.safetensors.index.json (+ shards), or pytorch_model.bin",
        model_path.display()
    )))
}

/// Load multi-file `safetensors` via `HuggingFace` `model.safetensors.index.json`.
fn load_sharded_safetensors(
    model_path: &PathBuf,
    index_path: &PathBuf,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let index_str = std::fs::read_to_string(index_path).map_err(|e| {
        AxolotlError::Model(format!(
            "Failed to read sharded index {}: {e}",
            index_path.display()
        ))
    })?;
    let index: serde_json::Value = serde_json::from_str(&index_str).map_err(|e| {
        AxolotlError::Model(format!(
            "Failed to parse sharded index {}: {e}",
            index_path.display()
        ))
    })?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            AxolotlError::Model(format!(
                "Invalid {}: missing 'weight_map' object. This looks like a HuggingFace sharded layout — fix the index or merge shards with `safetensors` tooling.",
                index_path.display()
            ))
        })?;

    if weight_map.is_empty() {
        return Err(AxolotlError::Model(format!(
            "Empty weight_map in {}. No tensors to load.",
            index_path.display()
        )));
    }

    // Unique shard filenames
    let mut shard_files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    if shard_files.is_empty() {
        return Err(AxolotlError::Model(format!(
            "weight_map in {} has no valid shard filenames.",
            index_path.display()
        )));
    }

    let mut all_tensors: HashMap<String, Tensor> = HashMap::new();
    for shard_name in &shard_files {
        let shard_path = model_path.join(shard_name);
        if !shard_path.exists() {
            return Err(AxolotlError::Model(format!(
                "Sharded model index references missing shard '{shard_name}' (expected at {}). Place all shard files next to the index, or merge into a single model.safetensors before training. Refusing to fall back to a stub model.",
                shard_path.display()
            )));
        }
        tracing::info!("Loading safetensors shard: {}", shard_path.display());
        let tensors = candle_core::safetensors::load(&shard_path, device).map_err(|e| {
            AxolotlError::Model(format!(
                "Failed to load shard {}: {e}",
                shard_path.display()
            ))
        })?;
        all_tensors.extend(tensors);
    }

    // Soft check: report if index keys are missing from loaded tensors
    let missing: Vec<&String> = weight_map
        .keys()
        .filter(|k| !all_tensors.contains_key(k.as_str()))
        .collect();
    if !missing.is_empty() {
        tracing::warn!(
            "Sharded load: {} weight_map keys not found in shards (e.g. {:?})",
            missing.len(),
            missing.first()
        );
    }

    tracing::info!(
        "Loaded sharded safetensors: {} tensors from {} shard(s)",
        all_tensors.len(),
        shard_files.len()
    );
    Ok(all_tensors)
}

/// Load a `LLaMA` model from the given path.
fn load_llama_model(
    _axolotl_config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
    #[cfg(feature = "peft")] lora_params: Option<(&ModelInfo, &VarMap, &PeftLoraConfig)>,
    #[cfg(not(feature = "peft"))] _lora_params: Option<(&ModelInfo, &VarMap)>,
) -> Result<Box<dyn Module>> {
    // Try to load config.json first
    let config_path = model_path.join("config.json");
    let llama_config: LlamaConfig = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {e}")))?;
        let parsed: LlamaConfig = serde_json::from_str(&config_str)
            .map_err(|e| AxolotlError::Model(format!("Failed to parse config.json: {e}")))?;
        parsed
    } else {
        // Use default config for LLaMA 2 7B
        tracing::warn!("config.json not found, using default LLaMA 2 7B config");
        LlamaConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(LlamaEosToks::Single(2)),
            max_position_embeddings: 4096,
            rope_scaling: None,
            tie_word_embeddings: None,
        }
    };

    // Load model weights (single-file, sharded index, or pytorch)
    let vb = load_weight_varbuilder(model_path, device, dtype)?;

    // Convert LlamaConfig to Config for Llama::load
    let config = candle_transformers::models::llama::Config {
        hidden_size: llama_config.hidden_size,
        intermediate_size: llama_config.intermediate_size,
        vocab_size: llama_config.vocab_size,
        num_hidden_layers: llama_config.num_hidden_layers,
        num_attention_heads: llama_config.num_attention_heads,
        num_key_value_heads: llama_config.num_key_value_heads(),
        use_flash_attn: false, // TODO: make configurable
        rms_norm_eps: llama_config.rms_norm_eps,
        rope_theta: llama_config.rope_theta,
        bos_token_id: llama_config.bos_token_id,
        eos_token_id: llama_config.eos_token_id,
        rope_scaling: llama_config.rope_scaling,
        max_position_embeddings: llama_config.max_position_embeddings,
        tie_word_embeddings: llama_config.tie_word_embeddings.unwrap_or(false),
    };

    #[cfg(feature = "peft")]
    let model: Box<dyn Module> =
        if let Some((_model_info, trainable_params, lora_config)) = lora_params {
            tracing::info!("Loading LoraLlama with per-layer LoRA injection");

            // Create LoraLlama with internal adapters
            let model = LoraLlama::new_with_lora(&config, vb, lora_config, trainable_params)
                .map_err(|e| AxolotlError::Model(format!("Failed to create LoraLlama: {e}")))?;

            Box::new(model)
        } else {
            // Use standard Llama model wrapped for training
            let model = Llama::load(vb, &config)
                .map_err(|e| AxolotlError::Model(format!("Failed to create LLaMA model: {e}")))?;

            Box::new(LlamaWrapper::new(model, &config, device)?)
        };

    #[cfg(not(feature = "peft"))]
    let model: Box<dyn Module> = {
        // Use standard Llama model wrapped for training
        let model = Llama::load(vb, &config)
            .map_err(|e| AxolotlError::Model(format!("Failed to create LLaMA model: {e}")))?;

        Box::new(LlamaWrapper::new(model, &config, device)?)
    };

    tracing::info!(
        "Loaded LLaMA model with {} layers, {} hidden size",
        llama_config.num_hidden_layers,
        llama_config.hidden_size
    );

    Ok(model)
}

/// Load a QLoRA LLaMA model with quantized base weights and trainable LoRA adapters.
///
/// This function:
/// 1. Loads base model weights from safetensors/pytorch
/// 2. Quantizes transformer layers to NF4 format
/// 3. Creates trainable LoRA adapters at target modules
/// 4. Keeps embeddings, layer norms, and lm_head in FP32
///
/// # Arguments
/// * `axolotl_config` - Axolotl configuration
/// * `model_path` - Path to model files
/// * `device` - Device for computation
/// * `dtype` - Data type for non-quantized weights
/// * `qlora_config` - QLoRA configuration
/// * `trainable_params` - VarMap for registering LoRA parameters
///
/// # Errors
/// Returns error if model loading or quantization fails.
#[cfg(all(feature = "peft", feature = "qlora"))]
fn load_qlora_model(
    _axolotl_config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
    qlora_config: &qlora_rs::QLoraConfig,
    trainable_params: &VarMap,
) -> Result<Box<dyn Module>> {
    // Load config.json
    let config_path = model_path.join("config.json");
    let llama_config: LlamaConfig = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {}", e)))?;
        serde_json::from_str(&config_str)
            .map_err(|e| AxolotlError::Model(format!("Failed to parse config.json: {}", e)))?
    } else {
        return Err(AxolotlError::Model(
            "config.json required for QLoRA model loading".into(),
        ));
    };

    // Load model weights (single-file, sharded index, or pytorch)
    let vb = load_weight_varbuilder(model_path, device, dtype)?;

    // Convert to candle-transformers Config
    let config = candle_transformers::models::llama::Config {
        hidden_size: llama_config.hidden_size,
        intermediate_size: llama_config.intermediate_size,
        vocab_size: llama_config.vocab_size,
        num_hidden_layers: llama_config.num_hidden_layers,
        num_attention_heads: llama_config.num_attention_heads,
        num_key_value_heads: llama_config.num_key_value_heads(),
        use_flash_attn: false,
        rms_norm_eps: llama_config.rms_norm_eps,
        rope_theta: llama_config.rope_theta,
        bos_token_id: llama_config.bos_token_id,
        eos_token_id: llama_config.eos_token_id,
        rope_scaling: llama_config.rope_scaling,
        max_position_embeddings: llama_config.max_position_embeddings,
        tie_word_embeddings: llama_config.tie_word_embeddings.unwrap_or(false),
    };

    tracing::info!(
        "Loading QLoraLlama with {} layers, {} hidden size, r={}, alpha={}",
        config.num_hidden_layers,
        config.hidden_size,
        qlora_config.lora.r,
        qlora_config.lora.alpha
    );

    // Create QLoraLlama
    let model = QLoraLlama::new_with_qlora(&config, vb, qlora_config, trainable_params)
        .map_err(|e| AxolotlError::Model(format!("Failed to create QLoraLlama: {}", e)))?;

    // Prepare for training (validates setup, logs info)
    prepare_for_qlora_training(&model, trainable_params)
        .map_err(|e| AxolotlError::Model(format!("Failed to prepare QLoRA for training: {}", e)))?;

    let trainable_count: usize = trainable_params
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    let total_params = model.total_param_count();
    let trainable_pct = 100.0 * trainable_count as f64 / total_params as f64;

    tracing::info!(
        "QLoraLlama ready: {} total params, {} trainable ({:.2}%)",
        total_params,
        trainable_count,
        trainable_pct
    );

    Ok(Box::new(model))
}

/// Simple stub model for unsupported architectures.
struct SimpleModel {
    layer: candle_nn::Linear,
}

impl SimpleModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let layer = candle_nn::linear(10, 10, vb)?;
        Ok(Self { layer })
    }
}

impl Module for SimpleModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.layer.forward(xs)
    }
}

/// Wrapper for `LLaMA` model that implements the Module trait.
///
/// For training, we need logits for ALL positions, not just the last token.
/// The default candle Llama only returns last-token logits for inference.
pub struct LlamaWrapper {
    model: Llama,
    cache: std::cell::RefCell<Cache>,
    /// Whether to use training mode (all positions) or inference mode (last position only)
    #[allow(dead_code)]
    training_mode: bool,
}

impl LlamaWrapper {
    /// Create a new `LlamaWrapper` in training mode by default.
    pub fn new(
        model: Llama,
        config: &candle_transformers::models::llama::Config,
        device: &Device,
    ) -> Result<Self> {
        let cache = Cache::new(false, DType::F32, config, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to create cache: {e}")))?;
        Ok(Self {
            model,
            cache: std::cell::RefCell::new(cache),
            training_mode: true, // Default to training mode
        })
    }

    /// Set whether to use training mode (all positions) or inference mode (last position)
    #[allow(dead_code)]
    pub fn set_training_mode(&mut self, training: bool) {
        self.training_mode = training;
    }
}

impl Module for LlamaWrapper {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut cache = self.cache.borrow_mut();

        // Use standard forward - returns logits for last position only
        // For training, we compute loss on the last token prediction
        // This is simpler and faster than computing all-position logits
        self.model.forward(xs, 0, &mut cache)
    }
}

impl LlamaWrapper {
    /// Forward pass that returns logits for all positions (for training).
    ///
    /// Candle's `Llama.forward()` only returns logits for the last token,
    /// but for training we need logits for all positions to compute loss
    /// across the entire sequence.
    #[allow(dead_code)]
    fn forward_all_positions(&self, xs: &Tensor, cache: &mut Cache) -> candle_core::Result<Tensor> {
        // Get sequence length for later
        let (_b_sz, seq_len) = xs.dims2()?;

        // Embed input tokens
        // Access wte (word token embeddings) through public interface
        // Since we can't directly access model internals, we need a workaround

        // For training, we'll compute logits position-by-position
        // This is inefficient but works as a starting point
        let mut all_logits = Vec::new();

        for pos in 0..seq_len {
            // Get logits at each position by running forward with truncated input
            let input_slice = xs.i((.., 0..=pos))?;
            let logits = self.model.forward(&input_slice, 0, cache)?;
            all_logits.push(logits);

            // Clear cache between positions to avoid accumulation issues
            // (This is inefficient but correct for initial validation)
        }

        // Stack all logits: [batch, seq_len, vocab]
        let stacked = Tensor::stack(&all_logits, 1)?;
        Ok(stacked)
    }
}

/// Merge `LoRA` adapter ΔW into base linear weights and save a merged model directory.
///
/// For each module with adapter keys `{module}.lora_a.weight` / `{module}.lora_b.weight`
/// (or HF-style `lora_A` / `lora_B`), computes:
///
/// ```text
/// W' = W + (B @ A) * (alpha / r)
/// ```
///
/// and writes `model.safetensors` plus copies of `config.json` / tokenizer files.
///
/// # Arguments
/// * `config` - Axolotl config (`base_model`, `LoRA` r/alpha)
/// * `adapter_path` - Directory containing `adapter_model.safetensors` (or the file itself)
/// * `output_path` - Output directory for the merged model
///
/// # Errors
/// Returns an error if base/adapter weights are missing, shapes mismatch, or I/O fails.
pub fn merge_adapter(config: &AxolotlConfig, adapter_path: &str, output_path: &str) -> Result<()> {
    let device = Device::Cpu;
    let model_path = resolve_model_path(&config.base_model)?;

    // ---- load base weights ----
    let single = model_path.join("model.safetensors");
    let index = model_path.join("model.safetensors.index.json");
    let mut base_tensors: HashMap<String, Tensor> = if single.exists() {
        candle_core::safetensors::load(&single, &device).map_err(|e| {
            AxolotlError::Model(format!("Failed to load base model.safetensors: {e}"))
        })?
    } else if index.exists() {
        load_sharded_safetensors(&model_path, &index, &device)?
    } else {
        return Err(AxolotlError::Model(format!(
            "Cannot merge: base weights not found at {} (need model.safetensors or sharded index). Set base_model to a local path with full-precision weights.",
            model_path.display()
        )));
    };

    // ---- load adapter weights ----
    let adapter_p = PathBuf::from(adapter_path);
    let adapter_file = if adapter_p.is_dir() {
        adapter_p.join("adapter_model.safetensors")
    } else {
        adapter_p.clone()
    };
    if !adapter_file.exists() {
        return Err(AxolotlError::Model(format!(
            "Cannot merge: adapter weights not found at {}. Expected adapter_model.safetensors (train first or pass --adapter).",
            adapter_file.display()
        )));
    }
    let adapter_tensors = candle_core::safetensors::load(&adapter_file, &device).map_err(|e| {
        AxolotlError::Model(format!(
            "Failed to load adapter {}: {e}",
            adapter_file.display()
        ))
    })?;
    if adapter_tensors.is_empty() {
        return Err(AxolotlError::Model(
            "Cannot merge: adapter file contains zero tensors".into(),
        ));
    }

    // Scale: prefer adapter_config.json next to weights, else YAML lora settings
    let (alpha, r) = {
        let cfg_path = if adapter_p.is_dir() {
            adapter_p.join("adapter_config.json")
        } else {
            adapter_p
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("adapter_config.json")
        };
        if cfg_path.exists() {
            let s = std::fs::read_to_string(&cfg_path).map_err(|e| {
                AxolotlError::Model(format!("Failed to read adapter_config.json: {e}"))
            })?;
            let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| {
                AxolotlError::Model(format!("Failed to parse adapter_config.json: {e}"))
            })?;
            let alpha = v
                .get("lora_alpha")
                .or_else(|| v.get("alpha"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(config.lora.alpha as u64) as usize;
            let r = v
                .get("r")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(config.lora.r as u64) as usize;
            (alpha, r)
        } else {
            (config.lora.alpha, config.lora.r)
        }
    };
    if r == 0 {
        return Err(AxolotlError::Model(
            "Cannot merge: LoRA rank r must be > 0".into(),
        ));
    }
    let scale = alpha as f64 / r as f64;

    // Pair A/B by module prefix
    let mut modules: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();
    for (key, tensor) in &adapter_tensors {
        let (module, is_a) = if let Some(m) = key.strip_suffix(".lora_a.weight") {
            (m, true)
        } else if let Some(m) = key.strip_suffix(".lora_A.weight") {
            (m, true)
        } else if let Some(m) = key.strip_suffix(".lora_A.default.weight") {
            (m, true)
        } else if let Some(m) = key.strip_suffix(".lora_b.weight") {
            (m, false)
        } else if let Some(m) = key.strip_suffix(".lora_B.weight") {
            (m, false)
        } else if let Some(m) = key.strip_suffix(".lora_B.default.weight") {
            (m, false)
        } else {
            tracing::debug!("merge: skipping non-LoRA key {key}");
            continue;
        };
        let entry = modules.entry(module.to_string()).or_insert((None, None));
        if is_a {
            entry.0 = Some(tensor.clone());
        } else {
            entry.1 = Some(tensor.clone());
        }
    }

    if modules.is_empty() {
        return Err(AxolotlError::Model(
            "Cannot merge: no LoRA A/B tensors found in adapter file (expected keys like '*.lora_a.weight' / '*.lora_b.weight')"
                .into(),
        ));
    }

    let mut merged_count = 0usize;
    for (module, (a_opt, b_opt)) in &modules {
        let (Some(a), Some(b)) = (a_opt, b_opt) else {
            return Err(AxolotlError::Model(format!(
                "Cannot merge module '{module}': missing paired lora_a/lora_b tensor"
            )));
        };
        let base_key = format!("{module}.weight");
        let base = base_tensors.get(&base_key).ok_or_else(|| {
            AxolotlError::Model(format!(
                "Cannot merge: base weight '{base_key}' not found in base model. Adapter module prefix may not match base key layout."
            ))
        })?;

        // ΔW = B @ A * scale ; shapes: A [r, in], B [out, r], W [out, in]
        let delta = b
            .matmul(a)
            .map_err(|e| AxolotlError::Model(format!("merge matmul for {module}: {e}")))?;
        let delta = delta
            .affine(scale, 0.0)
            .map_err(|e| AxolotlError::Model(format!("merge scale for {module}: {e}")))?;
        let merged = base.broadcast_add(&delta).map_err(|e| {
            AxolotlError::Model(format!(
                "merge add for {module}: {e} (base={:?}, delta={:?})",
                base.dims(),
                delta.dims()
            ))
        })?;
        base_tensors.insert(base_key, merged);
        merged_count += 1;
        tracing::info!("Merged LoRA into {module} (scale={scale:.4})");
    }

    // ---- write output ----
    let out = PathBuf::from(output_path);
    std::fs::create_dir_all(&out)?;
    let out_weights = out.join("model.safetensors");
    candle_core::safetensors::save(&base_tensors, &out_weights).map_err(|e| {
        AxolotlError::Model(format!(
            "Failed to write merged model to {}: {e}",
            out_weights.display()
        ))
    })?;

    // Copy metadata files when present
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ] {
        let src = model_path.join(name);
        if src.exists() {
            let dst = out.join(name);
            std::fs::copy(&src, &dst).map_err(|e| {
                AxolotlError::Model(format!("Failed to copy {name} to output: {e}"))
            })?;
        }
    }

    // Record merge metadata
    let meta = serde_json::json!({
        "axolotl_rs_merge": true,
        "base_model": config.base_model,
        "adapter_path": adapter_path,
        "modules_merged": merged_count,
        "lora_r": r,
        "lora_alpha": alpha,
        "scale": scale,
    });
    std::fs::write(
        out.join("merge_info.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )?;

    tracing::info!(
        "Merged {merged_count} LoRA module(s) into base; wrote {}",
        out.display()
    );
    Ok(())
}

/// Download a model from Hugging Face Hub into `cache_dir/<sanitized_id>/`.
///
/// Fetches `config.json`, `tokenizer.json` (when present), and either
/// `model.safetensors` or sharded weights via `model.safetensors.index.json`.
///
/// Local paths are preferred at train time via `resolve_model_path`; this helper
/// is for the CLI `download` command and library callers who want a minimal Hub pull
/// without installing `huggingface-cli`.
///
/// # Errors
/// Returns an error when network/API fails, files are missing, or the download feature
/// is not enabled.
#[cfg(feature = "download")]
pub fn download_model(model_id: &str, cache_dir: &str) -> Result<String> {
    // Already local?
    let local = PathBuf::from(model_id);
    if local.is_dir()
        && (local.join("model.safetensors").exists()
            || local.join("model.safetensors.index.json").exists())
    {
        return Ok(local.display().to_string());
    }

    let sanitized = model_id.replace('/', "--");
    let dest = PathBuf::from(cache_dir).join(&sanitized);
    std::fs::create_dir_all(&dest)?;

    let client = reqwest::blocking::Client::builder()
        .user_agent(format!("axolotl-rs/{}", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_mins(10))
        .build()
        .map_err(|e| AxolotlError::Model(format!("Failed to build HTTP client: {e}")))?;

    let base = format!("https://huggingface.co/{model_id}/resolve/main");

    // Always try config.json
    download_hf_file(&client, &base, "config.json", &dest)?;

    // Tokenizer is required for train
    match download_hf_file(&client, &base, "tokenizer.json", &dest) {
        Ok(()) => {}
        Err(e) => {
            tracing::warn!("tokenizer.json not downloaded: {e}");
        }
    }
    // Best-effort extras
    for extra in [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ] {
        let _ = download_hf_file(&client, &base, extra, &dest);
    }

    // Weights: single file first, else sharded index
    if download_hf_file(&client, &base, "model.safetensors", &dest).is_ok() {
        tracing::info!("Downloaded single-file model.safetensors");
    } else {
        // Try sharded
        match download_hf_file(&client, &base, "model.safetensors.index.json", &dest) {
            Ok(()) => {
                let index_path = dest.join("model.safetensors.index.json");
                let index_str = std::fs::read_to_string(&index_path).map_err(|e| {
                    AxolotlError::Model(format!("Failed to read downloaded index: {e}"))
                })?;
                let index: serde_json::Value = serde_json::from_str(&index_str).map_err(|e| {
                    AxolotlError::Model(format!("Failed to parse downloaded index: {e}"))
                })?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| {
                        AxolotlError::Model(
                            "Downloaded model.safetensors.index.json missing weight_map".into(),
                        )
                    })?;
                let mut shards: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                shards.sort();
                shards.dedup();
                for shard in &shards {
                    download_hf_file(&client, &base, shard, &dest)?;
                }
                tracing::info!("Downloaded {} shard(s) for {model_id}", shards.len());
            }
            Err(_) => {
                // Try pytorch as last resort
                if download_hf_file(&client, &base, "pytorch_model.bin", &dest).is_err() {
                    return Err(AxolotlError::Model(format!(
                        "Failed to download weights for '{model_id}'. Tried model.safetensors, sharded index, and pytorch_model.bin. \
If the model is gated, set HF_TOKEN and retry, or run: huggingface-cli download {model_id} --local-dir <path> and set base_model to that path."
                    )));
                }
            }
        }
    }

    Ok(dest.display().to_string())
}

/// Download a single Hub file into `dest_dir/filename` (skips if already present).
#[cfg(feature = "download")]
fn download_hf_file(
    client: &reqwest::blocking::Client,
    base_url: &str,
    filename: &str,
    dest_dir: &Path,
) -> Result<()> {
    let dest = dest_dir.join(filename);
    if dest.exists() && dest.metadata().is_ok_and(|m| m.len() > 0) {
        tracing::debug!("Using cached {}", dest.display());
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let url = format!("{base_url}/{filename}");
    let mut req = client.get(&url);
    if let Ok(token) =
        std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
    {
        if !token.is_empty() {
            req = req.bearer_auth(token);
        }
    }
    let response = req
        .send()
        .map_err(|e| AxolotlError::Model(format!("HTTP request failed for {url}: {e}")))?;
    if !response.status().is_success() {
        return Err(AxolotlError::Model(format!(
            "HTTP {} downloading {url}",
            response.status()
        )));
    }
    let bytes = response
        .bytes()
        .map_err(|e| AxolotlError::Model(format!("Failed to read body for {url}: {e}")))?;
    // Atomic-ish write
    let tmp = dest.with_extension("partial");
    std::fs::write(&tmp, &bytes)
        .map_err(|e| AxolotlError::Model(format!("Failed to write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &dest)
        .map_err(|e| AxolotlError::Model(format!("Failed to finalize {}: {e}", dest.display())))?;
    tracing::info!("Downloaded {filename} ({} bytes)", bytes.len());
    Ok(())
}

/// Stub when the `download` feature is disabled.
#[cfg(not(feature = "download"))]
pub fn download_model(_model_id: &str, _cache_dir: &str) -> Result<String> {
    Err(AxolotlError::Model(
        "download feature not enabled; rebuild with --features download, or use `huggingface-cli download <id>` and set base_model to a local path."
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Test ModelInfo dimension calculations for different target modules.
    #[test]
    fn test_model_info_target_dims() {
        // SmolLM2-135M dimensions
        let smollm2 = ModelInfo {
            hidden_size: 576,
            num_layers: 30,
            num_attention_heads: 9,
            num_kv_heads: 3,
            intermediate_size: 1536,
        };

        // q_proj and o_proj: hidden_size -> hidden_size
        assert_eq!(smollm2.get_target_dims("q_proj"), (576, 576));
        assert_eq!(smollm2.get_target_dims("o_proj"), (576, 576));

        // k_proj and v_proj: hidden_size -> kv_dim (with GQA)
        // kv_dim = 576 * 3 / 9 = 192
        assert_eq!(smollm2.get_target_dims("k_proj"), (576, 192));
        assert_eq!(smollm2.get_target_dims("v_proj"), (576, 192));

        // MLP projections
        assert_eq!(smollm2.get_target_dims("gate_proj"), (576, 1536));
        assert_eq!(smollm2.get_target_dims("up_proj"), (576, 1536));
        assert_eq!(smollm2.get_target_dims("down_proj"), (1536, 576));
    }

    /// Test ModelInfo for TinyLlama-1.1B dimensions.
    #[test]
    fn test_model_info_tinyllama() {
        let tinyllama = ModelInfo {
            hidden_size: 2048,
            num_layers: 22,
            num_attention_heads: 32,
            num_kv_heads: 4,
            intermediate_size: 5632,
        };

        // q_proj: full hidden_size
        assert_eq!(tinyllama.get_target_dims("q_proj"), (2048, 2048));

        // k_proj with GQA: 2048 * 4 / 32 = 256
        assert_eq!(tinyllama.get_target_dims("k_proj"), (2048, 256));

        // MLP
        assert_eq!(tinyllama.get_target_dims("gate_proj"), (2048, 5632));
    }

    /// Test ModelInfo for LLaMA-7B dimensions (no GQA).
    #[test]
    fn test_model_info_llama7b() {
        let llama7b = ModelInfo::default_7b();

        // No GQA, so kv_heads == attn_heads
        assert_eq!(llama7b.get_target_dims("q_proj"), (4096, 4096));
        assert_eq!(llama7b.get_target_dims("k_proj"), (4096, 4096));
        assert_eq!(llama7b.get_target_dims("v_proj"), (4096, 4096));
    }

    /// Test loading a LLaMA 2 model configuration.
    ///
    /// Currently tests that the function can be called with a valid config
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_load_model_llama2() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let device = Device::Cpu;

        // Currently returns "Model not found" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            assert!(msg.contains("Model not found"));
        } else {
            panic!("Expected Model error");
        }
    }

    /// Test loading a Mistral model configuration.
    ///
    /// Currently tests that the function can be called with a valid config
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_load_model_mistral() {
        let config = AxolotlConfig {
            base_model: "mistralai/Mistral-7B-v0.1".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let device = Device::Cpu;

        // Currently returns "Model loading not yet implemented" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            assert!(msg.contains("Model not found"));
        } else {
            panic!("Expected Model error");
        }
    }

    /// Test loading a Phi-3 model configuration.
    ///
    /// Currently tests that the function can be called with a valid config
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_load_model_phi3() {
        let config = AxolotlConfig {
            base_model: "microsoft/Phi-3-mini-4k-instruct".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let device = Device::Cpu;

        // Currently returns "Model not found" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            assert!(msg.contains("Model not found"));
        } else {
            panic!("Expected Model error");
        }
    }

    /// Test merging a LoRA adapter into a base model.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_merge_adapter_lora() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                dropout: 0.0,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            },
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let temp_dir = TempDir::new().unwrap();
        let adapter_path = temp_dir.path().join("adapter");
        fs::create_dir(&adapter_path).unwrap();

        // Base model not local — merge should error (not silent success)
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            msg.contains("not found") || msg.contains("cannot merge") || msg.contains("model"),
            "unexpected merge error: {msg}"
        );
    }

    /// Test merging a QLoRA adapter with quantization.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_merge_adapter_qlora() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Qlora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                dropout: 0.0,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            },
            quantization: Some(QuantizationSettings {
                bits: 4,
                quant_type: QuantType::Nf4,
                double_quant: true,
                block_size: 64,
            }),
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let temp_dir = TempDir::new().unwrap();
        let adapter_path = temp_dir.path().join("adapter");
        fs::create_dir(&adapter_path).unwrap();

        // Base model not local — merge should error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
    }

    /// Test merging adapter weights back into base model.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_merge_adapter() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let temp_dir = TempDir::new().unwrap();
        let adapter_path = temp_dir.path().join("adapter");
        fs::create_dir(&adapter_path).unwrap();

        // Base model not local — merge should error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
    }

    /// Test downloading model from HuggingFace Hub.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    #[cfg(feature = "download")]
    fn test_download_model_from_hub() {
        // Use a clearly nonexistent repo id so we never succeed or pull multi-GB models.
        // Download may fail with HTTP/network/not-found — never claim success for garbage ids.
        let result: Result<String> = download_model(
            "axolotl-rs-ci/this-repo-does-not-exist-xyz",
            "/tmp/axolotl-rs-download-test-cache",
        );
        // Either errs (expected) or, if network blocked, also errs.
        assert!(
            result.is_err(),
            "download of nonexistent model must not succeed: {result:?}"
        );
    }

    /// Test error handling for invalid model paths.
    #[test]
    fn test_resolve_model_path_invalid() {
        let result = resolve_model_path("nonexistent-model-id");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => assert!(msg.contains("Model not found")),
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test tokenizer loading with missing tokenizer file.
    #[test]
    fn test_load_tokenizer_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let result = load_tokenizer(&temp_dir.path().to_path_buf());
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Tokenizer(e) => {
                assert!(e.to_string().contains("tokenizer.json not found"))
            }
            _ => panic!("Expected Tokenizer error, got {:?}", err),
        }
    }

    /// PR-063: non-LLaMA architectures must be refused (no silent 10×10 stub).
    #[test]
    fn test_load_model_architecture_refuses_unsupported() {
        let config = AxolotlConfig {
            base_model: "test-model".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };
        let temp_dir = TempDir::new().unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;

        let result = load_model_architecture(
            &config,
            &temp_dir.path().to_path_buf(),
            &device,
            dtype,
            None,
            None,
        );
        match result {
            Ok(_) => panic!("expected unsupported architecture error"),
            Err(e) => {
                let err = e.to_string();
                assert!(
                    err.contains("Unsupported model architecture")
                        || err.contains("Supported families"),
                    "unexpected error: {err}"
                );
            }
        }
    }

    /// PR-063: config.json with non-llama model_type is refused even if path name is neutral.
    #[test]
    fn test_refuse_mistral_architecture_from_config_json() {
        let temp_dir = TempDir::new().unwrap();
        let cfg = r#"{
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 128
        }"#;
        fs::write(temp_dir.path().join("config.json"), cfg).unwrap();

        let config = AxolotlConfig {
            base_model: temp_dir.path().to_string_lossy().to_string(),
            adapter: AdapterType::None,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };
        let result = load_model_architecture(
            &config,
            &temp_dir.path().to_path_buf(),
            &Device::Cpu,
            DType::F32,
            None,
            None,
        );
        match result {
            Ok(_) => panic!("expected unsupported architecture error"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("Unsupported"), "got: {msg}");
                assert!(
                    msg.contains("llama") || msg.contains("Supported families"),
                    "should list supported families: {msg}"
                );
            }
        }
    }

    /// PR-062: missing shard referenced by index.json is a hard error (no stub).
    #[test]
    fn test_sharded_index_missing_shard_is_error() {
        let temp_dir = TempDir::new().unwrap();
        let index = r#"{
            "metadata": {"total_size": 1},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors"
            }
        }"#;
        fs::write(temp_dir.path().join("model.safetensors.index.json"), index).unwrap();
        // Write a minimal llama config so architecture passes
        let cfg = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 64,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0
        }"#;
        fs::write(temp_dir.path().join("config.json"), cfg).unwrap();

        match load_weight_varbuilder(&temp_dir.path().to_path_buf(), &Device::Cpu, DType::F32) {
            Ok(_) => panic!("expected missing shard error"),
            Err(e) => {
                let err = e.to_string();
                assert!(
                    err.contains("missing shard") || err.contains("Sharded"),
                    "unexpected: {err}"
                );
            }
        }
    }

    /// Positive proof: fuse LoRA A/B into base weights and write merged model.
    #[test]
    fn test_merge_adapter_fuses_and_writes() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path().join("base");
        crate::fixture::write_tiny_llama_fixture(
            &model_dir,
            crate::fixture::TinyLlamaSpec {
                vocab_size: 32,
                hidden_size: 16,
                intermediate_size: 32,
                num_hidden_layers: 1,
                num_attention_heads: 4,
                num_key_value_heads: 4,
                max_position_embeddings: 64,
            },
        )
        .unwrap();

        let device = Device::Cpu;
        let h = 16usize;
        let r = 4usize;
        let alpha = 8usize;
        let scale = alpha as f64 / r as f64;

        // Distinct A/B so merged weights differ from base
        let a = Tensor::ones((r, h), DType::F32, &device).unwrap(); // [r, in]
        let b = Tensor::full(0.5f32, (h, r), &device).unwrap(); // [out, r]
        let module = "model.layers.0.self_attn.q_proj";
        let mut adapter_map = HashMap::new();
        adapter_map.insert(format!("{module}.lora_a.weight"), a.clone());
        adapter_map.insert(format!("{module}.lora_b.weight"), b.clone());

        let adapter_dir = temp_dir.path().join("adapter");
        fs::create_dir_all(&adapter_dir).unwrap();
        candle_core::safetensors::save(&adapter_map, adapter_dir.join("adapter_model.safetensors"))
            .unwrap();
        fs::write(
            adapter_dir.join("adapter_config.json"),
            serde_json::json!({"r": r, "lora_alpha": alpha, "peft_type": "LORA"}).to_string(),
        )
        .unwrap();

        let base_before =
            candle_core::safetensors::load(model_dir.join("model.safetensors"), &device).unwrap();
        let w_key = format!("{module}.weight");
        let before = base_before
            .get(&w_key)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let config = AxolotlConfig {
            base_model: model_dir.to_string_lossy().to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings {
                r,
                alpha,
                dropout: 0.0,
                target_modules: vec!["q_proj".into()],
            },
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: temp_dir.path().join("out").to_string_lossy().to_string(),
            seed: 1,
        };
        let out_dir = temp_dir.path().join("merged");
        merge_adapter(
            &config,
            adapter_dir.to_str().unwrap(),
            out_dir.to_str().unwrap(),
        )
        .expect("merge must succeed");

        assert!(out_dir.join("model.safetensors").exists());
        assert!(out_dir.join("config.json").exists());
        assert!(out_dir.join("merge_info.json").exists());

        let merged =
            candle_core::safetensors::load(out_dir.join("model.safetensors"), &device).unwrap();
        let after = merged
            .get(&w_key)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(before.len(), after.len());
        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(changed, "merged q_proj.weight must differ from base");

        // Analytic check: delta = B@A * scale; each row of B@A is 0.5 * r ones vector
        // so each element of delta is 0.5 * r * scale
        let expected_delta = 0.5f32 * (r as f32) * (scale as f32);
        let max_err = before
            .iter()
            .zip(after.iter())
            .map(|(b0, a0)| (a0 - (b0 + expected_delta)).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "merge math mismatch max_err={max_err}, expected_delta={expected_delta}"
        );
    }

    /// Local path is first-class for download helper (no network).
    #[test]
    #[cfg(feature = "download")]
    fn test_download_model_local_path_passthrough() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path().join("local_model");
        crate::fixture::write_tiny_llama_fixture(
            &model_dir,
            crate::fixture::TinyLlamaSpec::default(),
        )
        .unwrap();
        let resolved = download_model(
            model_dir.to_str().unwrap(),
            temp_dir.path().join("cache").to_str().unwrap(),
        )
        .expect("local path must resolve");
        assert_eq!(PathBuf::from(&resolved), model_dir);
    }
}
