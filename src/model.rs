//! Model loading and adapter merging.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use std::path::PathBuf;

use crate::config::AxolotlConfig;
use crate::error::{AxolotlError, Result};

/// Loaded model with configuration.
pub struct LoadedModel {
    /// Model weights and forward pass
    pub model: Box<dyn Module>,
    /// Tokenizer
    pub tokenizer: tokenizers::Tokenizer,
    /// Device where model is loaded
    pub device: Device,
    /// Model dtype
    pub dtype: DType,
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
            .map_err(|e| AxolotlError::Model(format!("Forward pass failed: {}", e)))
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
    
    // Load tokenizer
    let tokenizer = load_tokenizer(&model_path)?;
    tracing::info!("Loaded tokenizer with vocab size: {}", tokenizer.get_vocab_size(true));
    
    // Determine dtype
    let dtype = if config.quantization.is_some() {
        DType::F16 // Use F16 for quantized models
    } else {
        DType::F32
    };
    
    // Load model weights based on architecture
    let model = load_model_architecture(config, &model_path, device, dtype)?;
    
    tracing::info!("Model loaded successfully on {:?} with dtype {:?}", device, dtype);
    
    Ok(LoadedModel {
        model,
        tokenizer,
        device: device.clone(),
        dtype,
    })
}

/// Resolve model path from HuggingFace model ID or local path.
fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    // Check if it's a local path
    let path = PathBuf::from(model_id);
    if path.exists() {
        return Ok(path);
    }
    
    // Try HuggingFace cache directory
    let cache_dir = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.cache/huggingface", h)))
        .unwrap_or_else(|_| "/tmp/huggingface".to_string());
    
    let hf_path = PathBuf::from(format!("{}/hub/models--{}", 
        cache_dir, 
        model_id.replace("/", "--")
    ));
    
    if hf_path.exists() {
        Ok(hf_path)
    } else {
        Err(AxolotlError::Model(format!(
            "Model not found at '{}' or in HF cache at '{:?}'. Use `huggingface-cli download {}` to download.",
            model_id, hf_path, model_id
        )))
    }
}

/// Load tokenizer from model directory.
fn load_tokenizer(model_path: &PathBuf) -> Result<tokenizers::Tokenizer> {
    let tokenizer_file = model_path.join("tokenizer.json");
    
    if !tokenizer_file.exists() {
        return Err(AxolotlError::Tokenizer(
            format!("tokenizer.json not found in {:?}", model_path).into()
        ));
    }
    
    tokenizers::Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| AxolotlError::Tokenizer(format!("Failed to load tokenizer: {}", e).into()))
}

/// Load model architecture based on config.
fn load_model_architecture(
    _config: &AxolotlConfig,
    _model_path: &PathBuf,
    device: &Device,
    dtype: DType,
) -> Result<Box<dyn Module>> {
    // For now, we'll use a simple stub that returns a working module
    // In Phase 2, this will detect architecture and load proper models
    
    let vb = VarBuilder::zeros(dtype, device);
    
    // Create a simple pass-through module for testing
    let model = SimpleModel::new(vb)?;
    
    tracing::warn!("Using stub model architecture - full implementation pending");
    
    Ok(Box::new(model))
}

/// Simple stub model for testing.
struct SimpleModel {
    #[allow(dead_code)]
    weight: Tensor,
}

impl SimpleModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((768, 768), "weight")
            .map_err(|e| AxolotlError::Model(format!("Failed to create weight: {}", e)))?;
        Ok(Self { weight })
    }
}

impl Module for SimpleModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Simple pass-through for now
        Ok(xs.clone())
    }
}


/// Merge adapter weights into base model.
///
/// # Errors
///
/// Returns an error as adapter merging is not yet implemented.
pub fn merge_adapter(
    _config: &AxolotlConfig,
    _adapter_path: &str,
    _output_path: &str,
) -> Result<()> {
    // TODO: Implement adapter merging
    // 1. Load base model weights
    // 2. Load adapter weights
    // 3. Merge using LoRA merge formula: W' = W + BA * scaling
    // 4. Save merged weights
    Err(AxolotlError::Model(
        "Adapter merging not yet implemented".into(),
    ))
}

/// Download model from HuggingFace Hub.
#[cfg(feature = "download")]
pub async fn download_model(_model_id: &str, _cache_dir: &str) -> Result<String> {
    // TODO: Implement model download
    Err(AxolotlError::Model(
        "Model download not yet implemented".into(),
    ))
}
