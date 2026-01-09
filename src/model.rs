//! Model loading and adapter merging.

use candle_core::Device;

use crate::config::AxolotlConfig;
use crate::error::{AxolotlError, Result};

/// Load a model from the configuration.
///
/// # Errors
///
/// Returns an error if the model cannot be loaded or initialized.
pub fn load_model(_config: &AxolotlConfig, _device: &Device) -> Result<()> {
    // TODO: Implement model loading
    // 1. Download/load base model weights
    // 2. Initialize adapter if specified
    // 3. Quantize if QLoRA
    Err(AxolotlError::Model(
        "Model loading not yet implemented".into(),
    ))
}

/// Merge adapter weights into base model.
///
/// # Errors
///
/// Returns an error if the adapter or base model cannot be loaded or merged.
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

/// Download model from `HuggingFace` Hub.
///
/// # Errors
///
/// Returns an error if the model cannot be downloaded.
#[cfg(feature = "download")]
pub fn download_model(_model_id: &str, _cache_dir: &str) -> Result<String> {
    // TODO: Implement model download
    Err(AxolotlError::Model(
        "Model download not yet implemented".into(),
    ))
}
