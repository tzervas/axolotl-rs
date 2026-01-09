//! Configuration parsing and validation.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AxolotlError, Result};

/// Main configuration for Axolotl training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxolotlConfig {
    /// Base model identifier (`HuggingFace` model ID or local path).
    pub base_model: String,

    /// Adapter type.
    #[serde(default)]
    pub adapter: AdapterType,

    /// `LoRA` configuration (if using LoRA/QLoRA).
    #[serde(default)]
    pub lora: LoraSettings,

    /// Quantization configuration (if using `QLoRA`).
    #[serde(default)]
    pub quantization: Option<QuantizationSettings>,

    /// Dataset configuration.
    pub dataset: DatasetConfig,

    /// Training hyperparameters.
    #[serde(default)]
    pub training: TrainingConfig,

    /// Output directory.
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_output_dir() -> String {
    "./outputs".into()
}

fn default_seed() -> u64 {
    42
}

/// Adapter type for fine-tuning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdapterType {
    /// No adapter (full fine-tuning).
    None,
    /// Standard `LoRA`.
    #[default]
    Lora,
    /// 4-bit quantized `LoRA`.
    Qlora,
}

/// LoRA-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraSettings {
    /// Rank of low-rank decomposition.
    #[serde(default = "default_lora_r")]
    pub r: usize,

    /// Scaling factor.
    #[serde(default = "default_lora_alpha")]
    pub alpha: usize,

    /// Dropout probability.
    #[serde(default)]
    pub dropout: f64,

    /// Target modules for `LoRA`.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
}

fn default_lora_r() -> usize {
    64
}
fn default_lora_alpha() -> usize {
    16
}
fn default_target_modules() -> Vec<String> {
    vec![
        "q_proj".into(),
        "k_proj".into(),
        "v_proj".into(),
        "o_proj".into(),
    ]
}

impl Default for LoraSettings {
    fn default() -> Self {
        Self {
            r: default_lora_r(),
            alpha: default_lora_alpha(),
            dropout: 0.05,
            target_modules: default_target_modules(),
        }
    }
}

/// Quantization settings for `QLoRA`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSettings {
    /// Number of bits (4 for `QLoRA`).
    #[serde(default = "default_bits")]
    pub bits: u8,

    /// Quantization type.
    #[serde(default)]
    pub quant_type: QuantType,

    /// Use double quantization.
    #[serde(default = "default_true")]
    pub double_quant: bool,

    /// Block size for quantization.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
}

fn default_bits() -> u8 {
    4
}
fn default_true() -> bool {
    true
}
fn default_block_size() -> usize {
    64
}

impl Default for QuantizationSettings {
    fn default() -> Self {
        Self {
            bits: 4,
            quant_type: QuantType::Nf4,
            double_quant: true,
            block_size: 64,
        }
    }
}

/// Quantization type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantType {
    /// 4-bit `NormalFloat`.
    #[default]
    Nf4,
    /// 4-bit float point.
    Fp4,
}

/// Dataset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Path to dataset (local file or `HuggingFace` dataset ID).
    pub path: String,

    /// Dataset format type.
    #[serde(default)]
    pub format: DatasetFormat,

    /// Field containing input text.
    #[serde(default = "default_input_field")]
    pub input_field: String,

    /// Field containing output text.
    #[serde(default = "default_output_field")]
    pub output_field: String,

    /// Maximum sequence length.
    #[serde(default = "default_max_length")]
    pub max_length: usize,

    /// Validation split ratio.
    #[serde(default = "default_val_split")]
    pub val_split: f32,
}

fn default_input_field() -> String {
    "instruction".into()
}
fn default_output_field() -> String {
    "output".into()
}
fn default_max_length() -> usize {
    2048
}
fn default_val_split() -> f32 {
    0.05
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            format: DatasetFormat::Alpaca,
            input_field: default_input_field(),
            output_field: default_output_field(),
            max_length: default_max_length(),
            val_split: default_val_split(),
        }
    }
}

/// Dataset format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetFormat {
    /// Alpaca format: instruction, input, output.
    #[default]
    Alpaca,
    /// `ShareGPT` format: conversations array.
    Sharegpt,
    /// Simple completion: just text.
    Completion,
    /// Custom format with specified fields.
    Custom,
}

/// Training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs.
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// Batch size per device.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Gradient accumulation steps.
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,

    /// Learning rate.
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

    /// Learning rate scheduler.
    #[serde(default)]
    pub lr_scheduler: LrScheduler,

    /// Warmup ratio.
    #[serde(default = "default_warmup")]
    pub warmup_ratio: f32,

    /// Weight decay.
    #[serde(default)]
    pub weight_decay: f64,

    /// Maximum gradient norm for clipping.
    #[serde(default = "default_grad_norm")]
    pub max_grad_norm: f32,

    /// Save checkpoint every N steps.
    #[serde(default = "default_save_steps")]
    pub save_steps: usize,

    /// Log every N steps.
    #[serde(default = "default_log_steps")]
    pub logging_steps: usize,

    /// Use gradient checkpointing.
    #[serde(default)]
    pub gradient_checkpointing: bool,

    /// Use mixed precision training.
    #[serde(default = "default_true")]
    pub mixed_precision: bool,
}

fn default_epochs() -> usize {
    3
}
fn default_batch_size() -> usize {
    4
}
fn default_grad_accum() -> usize {
    4
}
fn default_lr() -> f64 {
    2e-4
}
fn default_warmup() -> f32 {
    0.03
}
fn default_grad_norm() -> f32 {
    1.0
}
fn default_save_steps() -> usize {
    500
}
fn default_log_steps() -> usize {
    10
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            gradient_accumulation_steps: default_grad_accum(),
            learning_rate: default_lr(),
            lr_scheduler: LrScheduler::Cosine,
            warmup_ratio: default_warmup(),
            weight_decay: 0.0,
            max_grad_norm: default_grad_norm(),
            save_steps: default_save_steps(),
            logging_steps: default_log_steps(),
            gradient_checkpointing: false,
            mixed_precision: true,
        }
    }
}

/// Learning rate scheduler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LrScheduler {
    /// Cosine annealing.
    #[default]
    Cosine,
    /// Linear decay.
    Linear,
    /// Constant learning rate.
    Constant,
}

impl AxolotlConfig {
    /// Load configuration from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a YAML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create a configuration from a preset.
    ///
    /// # Errors
    ///
    /// Returns an error if the preset name is unknown.
    pub fn from_preset(preset: &str) -> Result<Self> {
        match preset {
            "llama2-7b" => Ok(Self::llama2_7b_preset()),
            "mistral-7b" => Ok(Self::mistral_7b_preset()),
            "phi3-mini" => Ok(Self::phi3_mini_preset()),
            _ => Err(AxolotlError::Config(format!("Unknown preset: {preset}"))),
        }
    }

    fn llama2_7b_preset() -> Self {
        Self {
            base_model: "meta-llama/Llama-2-7b-hf".into(),
            adapter: AdapterType::Qlora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                ..Default::default()
            },
            quantization: Some(QuantizationSettings::default()),
            dataset: DatasetConfig {
                path: "./data/train.jsonl".into(),
                ..Default::default()
            },
            training: TrainingConfig {
                learning_rate: 2e-4,
                ..Default::default()
            },
            output_dir: "./outputs/llama2-7b-qlora".into(),
            seed: 42,
        }
    }

    fn mistral_7b_preset() -> Self {
        Self {
            base_model: "mistralai/Mistral-7B-v0.1".into(),
            adapter: AdapterType::Qlora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                target_modules: vec![
                    "q_proj".into(),
                    "k_proj".into(),
                    "v_proj".into(),
                    "o_proj".into(),
                    "gate_proj".into(),
                    "up_proj".into(),
                    "down_proj".into(),
                ],
                ..Default::default()
            },
            quantization: Some(QuantizationSettings::default()),
            dataset: DatasetConfig {
                path: "./data/train.jsonl".into(),
                ..Default::default()
            },
            training: TrainingConfig::default(),
            output_dir: "./outputs/mistral-7b-qlora".into(),
            seed: 42,
        }
    }

    fn phi3_mini_preset() -> Self {
        Self {
            base_model: "microsoft/phi-3-mini-4k-instruct".into(),
            adapter: AdapterType::Lora,
            lora: LoraSettings {
                r: 32,
                alpha: 16,
                ..Default::default()
            },
            quantization: None,
            dataset: DatasetConfig {
                path: "./data/train.jsonl".into(),
                max_length: 4096,
                ..Default::default()
            },
            training: TrainingConfig {
                learning_rate: 1e-4,
                ..Default::default()
            },
            output_dir: "./outputs/phi3-mini-lora".into(),
            seed: 42,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.base_model.is_empty() {
            return Err(AxolotlError::Config("base_model is required".into()));
        }

        if self.dataset.path.is_empty() {
            return Err(AxolotlError::Config("dataset.path is required".into()));
        }

        if self.lora.r == 0 {
            return Err(AxolotlError::Config("lora.r must be > 0".into()));
        }

        if matches!(self.adapter, AdapterType::Qlora) && self.quantization.is_none() {
            return Err(AxolotlError::Config(
                "quantization config required for QLoRA".into(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serialization() {
        let config = AxolotlConfig::llama2_7b_preset();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AxolotlConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.base_model, restored.base_model);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AxolotlConfig::llama2_7b_preset();
        assert!(config.validate().is_ok());

        config.base_model = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_presets() {
        assert!(AxolotlConfig::from_preset("llama2-7b").is_ok());
        assert!(AxolotlConfig::from_preset("mistral-7b").is_ok());
        assert!(AxolotlConfig::from_preset("phi3-mini").is_ok());
        assert!(AxolotlConfig::from_preset("invalid").is_err());
    }
}
