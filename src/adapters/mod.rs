//! Adapter integration layer.
//!
//! This module provides unified access to PEFT adapters (LoRA, QLoRA, etc.)
//! using either the real peft-rs/qlora-rs crates or mock implementations.

use candle_core::Device;
use candle_nn::VarMap;

use crate::config::{AdapterType, AxolotlConfig, LoraSettings, QuantizationSettings};
use crate::error::{AxolotlError, Result};

// Re-export based on features
#[cfg(feature = "peft")]
pub use peft_rs::{
    Adapter, AdapterConfig, LoraConfig as PeftLoraConfig, LoraLayer, Mergeable, PeftModel,
    Trainable,
};

#[cfg(feature = "qlora")]
pub use qlora_rs::{QLoraConfig, QLoraLayer, QuantizationConfig, QuantizedLinear, QuantizedTensor};

/// Unified adapter wrapper that works with both real and mock implementations.
pub struct AdapterWrapper {
    /// The type of adapter being used
    pub adapter_type: AdapterType,
    /// Whether quantization is enabled
    pub quantized: bool,
    /// Trainable parameters (LoRA weights)
    pub trainable_params: VarMap,
    /// Device where adapter is loaded
    pub device: Device,
}

impl AdapterWrapper {
    /// Create a new adapter based on configuration.
    ///
    /// # Arguments
    /// * `config` - The axolotl configuration
    /// * `device` - Device to create adapter on
    ///
    /// # Errors
    /// Returns an error if the adapter cannot be created.
    pub fn new(config: &AxolotlConfig, device: &Device) -> Result<Self> {
        let trainable_params = VarMap::new();

        match config.adapter {
            AdapterType::None => Ok(Self {
                adapter_type: AdapterType::None,
                quantized: false,
                trainable_params,
                device: device.clone(),
            }),
            AdapterType::Lora => {
                tracing::info!(
                    "Creating LoRA adapter with r={}, alpha={}",
                    config.lora.r,
                    config.lora.alpha
                );
                Ok(Self {
                    adapter_type: AdapterType::Lora,
                    quantized: false,
                    trainable_params,
                    device: device.clone(),
                })
            }
            AdapterType::Qlora => {
                if config.quantization.is_none() {
                    return Err(AxolotlError::Config(
                        "QLoRA requires quantization settings".into(),
                    ));
                }
                tracing::info!(
                    "Creating QLoRA adapter with r={}, alpha={}, quantization enabled",
                    config.lora.r,
                    config.lora.alpha
                );
                Ok(Self {
                    adapter_type: AdapterType::Qlora,
                    quantized: true,
                    trainable_params,
                    device: device.clone(),
                })
            }
        }
    }

    /// Convert axolotl LoRA settings to peft-rs config.
    #[cfg(feature = "peft")]
    pub fn to_peft_lora_config(settings: &LoraSettings) -> PeftLoraConfig {
        PeftLoraConfig {
            r: settings.r,
            alpha: settings.alpha as usize,
            dropout: settings.dropout as f64,
            target_modules: settings.target_modules.clone(),
            ..Default::default()
        }
    }

    /// Convert axolotl quantization settings to qlora-rs config.
    #[cfg(feature = "qlora")]
    pub fn to_qlora_config(
        lora: &LoraSettings,
        quant: &QuantizationSettings,
    ) -> Result<QLoraConfig> {
        let quant_config = QuantizationConfig {
            block_size: quant.block_size,
            double_quant: quant.double_quant,
            ..Default::default()
        };

        let lora_config = Self::to_peft_lora_config(lora);

        Ok(QLoraConfig {
            lora: lora_config,
            quantization: quant_config,
        })
    }

    /// Get the number of trainable parameters.
    pub fn trainable_param_count(&self) -> usize {
        self.trainable_params
            .all_vars()
            .iter()
            .map(|v| v.elem_count())
            .sum()
    }

    /// Apply adapter to a linear layer, returning a wrapped layer.
    #[cfg(feature = "peft")]
    pub fn wrap_linear(
        &self,
        in_features: usize,
        out_features: usize,
        lora_config: &PeftLoraConfig,
        vb: candle_nn::VarBuilder,
    ) -> Result<LoraLayer> {
        LoraLayer::new(in_features, out_features, lora_config.clone(), vb)
            .map_err(|e| AxolotlError::Model(format!("Failed to create LoRA layer: {}", e)))
    }
}

/// Configuration for applying adapters to a model.
#[derive(Debug, Clone)]
pub struct AdapterApplicationConfig {
    /// Target module patterns (e.g., "q_proj", "v_proj")
    pub target_modules: Vec<String>,
    /// LoRA rank
    pub r: usize,
    /// LoRA alpha scaling
    pub alpha: usize,
    /// LoRA dropout
    pub dropout: f32,
}

impl From<&LoraSettings> for AdapterApplicationConfig {
    fn from(settings: &LoraSettings) -> Self {
        Self {
            target_modules: settings.target_modules.clone(),
            r: settings.r,
            alpha: settings.alpha as usize,
            dropout: settings.dropout as f32,
        }
    }
}
