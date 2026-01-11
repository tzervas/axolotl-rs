//! Model loading and adapter merging.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks};
use std::path::PathBuf;

use crate::config::{AdapterType, AxolotlConfig, DatasetConfig, LoraSettings, QuantizationSettings, QuantType, TrainingConfig};
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
    config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
) -> Result<Box<dyn Module>> {
    // Detect model architecture from base_model name
    if config.base_model.to_lowercase().contains("llama") {
        load_llama_model(config, model_path, device, dtype)
    } else {
        // For other architectures, use stub for now
        tracing::warn!("Architecture not supported yet: {}, using stub model", config.base_model);
        let vb = VarBuilder::zeros(dtype, device);
        let model = SimpleModel::new(vb)?;
        Ok(Box::new(model))
    }
}

/// Load a LLaMA model from the given path.
fn load_llama_model(
    config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
) -> Result<Box<dyn Module>> {
    // Try to load config.json first
    let config_path = model_path.join("config.json");
    let llama_config: LlamaConfig = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {}", e)))?;
        serde_json::from_str(&config_str)
            .map_err(|e| AxolotlError::Model(format!("Failed to parse config.json: {}", e)))?
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

    // Load model weights
    let vb = if model_path.join("model.safetensors").exists() {
        let tensors = candle_core::safetensors::load(model_path.join("model.safetensors"), device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load safetensors: {}", e)))?;
        VarBuilder::from_tensors(tensors, dtype, device)
    } else if model_path.join("pytorch_model.bin").exists() {
        VarBuilder::from_pth(model_path.join("pytorch_model.bin"), dtype, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load pytorch model: {}", e)))?
    } else {
        return Err(AxolotlError::Model(format!(
            "No model weights found in {}. Expected model.safetensors or pytorch_model.bin",
            model_path.display()
        )));
    };

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

    // Create LLaMA model
    let model = Llama::load(vb, &config)
        .map_err(|e| AxolotlError::Model(format!("Failed to create LLaMA model: {}", e)))?;

    tracing::info!("Loaded LLaMA model with {} layers, {} hidden size", llama_config.num_hidden_layers, llama_config.hidden_size);

    Ok(Box::new(LlamaWrapper::new(model, &config, device)?))
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

/// Wrapper for LLaMA model that implements the Module trait.
pub struct LlamaWrapper {
    model: Llama,
    cache: std::cell::RefCell<Cache>,
}

impl LlamaWrapper {
    pub fn new(model: Llama, config: &candle_transformers::models::llama::Config, device: &Device) -> Result<Self> {
        let cache = Cache::new(false, DType::F32, config, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to create cache: {}", e)))?;
        Ok(Self {
            model,
            cache: std::cell::RefCell::new(cache),
        })
    }
}

impl Module for LlamaWrapper {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut cache = self.cache.borrow_mut();
        // For inference, we start from position 0
        self.model.forward(xs, 0, &mut cache)
    }
}

/// Merge adapter weights into base model.
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

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

        // Currently returns "Model loading not yet implemented" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            println!("Actual error message: {}", msg);
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

        // Currently returns "Model loading not yet implemented" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            assert!(msg.contains("Model not found"));
        } else {
            panic!("Expected Model error");
        }
    }

    /// Test applying a LoRA adapter to a base model.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_apply_adapter_lora() {
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

        // Currently returns "Adapter merging not yet implemented" error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => assert!(msg.contains("Adapter merging not yet implemented")),
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test applying a QLoRA adapter with quantization.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_apply_adapter_qlora() {
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

        // Currently returns "Adapter merging not yet implemented" error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => assert!(msg.contains("Adapter merging not yet implemented")),
            _ => panic!("Expected Model error, got {:?}", err),
        }
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

        // Currently returns "Adapter merging not yet implemented" error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => assert!(msg.contains("Adapter merging not yet implemented")),
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test downloading model from HuggingFace Hub.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    #[cfg(feature = "download")]
    fn test_download_model_from_hub() {
        // Currently returns "Model download not yet implemented" error
        let result: Result<String> = tokio::runtime::Runtime::new().unwrap().block_on(async {
            download_model("meta-llama/Llama-2-7b-hf", "/tmp/cache").await
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => assert!(msg.contains("Model download not yet implemented")),
            _ => panic!("Expected Model error, got {:?}", err),
        }
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
            AxolotlError::Tokenizer(e) => assert!(e.to_string().contains("tokenizer.json not found")),
            _ => panic!("Expected Tokenizer error, got {:?}", err),
        }
    }

    /// Test model architecture loading with stub implementation.
    #[test]
    fn test_load_model_architecture_stub() {
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

        let result = load_model_architecture(&config, &temp_dir.path().to_path_buf(), &device, dtype);
        assert!(result.is_ok());

        let model = result.unwrap();
        // Test that the stub model can perform forward pass
        let input = Tensor::zeros((1, 10), dtype, &device).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }
}
