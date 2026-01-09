//! Error types for axolotl-rs.

use thiserror::Error;

/// Result type alias for axolotl-rs operations.
pub type Result<T> = std::result::Result<T, AxolotlError>;

/// Errors that can occur in axolotl-rs.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum AxolotlError {
    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Invalid configuration file.
    #[error("invalid config file: {0}")]
    ConfigParse(#[from] serde_yaml::Error),

    /// Model loading error.
    #[error("model error: {0}")]
    Model(String),

    /// Dataset error.
    #[error("dataset error: {0}")]
    Dataset(String),

    /// Training error.
    #[error("training error: {0}")]
    Training(String),

    /// Checkpoint error.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// PEFT error.
    #[error("PEFT error: {0}")]
    Peft(#[from] peft_rs::PeftError),

    /// `QLoRA` error.
    #[error("QLoRA error: {0}")]
    Qlora(#[from] qlora_rs::QLoraError),

    /// Candle error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Tokenizer error.
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    /// Progress bar template error.
    #[error("template error: {0}")]
    Template(#[from] indicatif::style::TemplateError),

    /// Other errors.
    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_config_error_creation() {
        let error = AxolotlError::Config("invalid parameter".to_string());
        assert_eq!(error.to_string(), "configuration error: invalid parameter");
    }

    #[test]
    fn test_model_error_creation() {
        let error = AxolotlError::Model("model loading failed".to_string());
        assert_eq!(error.to_string(), "model error: model loading failed");
    }

    #[test]
    fn test_dataset_error_creation() {
        let error = AxolotlError::Dataset("dataset not found".to_string());
        assert_eq!(error.to_string(), "dataset error: dataset not found");
    }

    #[test]
    fn test_training_error_creation() {
        let error = AxolotlError::Training("training failed".to_string());
        assert_eq!(error.to_string(), "training error: training failed");
    }

    #[test]
    fn test_checkpoint_error_creation() {
        let error = AxolotlError::Checkpoint("checkpoint save failed".to_string());
        assert_eq!(
            error.to_string(),
            "checkpoint error: checkpoint save failed"
        );
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let error: AxolotlError = io_error.into();
        assert!(error.to_string().contains("IO error"));
        assert!(error.to_string().contains("file not found"));
    }

    #[test]
    fn test_config_parse_error_conversion() {
        let yaml_str = "invalid: yaml: :::";
        let yaml_error = serde_yaml::from_str::<serde_yaml::Value>(yaml_str).unwrap_err();
        let error: AxolotlError = yaml_error.into();
        assert!(error.to_string().contains("invalid config file"));
    }

    #[test]
    fn test_candle_error_conversion() {
        // Create a candle error by attempting an invalid operation
        use candle_core::{DType, Device, Tensor};

        // This will create a candle error (shape mismatch)
        let tensor1 = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        let tensor2 = Tensor::zeros((3, 4), DType::F32, &Device::Cpu).unwrap();

        // Attempting to add tensors with incompatible shapes
        let candle_error = tensor1.broadcast_add(&tensor2).unwrap_err();
        let error: AxolotlError = candle_error.into();
        assert!(error.to_string().contains("candle error"));
    }

    #[test]
    fn test_peft_error_conversion() {
        // Create a peft error
        let peft_error = peft_rs::PeftError::InvalidConfig("invalid config".to_string());
        let error: AxolotlError = peft_error.into();
        assert!(error.to_string().contains("PEFT error"));
    }

    #[test]
    fn test_qlora_error_conversion() {
        // Create a qlora error
        let qlora_error = qlora_rs::QLoraError::InvalidConfig("invalid qlora config".to_string());
        let error: AxolotlError = qlora_error.into();
        assert!(error.to_string().contains("QLoRA error"));
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = AxolotlError::Config("test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("Config"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_error_display_formatting() {
        let error = AxolotlError::Training("epoch failed".to_string());
        let display_str = format!("{}", error);
        assert_eq!(display_str, "training error: epoch failed");
    }

    #[test]
    fn test_io_error_from_implementation() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let error = AxolotlError::from(io_err);
        match error {
            AxolotlError::Io(_) => (),
            _ => panic!("Expected Io variant"),
        }
    }

    #[test]
    fn test_error_source_chain() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file.txt not found");
        let error: AxolotlError = io_error.into();

        // Test that the error has a source
        use std::error::Error;
        assert!(error.source().is_some());
    }

    #[test]
    fn test_multiple_error_variants() {
        let errors = vec![
            AxolotlError::Config("config".to_string()),
            AxolotlError::Model("model".to_string()),
            AxolotlError::Dataset("dataset".to_string()),
            AxolotlError::Training("training".to_string()),
            AxolotlError::Checkpoint("checkpoint".to_string()),
        ];

        assert_eq!(errors.len(), 5);
        assert!(errors[0].to_string().contains("configuration error"));
        assert!(errors[1].to_string().contains("model error"));
        assert!(errors[2].to_string().contains("dataset error"));
        assert!(errors[3].to_string().contains("training error"));
        assert!(errors[4].to_string().contains("checkpoint error"));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_result() -> Result<i32> {
            Ok(42)
        }

        fn returns_error() -> Result<i32> {
            Err(AxolotlError::Config("test error".to_string()))
        }

        assert!(returns_result().is_ok());
        assert!(returns_error().is_err());
        assert_eq!(returns_result().unwrap(), 42);
    }

    #[test]
    fn test_tokenizer_error_conversion() {
        // Create a tokenizer error by attempting invalid operations
        use tokenizers::Tokenizer;

        // Attempting to load from invalid path will create an IO error that wraps into tokenizer error
        let result = Tokenizer::from_file("/nonexistent/path/to/tokenizer.json");
        if let Err(tokenizer_error) = result {
            let error: AxolotlError = tokenizer_error.into();
            assert!(error.to_string().contains("tokenizer error"));
        }
    }

    #[test]
    fn test_other_error_creation() {
        let error = AxolotlError::Other("some other error".to_string());
        assert_eq!(error.to_string(), "some other error");
    }

    #[test]
    fn test_boxed_error_conversion() {
        // Test that various error types can be converted
        let io_error = std::io::Error::new(std::io::ErrorKind::Other, "test");
        let axolotl_error: AxolotlError = io_error.into();
        assert!(matches!(axolotl_error, AxolotlError::Io(_)));
    }

    #[test]
    fn test_template_error_conversion() {
        use indicatif::ProgressStyle;

        // Create an invalid template to generate TemplateError
        let result = ProgressStyle::default_bar().template("{invalid_placeholder}");

        if let Err(template_error) = result {
            let error: AxolotlError = template_error.into();
            assert!(error.to_string().contains("template error"));
        }
    }
}
