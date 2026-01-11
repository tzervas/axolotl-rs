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
    #[allow(dead_code)]
    Training(String),

    /// Checkpoint error.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// PEFT error (mock).
    #[error("PEFT error: {0}")]
    #[allow(dead_code)]
    Peft(String),

    /// `QLoRA` error (mock).
    #[error("QLoRA error: {0}")]
    #[allow(dead_code)]
    Qlora(String),

    /// Candle error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Tokenizer error.
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    /// Progress bar template error.
    #[error("template error: {0}")]
    Template(String),
}

impl From<indicatif::style::TemplateError> for AxolotlError {
    fn from(err: indicatif::style::TemplateError) -> Self {
        AxolotlError::Template(err.to_string())
    }
}
