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

    /// QLoRA error.
    #[error("QLoRA error: {0}")]
    Qlora(#[from] qlora_rs::QLoraError),

    /// Candle error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Tokenizer error.
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}
