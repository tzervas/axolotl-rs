//! Error types for axolotl-rs.
//!
//! This module provides error types and result aliases for the library.
//!
//! # Example - Error Handling
//!
//! ```rust
//! use axolotl_rs::{AxolotlConfig, AxolotlError, Result};
//!
//! fn try_load_config(path: &str) -> Result<AxolotlConfig> {
//!     match AxolotlConfig::from_file(path) {
//!         Ok(config) => Ok(config),
//!         Err(e) => {
//!             eprintln!("Failed to load config: {}", e);
//!             Err(e)
//!         }
//!     }
//! }
//! ```
//!
//! # Example - Pattern Matching
//!
//! ```rust
//! use axolotl_rs::{AxolotlConfig, AxolotlError};
//!
//! # fn main() {
//! match AxolotlConfig::from_preset("invalid-preset") {
//!     Ok(config) => println!("Loaded config"),
//!     Err(AxolotlError::Config(msg)) => {
//!         eprintln!("Configuration error: {}", msg);
//!     }
//!     Err(e) => eprintln!("Other error: {}", e),
//! }
//! # }
//! ```
//!
//! # Example - Using Result Type
//!
//! ```no_run
//! use axolotl_rs::{AxolotlConfig, Trainer, Result};
//!
//! fn train_model() -> Result<()> {
//!     let config = AxolotlConfig::from_file("config.yaml")?;
//!     let mut trainer = Trainer::new(config)?;
//!     trainer.train()?;
//!     Ok(())
//! }
//! ```

use thiserror::Error;

/// Result type alias for axolotl-rs operations.
///
/// # Example
///
/// ```rust
/// use axolotl_rs::{AxolotlConfig, Result};
///
/// fn load_and_validate(path: &str) -> Result<AxolotlConfig> {
///     let config = AxolotlConfig::from_file(path)?;
///     config.validate()?;
///     Ok(config)
/// }
/// ```
pub type Result<T> = std::result::Result<T, AxolotlError>;

/// Errors that can occur in axolotl-rs.
///
/// # Example
///
/// ```rust
/// use axolotl_rs::{AxolotlError, Result};
///
/// fn validate_path(path: &str) -> Result<()> {
///     if path.is_empty() {
///         return Err(AxolotlError::Config("Path cannot be empty".to_string()));
///     }
///     Ok(())
/// }
///
/// assert!(validate_path("").is_err());
/// assert!(validate_path("/valid/path").is_ok());
/// ```
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
