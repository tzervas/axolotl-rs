//! # axolotl-rs
//!
//! YAML-driven configurable fine-tuning toolkit for LLMs.
//!
//! This crate provides a user-friendly interface for fine-tuning language models,
//! similar to the Python Axolotl project but in pure Rust.
//!
//! ## Features
//!
//! - **YAML Configuration** - Define entire training runs in simple config files
//! - **Multiple Adapters** - Support for `LoRA`, `QLoRA`, full fine-tuning
//! - **Dataset Handling** - Automatic loading and preprocessing
//! - **Multi-GPU** - Distributed training support (planned)
//!
//! ## Quick Start
//!
//! ```bash
//! # Validate configuration
//! axolotl validate config.yaml
//!
//! # Start training
//! axolotl train config.yaml
//!
//! # Merge adapters
//! axolotl merge --config config.yaml --output ./merged-model
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod cli;
pub mod config;
pub mod dataset;
pub mod error;
pub mod model;
pub mod optimizer;
pub mod scheduler;
pub mod trainer;

// Mock modules for testing without external dependencies
#[cfg(any(
    feature = "mock-peft",
    feature = "mock-qlora",
    feature = "mock-unsloth"
))]
pub mod mocks;

pub use config::{AxolotlConfig, TrainingConfig};
pub use error::{AxolotlError, Result};
pub use trainer::Trainer;
