//! CLI entry point for axolotl-rs.

#![allow(dead_code)]

use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod cli;
mod config;
mod dataset;
mod error;
mod fixture;
#[cfg(feature = "peft")]
mod llama_common;
#[cfg(feature = "peft")]
mod lora_llama;
mod model;
mod optimizer;
#[cfg(all(feature = "peft", feature = "qlora"))]
mod qlora_llama;
mod scheduler;
mod trainer;

use config::AxolotlConfig;
use error::Result;

#[derive(Parser)]
#[command(name = "axolotl")]
#[command(about = "YAML-driven fine-tuning toolkit for LLMs")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a configuration file
    Validate {
        /// Path to configuration file
        config: String,
    },
    /// Start training
    Train {
        /// Path to configuration file
        config: String,
        /// Resume from checkpoint
        #[arg(long)]
        resume: Option<String>,
    },
    /// Merge LoRA adapter weights into base model linear weights
    Merge {
        /// Path to configuration file
        #[arg(long)]
        config: String,
        /// Path to adapter checkpoint directory (or adapter_model.safetensors)
        #[arg(long)]
        adapter: Option<String>,
        /// Output directory for merged model
        #[arg(long)]
        output: String,
    },
    /// Download model weights from HuggingFace Hub into a local directory
    Download {
        /// HuggingFace model id (e.g. HuggingFaceTB/SmolLM2-135M)
        model_id: String,
        /// Output / cache directory (model is written under `<output>/<sanitized-id>/`)
        #[arg(long, default_value = "./models")]
        output: String,
    },
    /// Generate a sample configuration file
    Init {
        /// Output path for config file
        #[arg(default_value = "config.yaml")]
        output: String,
        /// Model preset (llama2-7b, mistral-7b, phi3-mini)
        #[arg(long, default_value = "llama2-7b")]
        preset: String,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Validate { config } => {
            tracing::info!("Validating configuration: {}", config);
            let config = AxolotlConfig::from_file(&config)?;
            config.validate()?;
            println!("✓ Configuration is valid");
            println!("  Model: {}", config.base_model);
            println!("  Adapter: {:?}", config.adapter);
            println!("  Dataset: {}", config.dataset.path);
        }
        Commands::Train { config, resume } => {
            tracing::info!("Starting training with config: {}", config);
            let config = AxolotlConfig::from_file(&config)?;
            config.validate()?;

            let mut trainer = trainer::Trainer::new(config)?;
            if let Some(checkpoint) = resume {
                trainer.resume_from(&checkpoint)?;
            }
            trainer.train()?;
        }
        Commands::Merge {
            config,
            adapter,
            output,
        } => {
            tracing::info!("Merge requested (output={})", output);
            let config = AxolotlConfig::from_file(&config)?;
            let adapter_path = adapter.unwrap_or_else(|| {
                // Prefer explicit final checkpoint name; fall back to last step dir pattern
                let final_dir = format!("{}/checkpoint-final", config.output_dir);
                if std::path::Path::new(&final_dir).exists() {
                    final_dir
                } else {
                    format!("{}/checkpoint-final", config.output_dir)
                }
            });

            match model::merge_adapter(&config, &adapter_path, &output) {
                Ok(()) => {
                    println!("✓ Merged model saved to: {output}");
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    eprintln!(
                        "hint: ensure base_model points at local full-precision weights and \
adapter_model.safetensors exists (train with --features peft first)."
                    );
                    std::process::exit(2);
                }
            }
        }
        Commands::Download { model_id, output } => {
            tracing::info!("Download requested: {model_id} -> {output}");
            #[cfg(feature = "download")]
            {
                match model::download_model(&model_id, &output) {
                    Ok(path) => {
                        println!("✓ Model available at: {path}");
                        println!("  Set base_model: {path} in your config YAML");
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        eprintln!(
                            "hint: for gated models set HF_TOKEN, or use:\n  \
huggingface-cli download {model_id} --local-dir <path>\nand set base_model to that local path."
                        );
                        std::process::exit(2);
                    }
                }
            }
            #[cfg(not(feature = "download"))]
            {
                eprintln!(
                    "error: download feature not enabled in this build.\n\
hint: rebuild with --features download, or run:\n  \
huggingface-cli download {model_id} --local-dir <path>"
                );
                std::process::exit(2);
            }
        }
        Commands::Init { output, preset } => {
            tracing::info!("Generating config for preset: {}", preset);
            let config = AxolotlConfig::from_preset(&preset)?;
            config.to_file(&output)?;
            println!("✓ Configuration written to: {output}");
        }
    }

    Ok(())
}
