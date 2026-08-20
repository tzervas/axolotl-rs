//! Portable export of adapters and merged weights.
//!
//! Train NF4 if wanted; ship dense Hugging Face weights (PEFT adapters or merged).
//! Let llama.cpp quantize. This module never writes a custom `GGUF_TYPE_QLORA_NF4`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle_core::{Device, Tensor};

use crate::config::AxolotlConfig;
use crate::error::{AxolotlError, Result};
use crate::model::{merge_adapter, pair_lora_ab_tensors, write_hub_safe_adapter};

/// Exact convert command printed when llama.cpp tools are missing.
pub const GGUF_CONVERT_CMD: &str =
    "python convert_hf_to_gguf.py ./merged-model --outtype bf16 --outfile model-bf16.gguf";
/// Exact quantize command printed when llama.cpp tools are missing.
pub const GGUF_QUANTIZE_CMD: &str = "llama-quantize model-bf16.gguf model-q4_k_m.gguf Q4_K_M";

/// Output formats for [`run_export`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Hub-safe PEFT adapter directory (`adapter_model.safetensors` + config).
    Peft,
    /// Merge `LoRA` into dense Hugging Face weights.
    Hf,
    /// Ollama `Modelfile` with `FROM <base>` and `ADAPTER <adapter dir>`.
    OllamaAdapter,
    /// Ollama `Modelfile` with `FROM <merged>`; GGUF is preferred for Ollama FROM.
    OllamaMerged,
    /// Convert dense HF → GGUF via llama.cpp (never a custom NF4 GGUF).
    Gguf,
}

impl ExportFormat {
    /// Parse CLI `--format` (case-insensitive).
    ///
    /// # Errors
    /// Returns [`AxolotlError::Export`] for unknown names.
    pub fn from_cli(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "peft" => Ok(Self::Peft),
            "hf" => Ok(Self::Hf),
            "ollama-adapter" => Ok(Self::OllamaAdapter),
            "ollama-merged" => Ok(Self::OllamaMerged),
            "gguf" => Ok(Self::Gguf),
            other => Err(AxolotlError::Export(format!(
                "unknown export format '{other}'; expected peft, hf, ollama-adapter, ollama-merged, or gguf"
            ))),
        }
    }
}

/// Arguments for [`run_export`].
#[derive(Debug, Clone)]
pub struct ExportRequest {
    /// Target format.
    pub format: ExportFormat,
    /// Training / merge config (`base_model`, `LoRA` r/α, `output_dir`).
    pub config: AxolotlConfig,
    /// Destination directory.
    pub output: PathBuf,
    /// Adapter checkpoint directory or `adapter_model.safetensors`.
    pub adapter: Option<String>,
    /// Pre-merged dense HF directory (`gguf` / `ollama-merged`).
    pub merged: Option<String>,
    /// llama.cpp quant type (`gguf`), e.g. `Q4_K_M`.
    pub quantize: Option<String>,
}

/// Run a portable export.
///
/// # Errors
/// Returns [`AxolotlError::Export`] (or merge/IO errors) when the export cannot
/// complete. Missing llama.cpp tools for `--format gguf` is an export error.
pub fn run_export(req: &ExportRequest) -> Result<()> {
    std::fs::create_dir_all(&req.output)?;
    match req.format {
        ExportFormat::Peft => export_peft(req),
        ExportFormat::Hf => export_hf(req),
        ExportFormat::OllamaAdapter => export_ollama_adapter(req),
        ExportFormat::OllamaMerged => export_ollama_merged(req),
        ExportFormat::Gguf => export_gguf(req),
    }
}

fn default_adapter_path(config: &AxolotlConfig) -> String {
    format!("{}/checkpoint-final", config.output_dir)
}

fn resolve_adapter(req: &ExportRequest) -> String {
    req.adapter
        .clone()
        .unwrap_or_else(|| default_adapter_path(&req.config))
}

fn export_peft(req: &ExportRequest) -> Result<()> {
    let adapter = resolve_adapter(req);
    let adapter_p = PathBuf::from(&adapter);
    let adapter_file = if adapter_p.is_dir() {
        adapter_p.join("adapter_model.safetensors")
    } else {
        adapter_p.clone()
    };
    if !adapter_file.exists() {
        return Err(AxolotlError::Export(format!(
            "adapter weights not found at {}. Expected adapter_model.safetensors.",
            adapter_file.display()
        )));
    }
    let tensors = candle_core::safetensors::load(&adapter_file, &Device::Cpu).map_err(|e| {
        AxolotlError::Export(format!(
            "Failed to load adapter {}: {e}",
            adapter_file.display()
        ))
    })?;
    let paired = pair_lora_ab_tensors(&tensors);
    if paired.is_empty() {
        return Err(AxolotlError::Export(
            "adapter file has no LoRA A/B tensors to rewrite as Hub-safe PEFT keys".into(),
        ));
    }
    let mut modules: HashMap<String, (Tensor, Tensor)> = HashMap::new();
    for (module, (a_opt, b_opt)) in paired {
        match (a_opt, b_opt) {
            (Some(a), Some(b)) => {
                modules.insert(module, (a, b));
            }
            _ => {
                return Err(AxolotlError::Export(format!(
                    "module '{module}' missing paired LoRA A/B"
                )));
            }
        }
    }
    write_hub_safe_adapter(&modules, &req.output, Some(&req.config))?;
    tracing::info!("Wrote Hub-safe PEFT adapter to {}", req.output.display());
    Ok(())
}

fn export_hf(req: &ExportRequest) -> Result<()> {
    let adapter = resolve_adapter(req);
    let out = req.output.to_string_lossy().into_owned();
    merge_adapter(&req.config, &adapter, &out)?;
    Ok(())
}

fn write_modelfile(dir: &Path, body: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("Modelfile"), body)?;
    Ok(())
}

fn export_ollama_adapter(req: &ExportRequest) -> Result<()> {
    let adapter = resolve_adapter(req);
    let body = format!("FROM {}\nADAPTER {}\n", req.config.base_model, adapter);
    write_modelfile(&req.output, &body)?;
    tracing::info!(
        "Wrote Ollama adapter Modelfile to {}",
        req.output.join("Modelfile").display()
    );
    Ok(())
}

fn export_ollama_merged(req: &ExportRequest) -> Result<()> {
    let merged = if let Some(m) = &req.merged {
        PathBuf::from(m)
    } else {
        let dest = req.output.join("merged-model");
        let adapter = resolve_adapter(req);
        merge_adapter(
            &req.config,
            &adapter,
            dest.to_str().ok_or_else(|| {
                AxolotlError::Export("merged output path is not valid UTF-8".into())
            })?,
        )?;
        dest
    };
    let body = format!(
        "# GGUF is preferred for Ollama FROM (convert merged HF with llama.cpp).\n\
FROM {}\n",
        merged.display()
    );
    write_modelfile(&req.output, &body)?;
    tracing::info!(
        "Wrote Ollama merged Modelfile to {}",
        req.output.join("Modelfile").display()
    );
    Ok(())
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn missing_gguf_tools_error() -> AxolotlError {
    AxolotlError::Export(format!(
        "convert_hf_to_gguf.py and/or llama-quantize not found on PATH. \
axolotl export --format gguf does not write GGUF itself (never a custom NF4 GGUF). \
Install llama.cpp tools and run:\n{GGUF_CONVERT_CMD}\n{GGUF_QUANTIZE_CMD}"
    ))
}

fn export_gguf(req: &ExportRequest) -> Result<()> {
    let (Some(convert), Some(quant)) = (
        find_on_path("convert_hf_to_gguf.py"),
        find_on_path("llama-quantize"),
    ) else {
        return Err(missing_gguf_tools_error());
    };

    let qtype = req
        .quantize
        .as_deref()
        .unwrap_or("Q4_K_M")
        .trim()
        .to_string();
    let q_upper = qtype.to_ascii_uppercase();
    if q_upper.contains("QLORA") || q_upper.contains("NF4") || q_upper.contains("GGUF_TYPE") {
        return Err(AxolotlError::Export(
            "refusing to write custom NF4 GGUF (never GGUF_TYPE_QLORA_NF4). \
Train NF4 if wanted; ship dense HF and let llama.cpp quantize (e.g. Q4_K_M)."
                .into(),
        ));
    }

    let merged = if let Some(m) = &req.merged {
        PathBuf::from(m)
    } else {
        let dest = req.output.join("merged-model");
        let adapter = resolve_adapter(req);
        merge_adapter(
            &req.config,
            &adapter,
            dest.to_str().ok_or_else(|| {
                AxolotlError::Export("merged output path is not valid UTF-8".into())
            })?,
        )?;
        dest
    };

    let bf16_out = req.output.join("model-bf16.gguf");
    let py = if find_on_path("python3").is_some() {
        "python3"
    } else {
        "python"
    };
    let status = Command::new(py)
        .arg(&convert)
        .arg(&merged)
        .args(["--outtype", "bf16", "--outfile"])
        .arg(&bf16_out)
        .status()
        .map_err(|e| {
            AxolotlError::Export(format!("failed to spawn {py} convert_hf_to_gguf.py: {e}"))
        })?;
    if !status.success() {
        return Err(AxolotlError::Export(format!(
            "convert_hf_to_gguf.py failed with {status}"
        )));
    }

    let skip_quant = q_upper == "BF16" || q_upper == "F16" || q_upper == "NONE";
    if skip_quant {
        tracing::info!("Wrote {}", bf16_out.display());
        return Ok(());
    }

    let q_out = req
        .output
        .join(format!("model-{}.gguf", qtype.to_ascii_lowercase()));
    let status = Command::new(&quant)
        .arg(&bf16_out)
        .arg(&q_out)
        .arg(&qtype)
        .status()
        .map_err(|e| AxolotlError::Export(format!("failed to spawn llama-quantize: {e}")))?;
    if !status.success() {
        return Err(AxolotlError::Export(format!(
            "llama-quantize failed with {status}"
        )));
    }
    tracing::info!("Wrote {}", q_out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_from_cli() {
        assert_eq!(ExportFormat::from_cli("peft").unwrap(), ExportFormat::Peft);
        assert_eq!(ExportFormat::from_cli("HF").unwrap(), ExportFormat::Hf);
        assert_eq!(
            ExportFormat::from_cli("ollama-adapter").unwrap(),
            ExportFormat::OllamaAdapter
        );
        assert_eq!(ExportFormat::from_cli("gguf").unwrap(), ExportFormat::Gguf);
        assert!(ExportFormat::from_cli("onnx").is_err());
    }
}
