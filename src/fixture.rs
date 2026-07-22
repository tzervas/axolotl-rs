//! Tiny LLaMA-shaped fixtures for CPU E2E tests and local demos.
//!
//! Creates a minimal on-disk model directory:
//! - `config.json` (LlamaForCausalLM)
//! - `model.safetensors` (random base weights)
//! - `tokenizer.json` (WordLevel, small vocab)

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use crate::error::{AxolotlError, Result};

/// Dimensions for the default tiny LLaMA fixture (fits comfortably on CPU).
#[derive(Debug, Clone, Copy)]
pub struct TinyLlamaSpec {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden / embedding dimension.
    pub hidden_size: usize,
    /// MLP intermediate size.
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Attention heads.
    pub num_attention_heads: usize,
    /// Key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Max sequence length for RoPE tables.
    pub max_position_embeddings: usize,
}

impl Default for TinyLlamaSpec {
    fn default() -> Self {
        Self {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 128,
        }
    }
}

/// Write a complete tiny LLaMA-shaped model directory under `dir`.
///
/// # Errors
/// Returns an error if files cannot be written or tensors cannot be created.
pub fn write_tiny_llama_fixture<P: AsRef<Path>>(dir: P, spec: TinyLlamaSpec) -> Result<PathBuf> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    write_config_json(dir, &spec)?;
    write_tokenizer_json(dir, spec.vocab_size)?;
    write_model_safetensors(dir, &spec)?;

    Ok(dir.to_path_buf())
}

fn write_config_json(dir: &Path, spec: &TinyLlamaSpec) -> Result<()> {
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": spec.vocab_size,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_hidden_layers": spec.num_hidden_layers,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": spec.max_position_embeddings,
        "tie_word_embeddings": false,
    });
    let path = dir.join("config.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&config)
            .map_err(|e| AxolotlError::Model(format!("fixture config serialize: {e}")))?,
    )?;
    Ok(())
}

fn write_tokenizer_json(dir: &Path, vocab_size: usize) -> Result<()> {
    // WordLevel tokenizer with a deterministic mini-vocab.
    let mut vocab = HashMap::new();
    let specials = ["<unk>", "<s>", "</s>", "<pad>"];
    for (i, tok) in specials.iter().enumerate() {
        vocab.insert((*tok).to_string(), i);
    }
    for i in specials.len()..vocab_size {
        vocab.insert(format!("tok{i}"), i);
    }

    let mut added_tokens = Vec::new();
    for (i, tok) in specials.iter().enumerate() {
        added_tokens.push(serde_json::json!({
            "id": i,
            "content": tok,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }));
    }

    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        }
    });

    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tokenizer)
            .map_err(|e| AxolotlError::Model(format!("fixture tokenizer serialize: {e}")))?,
    )?;
    Ok(())
}

fn write_model_safetensors(dir: &Path, spec: &TinyLlamaSpec) -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let h = spec.hidden_size;
    let inter = spec.intermediate_size;
    let head_dim = h / spec.num_attention_heads;
    let size_q = head_dim * spec.num_attention_heads;
    let size_kv = head_dim * spec.num_key_value_heads;

    // Small random scale keeps activations reasonable.
    let scale = 0.02f32;

    let embed = Tensor::randn(0f32, scale, (spec.vocab_size, h), &device)
        .map_err(|e| AxolotlError::Model(format!("fixture embed: {e}")))?
        .to_dtype(dtype)
        .map_err(|e| AxolotlError::Model(format!("fixture embed dtype: {e}")))?;
    tensors.insert("model.embed_tokens.weight".into(), embed);

    let ones = |n: usize| -> Result<Tensor> {
        Tensor::ones(n, dtype, &device)
            .map_err(|e| AxolotlError::Model(format!("fixture ones: {e}")))
    };
    let rand_mat = |rows: usize, cols: usize| -> Result<Tensor> {
        Tensor::randn(0f32, scale, (rows, cols), &device)
            .map_err(|e| AxolotlError::Model(format!("fixture rand: {e}")))?
            .to_dtype(dtype)
            .map_err(|e| AxolotlError::Model(format!("fixture rand dtype: {e}")))
    };

    for layer in 0..spec.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        tensors.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            rand_mat(size_q, h)?,
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            rand_mat(size_kv, h)?,
        );
        tensors.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            rand_mat(size_kv, h)?,
        );
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            rand_mat(h, size_q)?,
        );
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            rand_mat(inter, h)?,
        );
        tensors.insert(format!("{prefix}.mlp.up_proj.weight"), rand_mat(inter, h)?);
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            rand_mat(h, inter)?,
        );
        tensors.insert(format!("{prefix}.input_layernorm.weight"), ones(h)?);
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            ones(h)?,
        );
    }

    tensors.insert("model.norm.weight".into(), ones(h)?);
    tensors.insert("lm_head.weight".into(), rand_mat(spec.vocab_size, h)?);

    let path = dir.join("model.safetensors");
    candle_core::safetensors::save(&tensors, &path)
        .map_err(|e| AxolotlError::Model(format!("fixture safetensors save: {e}")))?;
    Ok(())
}

/// Write a tiny Alpaca JSONL dataset for training smoke tests.
///
/// # Errors
/// Returns an error if the file cannot be written.
pub fn write_tiny_alpaca_jsonl<P: AsRef<Path>>(path: P, n: usize) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut lines = Vec::with_capacity(n);
    // Tokens must exist in the WordLevel vocab (tok4..).
    for i in 0..n {
        let instruction = format!("tok4 tok5 tok6 {i}");
        let output = format!("tok7 tok8 tok9 {i}");
        let row = serde_json::json!({
            "instruction": instruction,
            "input": "",
            "output": output
        });
        lines.push(row.to_string());
    }
    fs::write(path, lines.join("\n") + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_tiny_llama_fixture_files_exist() {
        let tmp = TempDir::new().unwrap();
        let dir = write_tiny_llama_fixture(tmp.path(), TinyLlamaSpec::default()).unwrap();
        assert!(dir.join("config.json").exists());
        assert!(dir.join("tokenizer.json").exists());
        assert!(dir.join("model.safetensors").exists());

        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        assert_eq!(tok.get_vocab_size(true), 64);
    }
}
