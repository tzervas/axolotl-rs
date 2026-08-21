//! CPU checks for `--features unsloth` (CustomOp CE/RoPE/RMSNorm).
//!
//! RoPE lives in `llama_common`, which is compiled only with `--features peft`.
//! Isolated `--features unsloth` still covers `RmsNormWrapper`. Combined
//! `--features peft,qlora,unsloth` covers RoPE as well.
//!
//! ```bash
//! cargo test --features unsloth --test unsloth_cpu
//! cargo test --features peft,unsloth --test unsloth_cpu
//! ```

#![cfg(feature = "unsloth")]

use axolotl_rs::normalization::RmsNormWrapper;
use candle_core::{Device, Tensor};

#[test]
fn rmsnorm_wrapper_cpu_preserves_shape() {
    let device = Device::Cpu;
    let layer = RmsNormWrapper::new(32, 1e-5, &device).expect("RmsNormWrapper");
    let x = Tensor::randn(0f32, 1f32, (2, 8, 32), &device).unwrap();
    let y = layer.forward(&x).expect("forward");
    assert_eq!(y.dims(), x.dims());
    assert!(y
        .to_vec3::<f32>()
        .unwrap()
        .iter()
        .flatten()
        .flatten()
        .all(|v| v.is_finite()));
}

#[cfg(feature = "peft")]
mod rope {
    use axolotl_rs::llama_common::{apply_rotary_emb, Cache};
    use candle_core::{DType, Device, Tensor};
    use candle_transformers::models::llama::{Config, LlamaEosToks};

    fn tiny_llama_config() -> Config {
        Config {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            use_flash_attn: false,
            bos_token_id: Some(1),
            eos_token_id: Some(LlamaEosToks::Single(2)),
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn apply_rotary_emb_cpu_finite() {
        let device = Device::Cpu;
        let cfg = tiny_llama_config();
        let cache = Cache::new(false, DType::F32, &cfg, &device).expect("cache");
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device).unwrap();
        let y = apply_rotary_emb(&x, 0, &cache).expect("rope");
        assert_eq!(y.dims(), x.dims());
        let flat = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }
}
