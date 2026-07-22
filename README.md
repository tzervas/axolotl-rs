# axolotl-rs

<!-- FLEET-BADGES:BEGIN -->
[![CI](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-ci.yml/badge.svg?branch=main)](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-ci.yml?query=branch%3Amain)
[![Security](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-security.yml/badge.svg?branch=main)](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-security.yml?query=branch%3Amain)
<!-- FLEET-BADGES:END -->

YAML-driven fine-tuning **orchestrator** for LLaMA-family LLMs in Rust (inspired by Python [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)).

[![Crates.io](https://img.shields.io/crates/v/axolotl-rs.svg)](https://crates.io/crates/axolotl-rs)
[![Documentation](https://docs.rs/axolotl-rs/badge.svg)](https://docs.rs/axolotl-rs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

> **Status honesty:** Version **1.2.0** is a working LLaMA-family LoRA trainer/orchestrator on local
> weights — **not** full Python Axolotl parity. See the capability matrix.

## Capability matrix (1.2.0)

| Capability | Default features | `--features peft` | `--features peft,qlora` | Notes |
|------------|------------------|-------------------|-------------------------|-------|
| YAML parse / validate / presets | ✅ | ✅ | ✅ | Real |
| Dataset loaders (Alpaca, ShareGPT, completion, custom) | ✅ | ✅ | ✅ | Local JSONL only |
| CLI `validate` / `init` / `train` | ✅ | ✅ | ✅ | `train` needs local model files |
| CLI `merge` | ✅ | ✅ | ✅ | Fuses LoRA A/B into base `W` (`W + scale·B@A`) |
| CLI `download` | ✅ local resolve | ✅ | ✅ | Local path first-class; Hub pull via `reqwest` when `download` feature on |
| LoRA training path (`LoraLlama`) | ❌ not linked | ✅ | ✅ | Needs peft-rs + local weights |
| QLoRA training path (`QLoraLlama`) | ❌ | ❌ | ✅ | Needs peft+qlora |
| Checkpoint save/load LoRA A/B | ❌ | ✅ | ✅ | `adapter_model.safetensors` round-trip |
| Sharded safetensors | ✅ | ✅ | ✅ | Loads index+shards or hard-errors if shard missing |
| Architecture gate | ✅ | ✅ | ✅ | Non-LLaMA → clear `Unsupported` (no 10×10 stub) |
| Grad accumulation / LR schedule / warmup / grad clip | ✅ | ✅ | ✅ | From YAML |
| Real grad/param norms | ✅ | ✅ | ✅ | Not placeholder constants |
| Multi-GPU / packing / DPO | ❌ | ❌ | ❌ | Out of scope |
| GPU E2E | ⚠️ | ⚠️ | ⚠️ | Often blocked by Candle CUDA RMSNorm — see [CUDA_STATUS.md](CUDA_STATUS.md) |

**Sister crates (this monorepo SoT):** path dependencies on `../peft-rs` (1.1) and `../qlora-rs` (1.1)
plus `[patch.crates-io] peft-rs`. Align `safetensors` to **0.7**.

## Installation

```bash
# From crates.io (default features)
cargo install axolotl-rs

# From this monorepo with adapters
git clone https://github.com/tzervas/axolotl-rs
cd axolotl-rs
cargo build --release --features peft,qlora
```

## Quick Start

### 1. Generate a Configuration

```bash
axolotl init config.yaml --preset llama2-7b
```

### 2. Prepare Your Dataset

Create a JSONL file in Alpaca format:

```json
{"instruction": "Explain quantum computing", "input": "", "output": "Quantum computing uses..."}
{"instruction": "Write a haiku about Rust", "input": "", "output": "Memory safe code\n..."}
```

### 3. Get base model weights (local path first-class)

```bash
# Preferred: pre-download with Hugging Face CLI
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/tinyllama

# Or use axolotl download (Hub pull; set HF_TOKEN for gated models)
axolotl download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output ./models
```

Point `base_model` in YAML at the **local directory** containing `config.json`,
`tokenizer.json`, and `model.safetensors` (or a sharded index + shards).

### 4. Validate & train

```bash
axolotl validate config.yaml

# LoRA/QLoRA require feature flags at build time
cargo run --features peft -- train config.yaml
```

### 5. Merge adapters

After training, fuse LoRA into base weights for inference:

```bash
axolotl merge --config config.yaml --adapter ./outputs/checkpoint-100 --output ./merged-model
```

Writes `model.safetensors`, copies tokenizer/config, and `merge_info.json`.

## Configuration

### Full Example

```yaml
# config.yaml
base_model: /path/to/local/Llama-2-7b-hf   # local path preferred
adapter: lora

# LoRA settings
lora:
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

# Quantization (for QLoRA; needs --features peft,qlora)
# quantization:
#   bits: 4
#   quant_type: nf4
#   double_quant: true

# Dataset
dataset:
  path: ./data/train.jsonl
  format: alpaca
  max_length: 2048
  val_split: 0.05

# Training (these knobs are honored by the trainer)
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  save_steps: 500
  # gradient_checkpointing / mixed_precision: parsed but not implemented (warned)

output_dir: ./outputs/my-model
seed: 42
```

### Dataset Formats

| Format | Description | Fields |
|--------|-------------|--------|
| `alpaca` | Standard Alpaca | `instruction`, `input`, `output` |
| `sharegpt` | Conversation format | `conversations[{from, value}]` |
| `completion` | Raw text | `text` |
| `custom` | User-defined | Configure `input_field`, `output_field` |

### Available Presets

- `llama2-7b` - LLaMA-2 7B with QLoRA settings
- `mistral-7b` - Mistral 7B with QLoRA settings *(config preset only — runtime refuses non-LLaMA)*
- `phi3-mini` - Phi-3 Mini with LoRA settings *(config preset only — runtime refuses non-LLaMA)*

## CLI Commands

```bash
# Validate configuration
axolotl validate <config.yaml>

# Start training (requires local model files; use --features peft for LoRA)
axolotl train <config.yaml>
axolotl train <config.yaml> --resume ./checkpoint-1000

# Merge LoRA adapter into base model weights
axolotl merge --config <config.yaml> --adapter <checkpoint_dir> --output <path>

# Resolve local path or download from Hub into --output
axolotl download <model_id_or_path> --output ./models

# Generate sample config
axolotl init <output.yaml> --preset <preset>
```

## Architecture

```
axolotl-rs
├── config     - YAML parsing & validation
├── dataset    - Data loading & preprocessing
├── model      - Model loading, merge, download, sharded weights
├── fixture    - Tiny LLaMA on-disk fixtures for CPU E2E
├── lora_llama - Per-layer LoRA injection (feature peft)
├── qlora_llama- QLoRA path (features peft,qlora)
└── trainer    - Training loop, checkpoint A/B I/O

Dependencies:
├── candle-*   - Tensor operations and transformer models
├── tokenizers - HuggingFace tokenizer bindings
├── peft-rs    - LoRA adapters (optional, path/crates.io 1.1)
├── qlora-rs   - 4-bit quantization (optional, 1.1)
└── unsloth-rs - Optimized kernels (optional; not required for core path)
```

## Feature Flags

| Flag | Description | Reality check |
|------|-------------|---------------|
| `download` (default) | Enables `reqwest` (+ blocking) | Hub download **implemented**; local paths still preferred |
| `peft` | peft-rs LoRA path | Path dep to `../peft-rs` in this tree |
| `qlora` | qlora-rs + peft | Implies `peft` |
| `unsloth` | unsloth-rs kernels | Optional |
| `cuda` | Candle CUDA | GPU training may still hit RMSNorm gaps |

## CPU E2E proof

```bash
# Unit + lib tests (default features)
cargo test --lib

# LoRA train + checkpoint + sharded load + arch refuse
cargo test --features peft --test e2e_lora_cpu

# Compile QLoRA path
cargo check --features peft,qlora
```

## License

MIT — see [LICENSE-MIT](LICENSE-MIT).
