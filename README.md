# axolotl-rs

<!-- FLEET-BADGES:BEGIN -->
[![CI](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-ci.yml/badge.svg?branch=main)](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-ci.yml?query=branch%3Amain)
[![Security](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-security.yml/badge.svg?branch=main)](https://github.com/tzervas/axolotl-rs/actions/workflows/fleet-security.yml?query=branch%3Amain)
<!-- FLEET-BADGES:END -->

YAML-driven fine-tuning **orchestrator** for LLaMA-family LLMs in Rust (inspired by Python [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)).

[![Crates.io](https://img.shields.io/crates/v/axolotl-rs.svg)](https://crates.io/crates/axolotl-rs)
[![Documentation](https://docs.rs/axolotl-rs/badge.svg)](https://docs.rs/axolotl-rs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

> **Status honesty:** Version **1.4.0** is a working LLaMA-family LoRA trainer/orchestrator on local
> weights — **not** full Python Axolotl parity. Candle **0.11**, MSRV **1.96**.
> Optional `vsa-optim` is **not** a claimed/tested acceleration path in this release.
> **BitNet QAT is not wired until candle 0.9/0.11 is aligned. Do not AbsMean PTQ.
> Sister runtime already generates official 2B4T.** See the capability matrix.
>
> **Docs:** [CHANGELOG.md](CHANGELOG.md) · [roadmap.md](roadmap.md) · [CUDA_STATUS.md](CUDA_STATUS.md) ·
> [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) (leaf crate; no reverse deps / no cycles) ·
> [docs/VERSIONING.md](docs/VERSIONING.md) ·
> [docs/UNSLOTH_KERNEL_WIRING.md](docs/UNSLOTH_KERNEL_WIRING.md) ·
> [docs/archive/](docs/archive/) (historical COMPLETE fiction — do not treat as current status)

## Capability matrix

Live version is [`Cargo.toml`](Cargo.toml) / `cz version --project` ([docs/VERSIONING.md](docs/VERSIONING.md)).

Hosted CI `cargo test` is **default features**. Adapter/unsloth paths are gated in the
`adapter-features` job (`peft`, `unsloth` isolate-check, then `peft,qlora,unsloth` tests).
Self-hosted `fleet-ci` does not rustc `candle-transformers` (OOM on that runner).
Kitchen-sink check/test is GitHub-hosted (`fleet-ci` `ubuntu-latest` + this `ci.yml`).

| Capability | Default features | `--features peft` | `--features peft,qlora` | `--features unsloth` | Notes |
|------------|------------------|-------------------|-------------------------|----------------------|-------|
| YAML parse / validate / presets | ✅ | ✅ | ✅ | ✅ | Real |
| Dataset loaders (Alpaca, ShareGPT, completion, custom) | ✅ | ✅ | ✅ | ✅ | Local JSONL only |
| CLI `validate` / `init` / `train` | ✅ | ✅ | ✅ | ✅ | `train` needs local model files |
| CLI `merge` | ✅ | ✅ | ✅ | ✅ | Fuses LoRA A/B into dense `W` (`W + scale·B@A`); copies HF sidecars; rejects U8/NF4 packed tensors |
| CLI `export` | ✅ | ✅ | ✅ | ✅ | `--format peft\|hf\|ollama-adapter\|ollama-merged\|gguf` — never writes custom NF4 GGUF |
| CLI `download` | ✅ local resolve | ✅ | ✅ | ✅ | Local path first-class; Hub pull via `reqwest` when `download` feature on |
| LoRA training path (`LoraLlama`) | ❌ not linked | ✅ | ✅ | ❌ not linked | Needs `--features peft` + local weights. Combine with `unsloth` for CustomOp kernels. |
| QLoRA training path (`QLoraLlama`) | ❌ | ❌ | ✅ | ❌ | Needs `--features peft,qlora`; NF4 is a **training** codec |
| Checkpoint save/load LoRA A/B | ❌ | ✅ | ✅ | ❌ | Hub-safe `lora_A.default.weight` / `lora_B.default.weight` in one `adapter_model.safetensors` |
| Sharded safetensors | ✅ | ✅ | ✅ | ✅ | Loads index+shards or hard-errors if shard missing |
| Architecture gate | ✅ | ✅ | ✅ | ✅ | Non-LLaMA → clear `Unsupported` (no 10×10 stub) |
| Grad accumulation / LR schedule / warmup / grad clip | ✅ | ✅ | ✅ | ✅ | From YAML |
| Real grad/param norms | ✅ | ✅ | ✅ | ✅ | Not placeholder constants |
| Unsloth RoPE / chunked CE | ❌ | ❌ | ❌ | ✅ | CustomOp on `LoraAttention` + trainer CE. `RmsNormWrapper` exists but **is not** on the LoRA train graph (`candle_nn::RmsNorm` + `forward_diff`). See [docs/UNSLOTH_KERNEL_WIRING.md](docs/UNSLOTH_KERNEL_WIRING.md). Combine with `peft` / `qlora`. |
| Multi-GPU / packing / DPO | ❌ | ❌ | ❌ | ❌ | Out of scope |
| GPU E2E | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Often blocked by Candle CUDA RMSNorm — see [CUDA_STATUS.md](CUDA_STATUS.md) |
| BitNet b1.58 QAT (`adapter: bitnet`) | ❌ not wired | ❌ | ❌ | ❌ | Recipe only. axolotl-rs is candle **0.11**; `bitnet-quantize` v0.5.1 is candle **0.9**. **No cargo dep.** AbsMean PTQ is forbidden. |

### BitNet QAT (not wired)

BitNet QAT is not wired until candle 0.9/0.11 is aligned. Do not AbsMean PTQ.
Sister runtime already generates official 2B4T.

[`examples/configs/bitnet_qat_2b4t.yaml`](examples/configs/bitnet_qat_2b4t.yaml) documents the
**intelligent** conversion path (rewire SubLN + gated ReLU², STE QAT, KL distill, packed
TwoBit export, RTX 5080 VRAM cap, no NF4, no 9GB BF16 on device). `adapter: bitnet` is a
future key: the parser **rejects** it with a clear error. Do not add `bitnet-quantize` as a
Cargo dependency while the candles diverge. Official packed 2B4T generate lives in
[`ternary-inference-rs`](https://github.com/tzervas/ternary-inference-rs). Conversion is
[`bitnet-quantize`](https://github.com/tzervas/bitnet-quantize) Track B — not this crate,
not this recipe.

### Sister crates & dependency policy

| Build context | How peft / qlora / unsloth resolve |
|---------------|-------------------------------------|
| **GitHub CI / crates.io** | Registry versions only (`peft-rs = "1.2"`, `qlora-rs = "1.2"`, `unsloth-rs = "1.2"`) — **no path deps** |
| **Local fleet / SoT development** | Run `bash scripts/use-local-path-deps.sh` to write gitignored `.cargo/config.toml` `paths = ["../peft-rs", …]` so local sister trees override the registry |

`safetensors` is pinned to **0.8** (matches candle 0.11 / peft-rs 1.2). See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## Installation

```bash
# From crates.io (default features)
cargo install axolotl-rs

# From source with adapters (registry peft/qlora until 1.1 floors land)
git clone https://github.com/tzervas/axolotl-rs
cd axolotl-rs
cargo build --release --features peft,qlora

# Local monorepo: prefer sister SoTs at ../peft-rs and ../qlora-rs
bash scripts/use-local-path-deps.sh
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

# Portable export (PEFT adapter, dense HF, Ollama Modelfile, or llama.cpp GGUF)
axolotl export --format peft --config <config.yaml> --adapter <checkpoint_dir> --output ./peft-adapter
axolotl export --format hf --config <config.yaml> --adapter <checkpoint_dir> --output ./merged-model
axolotl export --format ollama-adapter --config <config.yaml> --adapter ./peft-adapter --output ./ollama-adapter
axolotl export --format ollama-merged --config <config.yaml> --merged ./merged-model --output ./ollama-merged
axolotl export --format gguf --config <config.yaml> --merged ./merged-model --output ./gguf --quantize Q4_K_M
axolotl export --format gguf --config <config.yaml> --merged ./merged-model --output ./gguf --quantize F16
```

## Deploying

Train NF4/QLoRA if you want; **ship dense HuggingFace weights** (PEFT adapters or a merged dense model). Let llama.cpp quantize. `axolotl export --format gguf` never writes a custom `GGUF_TYPE_QLORA_NF4`.

### vLLM (PEFT adapter or merged dense)

```bash
axolotl export --format peft --config config.yaml --adapter ./outputs/checkpoint-final --output ./peft-adapter

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/base-model \
  --enable-lora \
  --lora-modules mylora=./peft-adapter
```

Or merge first and serve a single dense checkpoint:

```bash
axolotl export --format hf --config config.yaml --adapter ./outputs/checkpoint-final --output ./merged-model
vllm serve ./merged-model
```

### Ollama

Adapter (base + PEFT dir):

```bash
axolotl export --format ollama-adapter --config config.yaml --adapter ./peft-adapter --output ./ollama-adapter
# Modelfile: ADAPTER ./adapter (next to the Modelfile). FROM is an Ollama
# library name, local GGUF, or local HF dir — not a Hub id like org/name.
ollama create mylora -f ./ollama-adapter/Modelfile
```

Merged HF (GGUF is preferred for Ollama `FROM`):

```bash
axolotl export --format ollama-merged --config config.yaml --merged ./merged-model --output ./ollama-merged
# Prefer: convert merged HF → GGUF (below) and `FROM ./model-q4_k_m.gguf`
ollama create mymerged -f ./ollama-merged/Modelfile
```

### llama.cpp GGUF

```bash
axolotl export --format gguf --config config.yaml --merged ./merged-model --output ./gguf --quantize Q4_K_M
```

If `convert_hf_to_gguf.py` / `llama-quantize` are not on `PATH`, the command prints:

```bash
python convert_hf_to_gguf.py ./merged-model --outtype bf16 --outfile model-bf16.gguf
llama-quantize model-bf16.gguf model-q4_k_m.gguf Q4_K_M
```

## Architecture

```
axolotl-rs
├── config     - YAML parsing & validation
├── dataset    - Data loading & preprocessing
├── model      - Model loading, merge, download, sharded weights
├── export     - PEFT / dense HF / Ollama / llama.cpp GGUF
├── fixture    - Tiny LLaMA on-disk fixtures for CPU E2E
├── lora_llama - Per-layer LoRA injection (feature peft)
├── qlora_llama- QLoRA path (features peft,qlora)
└── trainer    - Training loop, checkpoint A/B I/O

Dependencies:
├── candle-*   - Tensor operations and transformer models
├── tokenizers - HuggingFace tokenizer bindings
├── peft-rs    - LoRA adapters (optional feature `peft`; crates.io or local paths)
├── qlora-rs   - 4-bit quantization (optional feature `qlora`; implies peft)
└── unsloth-rs - Kernel building blocks (optional; not required for core path)
```

## Feature Flags

| Flag | Description | Reality check |
|------|-------------|---------------|
| `download` (default) | Enables `reqwest` (+ blocking) | Hub download **implemented**; local paths still preferred |
| `peft` | peft-rs LoRA path | Registry (or local override via `use-local-path-deps.sh`) |
| `qlora` | qlora-rs + peft | Implies `peft` |
| `unsloth` | unsloth-rs kernels | Optional |
| `cuda` | Candle CUDA | GPU training may still hit RMSNorm gaps — see [CUDA_STATUS.md](CUDA_STATUS.md) |

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
