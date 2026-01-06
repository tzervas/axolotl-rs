# axolotl-rs

YAML-driven configurable fine-tuning toolkit for LLMs in Rust.

[![Crates.io](https://img.shields.io/crates/v/axolotl-rs.svg)](https://crates.io/crates/axolotl-rs)
[![Documentation](https://docs.rs/axolotl-rs/badge.svg)](https://docs.rs/axolotl-rs)
[![License](https://img.shields.io/crates/l/axolotl-rs.svg)](LICENSE-MIT)

## Overview

`axolotl-rs` provides a user-friendly interface for fine-tuning language models, inspired by the Python [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) project:

- **YAML Configuration** - Define entire training runs in simple config files
- **Multiple Adapters** - Support for LoRA, QLoRA, and full fine-tuning
- **Dataset Handling** - Automatic loading and preprocessing (Alpaca, ShareGPT, custom)
- **CLI Interface** - Simple commands for training, validation, and merging

## Installation

```bash
# From crates.io
cargo install axolotl-rs

# From source
cargo install --path axolotl-rs
```

## Quick Start

### 1. Generate a Configuration

```bash
# Create a config for LLaMA-2 7B with QLoRA
axolotl init config.yaml --preset llama2-7b
```

### 2. Prepare Your Dataset

Create a JSONL file in Alpaca format:

```json
{"instruction": "Explain quantum computing", "input": "", "output": "Quantum computing uses..."}
{"instruction": "Write a haiku about Rust", "input": "", "output": "Memory safe code\n..."}
```

### 3. Validate Configuration

```bash
axolotl validate config.yaml
```

### 4. Start Training

```bash
axolotl train config.yaml
```

### 5. Merge Adapters (Optional)

```bash
axolotl merge --config config.yaml --output ./merged-model
```

## Configuration

### Full Example

```yaml
# config.yaml
base_model: meta-llama/Llama-2-7b-hf
adapter: qlora

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

# Quantization (for QLoRA)
quantization:
  bits: 4
  quant_type: nf4
  double_quant: true

# Dataset
dataset:
  path: ./data/train.jsonl
  format: alpaca
  max_length: 2048
  val_split: 0.05

# Training
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  save_steps: 500
  gradient_checkpointing: true

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

- `llama2-7b` - LLaMA-2 7B with QLoRA
- `mistral-7b` - Mistral 7B with QLoRA  
- `phi3-mini` - Phi-3 Mini with LoRA

## CLI Commands

```bash
# Validate configuration
axolotl validate <config.yaml>

# Start training
axolotl train <config.yaml>
axolotl train <config.yaml> --resume ./checkpoint-1000

# Merge adapter into base model
axolotl merge --config <config.yaml> --output <path>

# Generate sample config
axolotl init <output.yaml> --preset <preset>
```

## Architecture

```
axolotl-rs
├── config     - YAML parsing & validation
├── dataset    - Data loading & preprocessing
├── model      - Model loading & adapter management
└── trainer    - Training loop & optimization

Dependencies:
├── peft-rs    - PEFT adapter implementations
├── qlora-rs   - 4-bit quantization
└── unsloth-rs - Optimized kernels (optional)
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `download` | Enable model downloading from HF Hub |
| `cuda` | Enable CUDA GPU support |
| `optimized` | Use unsloth-rs optimized kernels |

## Contributing

See workspace [AGENTS.md](../AGENTS.md) for coding conventions.

## License

Licensed under MIT or Apache-2.0 at your option.
