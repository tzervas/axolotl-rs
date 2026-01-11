# axolotl-rs

> ⚠️ **Early Development**: This project is a framework scaffold under active development. Core training functionality is not yet implemented. The configuration system, CLI interface, and dataset loaders are functional, but actual model training, adapter management, and checkpoint handling are planned for future releases.

YAML-driven configurable fine-tuning toolkit for LLMs in Rust.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

## Overview

`axolotl-rs` is a Rust port of the Python [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) project, providing a framework for fine-tuning language models.

**Currently Implemented:**
- ✅ **YAML Configuration** - Parse and validate training configuration files
- ✅ **Dataset Handling** - Load datasets in Alpaca, ShareGPT, completion, and custom formats
- ✅ **CLI Interface** - Commands for `validate`, `train`, `merge`, `init` (validation works)
- ✅ **Configuration Presets** - Templates for LLaMA-2, Mistral, and Phi-3 models

**Planned (Not Yet Implemented):**
- 🚧 Model loading and adapter management (LoRA, QLoRA)
- 🚧 Actual training loop with forward/backward passes
- 🚧 Checkpoint saving and loading
- 🚧 Adapter merging
- 🚧 Multi-GPU distributed training

## Installation

```bash
# From source (not yet published to crates.io)
git clone https://github.com/tzervas/axolotl-rs
cd axolotl-rs
cargo build --release
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
# Note: Training loop is not yet implemented
# Currently validates config and creates output directories
axolotl train config.yaml
```

### 5. Merge Adapters (Optional)

```bash
# Note: Adapter merging is not yet implemented
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
├── config     - YAML parsing & validation (✅ implemented)
├── dataset    - Data loading & preprocessing (✅ implemented)
├── model      - Model loading & adapter management (🚧 stub/planned)
└── trainer    - Training loop & optimization (🚧 stub/planned)

Dependencies:
├── candle-*   - Tensor operations and transformer models
├── tokenizers - HuggingFace tokenizer bindings
└── Mock impls - Temporary mocks for peft-rs, qlora-rs, unsloth-rs
```

## Feature Flags

| Flag | Description | Status |
|------|-------------|--------|
| `download` | Enable model downloading from HF Hub | Planned |
| `mock-peft` | Use mock PEFT implementation | Active |
| `mock-qlora` | Use mock QLoRA implementation | Active |
| `mock-unsloth` | Use mock Unsloth implementation | Active |

## Development Status

See [TEST_COVERAGE_PLAN.md](TEST_COVERAGE_PLAN.md) for detailed development roadmap and test coverage goals (target: 80% coverage).

**Porting from Python:** This is a Rust port of the Python [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) project, designed for better performance and efficiency. See [PORTING_PLAN.md](PORTING_PLAN.md) for the complete porting roadmap and [ARCHITECTURE.md](ARCHITECTURE.md) for technical architecture details.

## Contributing

Contributions welcome! This is an early-stage project. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the [MIT License](LICENSE-MIT).
