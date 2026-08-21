# axolotl-rs - YAML-Driven Fine-Tuning Toolkit

## Overview

High-level fine-tuning orchestration layer. Rust port of Python Axolotl, providing YAML-driven configuration for LLM training.

**Status**: 1.4.0 - LLaMA-family LoRA/QLoRA trainer & orchestrator with dense HF / PEFT / Ollama / GGUF export. Candle 0.11, MSRV 1.96.

## Architecture

```
src/
├── lib.rs           # Public API exports
├── main.rs          # CLI entry point
├── cli.rs           # Clap CLI definitions
├── config.rs        # YAML configuration parsing
├── dataset.rs       # Dataset loaders (Alpaca, ShareGPT, etc.)
├── export.rs        # Dense HF / PEFT / Ollama / GGUF export
├── model.rs         # Model loading and architecture
├── lora_llama.rs    # LLaMA with LoRA integration
├── llama_common.rs  # Shared LLaMA utilities
├── qlora_llama.rs   # QLoRA integration
├── error.rs         # Error types
├── normalization.rs # Normalization utilities
├── optimizer.rs     # Optimizer definitions (AdamW)
├── scheduler.rs     # Cosine, Linear, Constant LR schedulers
├── trainer.rs       # Train loop, metrics, and checkpointing
├── vsa_accel.rs     # VSA-accelerated training integration
├── adapters/        # Adapter integration layer
│   └── mod.rs       # Feature-gated adapter loading
└── mocks/           # Mock implementations for testing
    └── mod.rs
```

## Feature Flags

```toml
[features]
default = ["download"]
download = ["reqwest"]           # Model downloads from HuggingFace

# Real adapter integrations
peft = ["peft-rs"]              # Enable peft-rs adapters
qlora = ["qlora-rs", "peft"]    # QLoRA (requires peft)
unsloth = ["unsloth-rs"]        # Optimized kernels
vsa-optim = ["vsa-optim-rs"]    # VSA-accelerated training

# Testing without real deps
mock-peft = []
mock-qlora = []
mock-unsloth = []

# GPU support
cuda = ["candle-core/cuda"]
```

## Key Components

### Configuration (`config.rs`)
```rust
#[derive(Debug, Deserialize)]
pub struct AxolotlConfig {
    pub base_model: String,
    pub adapter: AdapterType,
    pub lora: LoraSettings,
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
    pub output_dir: String,
    pub seed: u64,
}
```

### Dataset Loading (`dataset.rs`)
Supports formats:
- Alpaca (instruction/input/output)
- ShareGPT (conversations)
- Completion (raw text)
- Custom with column mapping

### Model & Export (`model.rs`, `export.rs`)
Loads LLaMA-family architectures, sharded/index safetensors, model-adapter fusion (merging), and exports to PEFT adapter, dense HF, Ollama Modelfile, or llama.cpp GGUF formats.

## Development Commands

```bash
# Check
cargo check -p axolotl-rs

# Check with features
cargo check -p axolotl-rs --features "peft,qlora,unsloth,vsa-optim"

# Test
cargo test -p axolotl-rs

# CPU E2E LoRA Tests
cargo test --features peft --test e2e_lora_cpu

# Run with all features enabled
cargo test --features "peft,qlora,unsloth,vsa-optim"

# CLI validation
cargo run -p axolotl-rs -- validate config.yaml

# Build CLI
cargo build -p axolotl-rs --release
```

## CLI Commands

```bash
# Validate configuration
axolotl validate config.yaml

# Initialize new config
axolotl init --preset llama2-7b

# Train
axolotl train config.yaml

# Merge adapters
axolotl merge --config config.yaml --adapter checkpoints/ --output merged_model/

# Download weights
axolotl download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output ./models

# Export model (formats: peft | hf | ollama-adapter | ollama-merged | gguf)
axolotl export --format peft --config config.yaml --adapter ./checkpoint --output ./peft-adapter
axolotl export --format gguf --config config.yaml --merged ./merged-model --output ./gguf --quantize Q4_K_M
```

## Testing Strategy

- CLI tests: Validate command parsing
- Config tests: YAML parsing and validation
- Integration: End-to-end with real and mock adapters
- GPU tests: Real training loops (ignored without CUDA)

## 1.4 Checklist

- [x] YAML configuration parsing
- [x] Dataset loaders (4 formats)
- [x] CLI interface
- [x] Configuration presets
- [x] Clean compilation (no warnings)
- [x] Working training loop (CPU/GPU)
- [x] Checkpoint save/load
- [x] Adapter merging
- [x] Portable export CLI (PEFT / dense HF / Ollama / llama.cpp GGUF)
- [x] Candle 0.11 upgrade and safetensors 0.8 alignment
- [x] VSA accelerator support
- [x] Robust error handling and custom exceptions
- [x] CI/CD pipeline with GitHub Actions
- [ ] Multi-GPU support
- [ ] Progress reporting
- [ ] Metrics logging (TensorBoard/W&B)
- [ ] Examples directory
- [ ] 100% doc coverage

## Configuration Example

```yaml
base_model: /path/to/local/Llama-2-7b-hf
adapter: lora

lora:
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

dataset:
  path: ./data/alpaca.json
  format: alpaca

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  gradient_accumulation_steps: 4

output_dir: ./outputs/my-model
seed: 42
```
