# axolotl-rs (agent notes)

**SoT:** [`Cargo.toml`](Cargo.toml) + [README.md](README.md) + [docs/VERSIONING.md](docs/VERSIONING.md).

YAML-driven **LLaMA-family LoRA trainer/orchestrator**. **Not** Python Axolotl parity.

- Live version is `Cargo.toml` (1.x). Candle **0.11**, MSRV **1.96**.
- Default features: config, datasets, CLI (`validate`/`init`/`train`/`merge`/`export`). LoRA needs `--features peft`; QLoRA `--features peft,qlora`; CustomOp CE/RoPE `--features unsloth`.
- `LoraLlama` / `QLoraLlama` `Module::forward` is real per-layer inject. `LoadedModel::forward_with_adapters` delegates to that.
- Unsloth `RmsNormWrapper` is **not** on the LoRA train graph (`candle_nn::RmsNorm` + `forward_diff`).
- Homelab CPU must **not** rustc `candle-transformers` (OOM/SIGKILL). `fleet-ci` `detect stack` is GitHub-hosted; kitchen-sink `cargo check/test` is **akula-prime** (`gpu` label = host routing, not CUDA).
- Hosted `ci.yml` `adapter-features` is the peft/qlora/unsloth test gate.
- BitNet QAT is recipe-only; `adapter: bitnet` is rejected. No `bitnet-quantize` cargo dep.
