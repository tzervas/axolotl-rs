# CUDA/GPU Support Status

> **Honesty:** GPU training is **not** production-complete. A 10-step GPU
> smoke is a later PR. This file is host + kernel facts, not a COMPLETE banner.

## Host facts (this SoT machine, 2026-08-21)

| Check | Result |
|-------|--------|
| GPU | NVIDIA GeForce RTX 5080 (16GB, compute 12.0 / `sm_120`) |
| Driver | 610.57.04 (`nvidia-smi` OK, `/dev/nvidia0` present) |
| CUDA toolkit | **13.1** (`nvcc` **V13.1.115**) |
| `nvcc --list-gpu-code` | includes **`sm_120`** (also `sm_100` / `sm_121`; **not** capped at 90) |
| Candle | **0.11** (`candle-core` / `candle-nn` / `candle-transformers`) |
| `candle-nn` RMSNorm CUDA | **present** — `candle-nn` 0.11 `src/ops.rs` implements `rms-norm` `cuda_fwd` |

`gpu` as a GitHub Actions runner label is **host routing** (akula-prime RAM),
not “enable CUDA”. Kitchen-sink `cargo check/test` stays `gpu` +
`scribe-cpu-build`. Do not `--all-features`. Do not add `large`.

### Claims that are **false** (do not restore)

These were true of an older toolkit / Candle **0.9** pin. They are **not** true
on this host:

- `nvcc` max arch **90** / cannot target **120**
- live Candle is **0.9**
- GPU training blocked because Candle has **no** CUDA `rms-norm`

## What still blocks production GPU E2E

Candle's **optimized** `rms_norm` uses `apply_op2_no_bwd` (`candle-nn` 0.11
`ops.rs`). That kernel does not track gradients.

Train (`LoraLlama` / `QLoraLlama`) therefore uses `candle_nn::RmsNorm` +
**`forward_diff`** so LoRA grads flow. See comments in `src/lora_llama.rs`.

`--features unsloth` ships `RmsNormWrapper`, and unsloth-rs `RmsNormOp::bwd`
**exists** (`unsloth-rs` `src/kernels/custom_op/rmsnorm.rs`). That wrapper is
**not** on the LoRA train graph (later PR; do not wire it here).

GPU E2E is still **not** production-complete. The 10-step GPU smoke
(`test_gpu_quick_iteration`) is a **later PR**. 7B GPU tests stay ignored.

## Working (CPU)

- CPU LoRA E2E (`cargo test --features peft --test e2e_lora_cpu`) — tiny LLaMA fixture
- Device selection: `AXOLOTL_FORCE_CPU` / `AXOLOTL_CUDA_DEVICE`
- CUDA feature plumbing (`--features cuda`) when building kernels for this GPU

```bash
export AXOLOTL_FORCE_CPU=1
cargo test --lib
cargo test --features peft --test e2e_lora_cpu
cargo check --features peft,qlora
```

## GPU tests (ignored; not this PR)

- `test_gpu_quick_iteration` — 10 steps, SmolLM2-135M (later smoke PR)
- `test_gpu_loss_convergence_100_steps`
- `test_gpu_gradient_flow`
- `test_gpu_tinyllama_*` / `test_gpu_llama7b_*` — stay ignored

```bash
cargo test --features peft,cuda --test gpu_training --release -- --ignored
```
