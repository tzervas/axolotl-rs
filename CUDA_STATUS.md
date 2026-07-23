# CUDA/GPU Support Status

> **Honesty:** GPU training is **not** production-complete on this host. See README capability matrix.

## Current Status (2026-07-22 / 1.2.0)

### Working
- ✅ CPU LoRA E2E (`cargo test --features peft --test e2e_lora_cpu`) — tiny LLaMA fixture
- ✅ CUDA feature plumbing (`--features cuda`) when toolkit supports the GPU arch
- ✅ Device selection with `AXOLOTL_FORCE_CPU` / `AXOLOTL_CUDA_DEVICE`

### BLOCKED:env (this host)

| Check | Result |
|-------|--------|
| `nvidia-smi` | RTX 5080 visible |
| `/dev/nvidia0` | present |
| `cargo check --features peft,cuda` | **FAIL** — `nvcc cannot target gpu arch 120` |
| Available nvcc targets | 50…90 only (no sm_120 / Blackwell) |

**Cause:** Candle/CUDA build detects compute capability **120** (Blackwell) but the installed
CUDA toolkit `nvcc` only goes up to arch **90**. Rebuild requires a newer CUDA toolkit with
Blackwell support (or force a lower `CUDA_COMPUTE_CAP` with binary incompatibility risk).

**CPU gate remains green:**

```bash
export AXOLOTL_FORCE_CPU=1
cargo test --lib
cargo test --features peft --test e2e_lora_cpu
cargo check --features peft,qlora
```

### Known Candle limitations (when CUDA does build)

#### RMS Norm CUDA Implementation
GPU training may still fail with:
```
no cuda implementation for rms-norm
```

Candle 0.9.x may lack CUDA kernels for RMS normalization used by LLaMA-family models.
Workarounds: cuDNN feature, CPU fallback, or upstream kernels.

### GPU Test Suite (when env allows)
- `test_gpu_quick_iteration` - 10 steps, SmolLM2-135M
- `test_gpu_loss_convergence_100_steps` - 100 steps
- `test_gpu_gradient_flow` - gradient flow
- `test_gpu_tinyllama_*` / `test_gpu_llama7b_*` - larger models

```bash
cargo test --features peft,cuda --test gpu_training --release -- --ignored
```

### System snapshot (this SoT host)
- **GPU:** NVIDIA GeForce RTX 5080 (16GB, compute 12.0)
- **Driver:** 610.x / nvidia-smi OK
- **nvcc arch max:** 90 → **cannot build kernels for 120**
