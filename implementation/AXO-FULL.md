# AXO-FULL — axolotl-rs 1.2.0 gap close

**Branch:** `feat/axolotl-full-gap-close`  
**Crate path:** `/root/work/axolotl-rs` (SoT; not `rust-ai/axolotl-rs`)  
**Date:** 2026-07-22  
**Version:** 1.2.0  

## Why "UNSUPPORTED" existed

Honesty gates after Wave-3 found stubs marketed as working (`merge_adapter` returned success fiction; Hub `download` only pulled `reqwest`). User asked for **real** implementation.

## Delivered (MUST list)

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **merge_adapter** fuse LoRA into base, write output, tests succeed | ✅ | `src/model.rs::merge_adapter` — `W' = W + (B@A)*(α/r)`; `test_merge_adapter_fuses_and_writes`; E2E `test_merge_adapter_happy_path_fixture` |
| 2 | **CPU E2E LoRA train** `--features peft`: finite loss + weight change | ✅ | `tests/e2e_lora_cpu.rs::test_cpu_e2e_lora_train_loss_finite_and_progress` |
| 3 | **Checkpoint** save/load LoRA A/B round-trip | ✅ | `LoadedModel::save_adapter_weights` / `load_adapter_weights` (VarMap + adapter_layers); E2E `test_embedded_lora_checkpoint_roundtrip_ab` |
| 4 | **Architecture** refuse non-LLaMA with clear Err | ✅ | `load_model_architecture` + `is_supported_llama_family`; unit + E2E refuse tests |
| 5 | **Sharded safetensors** load or hard error | ✅ | `load_sharded_safetensors`; missing shard → hard error unit test |
| 6 | **Download** local-path first; Hub optional | ✅ | `download_model`: local path passthrough; Hub via `reqwest` blocking + `HF_TOKEN`; CLI `axolotl download` |
| 7 | Version **1.2.0** + honest README matrix | ✅ | `Cargo.toml` 1.2.0; README capability matrix |
| 8 | Gates green | ✅ | See verification |
| 9 | Reports | ✅ | This file + `releases/axolotl-rs-RELEASE.md` |
| 10 | Push branch + PR | ✅ | (see release note for PR URL) |

## Key implementation notes

### merge_adapter
- Loads base `model.safetensors` or sharded index.
- Loads `adapter_model.safetensors` (+ optional `adapter_config.json` for r/α).
- Pairs `*.lora_a.weight` / `*.lora_b.weight` (also HF `lora_A` / `lora_B` / `.default`).
- Writes merged single-file weights + copies config/tokenizer + `merge_info.json`.
- Works **without** `--features peft` (pure Candle matmul).

### Checkpoint
- Embedded `LoraLlama` path: `VarMap::save` / `VarMap::load` on `adapter_model.safetensors`.
- Separate `adapter_layers` path: peft-rs `state_dict` / `load_state_dict`.
- Trainer writes `adapter_config.json` (PEFT-compatible fields).

### Download
- Prefer local directories that already contain weights.
- Else HTTP GET from `huggingface.co/<id>/resolve/main/...` into `cache_dir/<sanitized>/`.
- Docs still recommend `huggingface-cli` for large/gated models.

### Dependencies
- `peft-rs` path `../peft-rs` **1.1**
- `qlora-rs` path `../qlora-rs` **1.1**
- `safetensors` **0.7**, `[patch.crates-io] peft-rs`

## Verification (this agent)

```text
cargo test --lib
  → 123 passed

cargo test --features peft --test e2e_lora_cpu
  → 5 passed

cargo test --test cli_tests
  → 11 passed

cargo check --features peft,qlora
  → Finished (ok)
```

## Out of scope (honest)

- Multi-GPU, packing, DPO
- Gradient checkpointing / mixed precision (config fields warned, not implemented)
- Full non-LLaMA architectures (presets exist; runtime refuses)
- Production parity with Python Axolotl

**PR:** https://github.com/tzervas/axolotl-rs/pull/37
