# axolotl-rs 1.2.0 Release Notes

**Package:** `axolotl-rs`  
**Version:** 1.2.0  
**Branch:** `feat/axolotl-full-gap-close`  
**Date:** 2026-07-22  

## Summary

Closes the honesty-gate gaps from 1.1.x: real **LoRA merge**, **checkpoint A/B I/O**, **CPU E2E train proof**, **sharded weight load**, **non-LLaMA refuse**, and **local-first download** (Hub pull optional via `download` feature).

## Highlights

- `axolotl merge` fuses LoRA into base linear weights and writes a deployable model directory.
- `axolotl download` resolves local paths or pulls Hub files with `reqwest` (set `HF_TOKEN` when gated).
- `--features peft` CPU E2E: train on tiny fixture → finite loss + adapter weight change + checkpoint round-trip.
- Sharded HuggingFace safetensors: multi-file load or explicit missing-shard error (no stub fallback).
- Architecture gate: only LLaMA-family (`model_type=llama` / `LlamaForCausalLM`).

## Capability matrix (honest)

| Surface | 1.2.0 |
|---------|-------|
| YAML / datasets / CLI train | ✅ |
| LoRA train (`peft`) | ✅ (LLaMA-family, local weights) |
| QLoRA train (`peft,qlora`) | ✅ code path |
| Adapter merge | ✅ |
| Checkpoint A/B | ✅ |
| Sharded safetensors | ✅ |
| Hub download | ✅ (optional; local preferred) |
| Non-LLaMA | ❌ clear error |
| Multi-GPU / DPO | ❌ |

## Upgrade

```toml
# Cargo.toml consumers
axolotl-rs = "1.2.0"
# Optional:
# features = ["peft"] or ["peft", "qlora"]
```

Sister pins in monorepo SoT: peft-rs **1.1**, qlora-rs **1.1**, safetensors **0.7**.

## Verify

```bash
cargo test --lib
cargo test --features peft --test e2e_lora_cpu
cargo check --features peft,qlora
```

## Links

- Implementation report: [`implementation/AXO-FULL.md`](../implementation/AXO-FULL.md)
- Prior honesty note: plans `AXOLOTL_UNSUPPORTED.md` (gap-close ledger)
