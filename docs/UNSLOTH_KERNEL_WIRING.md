# unsloth-rs kernel wiring (axolotl-rs)

**Status:** implemented 2026-08-17 on `feat/unsloth-customop-wire`  

**Depends on:** unsloth-rs 1.0.4 CustomOp family ([unsloth-rs#88](https://github.com/tzervas/unsloth-rs/pull/88) + [#89](https://github.com/tzervas/unsloth-rs/pull/89))  
**Parent:** [axolotl-rs#69](https://github.com/tzervas/axolotl-rs/issues/69)

axolotl is a **leaf**. This crate calls unsloth; it does not grow a second kernel
tree. No peft/qlora/core edits in this pass.

## What is Cargo-only today

`unsloth` feature pulls `unsloth-rs` but the train loop still uses:

| Site | Today | Waste |
|------|--------|--------|
| `trainer::compute_cross_entropy_loss` | `to_vec1` labels + `log_softmax` + gather | Host copy of labels every step; full `[N,V]` softmax |
| `llama_common::apply_rotary_emb` | `candle_nn::rotary_emb::rope` | Extra temps (narrow/broadcast/cat) |
| `lora_llama` RoPE | same | same |
| `candle_nn::RmsNorm` in LLaMA | Candle layer | Fine; weights live in `VarBuilder`. Do **not** replace with ones-init `unsloth_rs::RmsNorm`. |
| `RmsNormWrapper` | unused in train | Cosmetic; can always call CustomOp when feature is on |
| MLP SwiGLU | inside `candle-transformers` | Out of this pass (would fork MLP) |

## This pass

1. **CE** — if `feature = "unsloth"`, call
   `unsloth_rs::kernels::chunked_cross_entropy(logits, labels, -100, 4096)`.
   Keep the existing Candle path as the default-feature implementation.
2. **RoPE** — if `feature = "unsloth"`,
   `llama_common::apply_rotary_emb` and `lora_llama` use
   `unsloth_rs::kernels::rope_custom_op`. Layout already matches
   (`x: [B,H,S,D]`, cache `[S, D/2]`).
3. **RmsNormWrapper** — when feature on, always construct `unsloth_rs::RmsNorm`
   (CPU CustomOp works; do not gate on `is_cuda()`).
4. **Cargo.toml** — keep `unsloth-rs = { version = "1.0", optional = true }` so
   crates.io resolve works before 1.0.4 is published. The `unsloth` **feature**
   requires 1.0.4 APIs (`chunked_cross_entropy`, `rope_custom_op`). Use
   `scripts/use-local-path-deps.sh` until 1.0.4 is on crates.io, then bump the
   floor to `1.0.4`.

## Out of this pass

- Replacing `candle_nn::RmsNorm` in LLaMA (need weight-sharing with `VarBuilder`).
- Fused linear+CE (no `dlogits`) — next cut after CE is consumed.
- SwiGLU inside candle-transformers MLP.
- rust-ai-core / peft / qlora.

## Acceptance

- Default `cargo test` still green (no unsloth).
- `cargo test --features unsloth` green against local unsloth-rs 1.0.4.
- CE path does not `to_vec1` labels when unsloth is on.
- No 2× / VRAM claims in README.
