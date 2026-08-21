# unsloth-rs kernel wiring (axolotl-rs)

axolotl is a **leaf**. This crate *calls* unsloth; it does not grow a second
kernel tree.

`Cargo.toml`: `unsloth-rs = { version = "1.2", optional = true }` behind
`--features unsloth`. Default `cargo test` does not pull it.

## Wired (feature `unsloth`)

| Site | Default | With `unsloth` |
|------|---------|----------------|
| `trainer::compute_cross_entropy_loss` | host `to_vec1` labels + `log_softmax` | `chunked_cross_entropy` (ignore −100) |
| `llama_common::apply_rotary_emb` / `LoraAttention` | `candle_nn::rotary_emb::rope` | `rope_custom_op` |
| `RmsNormWrapper` | ones-init CPU RMS | `unsloth_rs::kernels::RmsNorm` (CustomOp). **Not** used by `LoraLlama` / `QLoraLlama` — those keep `candle_nn::RmsNorm` + `forward_diff` so LoRA grads track. |

`apply_rotary_emb_ids` (packed positions) is feature-gated.

## Not in this pass

- Replacing `candle_nn::RmsNorm` on the LoRA train graph (CustomOp has no `bwd`; train uses `forward_diff`).
- MLP SwiGLU inside candle-transformers.
- Fused linear+CE (would skip materializing `[N,V]` logits).
- No 2× / VRAM claims.

## Check

```bash
cargo test
cargo check --features unsloth
cargo test --features peft,qlora,unsloth --lib --bins --tests
```

Hosted CI job `adapter-features` runs the isolate `cargo check` plus the combined
feature test. Default `cargo test` / self-hosted `fleet-ci` stay default features.
