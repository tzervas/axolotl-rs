# Roadmap

## 1.2.0 (shipped)
- [x] CPU E2E LoRA train proof (tiny fixture)
- [x] Adapter merge into base weights
- [x] Embedded LoRA checkpoint save/load
- [x] Sharded safetensors + non-LLaMA hard refuse
- [x] Hub download CLI (minimal) + local path first-class
- [ ] GPU E2E production-complete (host nvcc 13.1 lists `sm_120`; 10-step smoke later)

# Axolotl-RS Roadmap

## Honest positioning
Experimental YAML-driven Candle fine-tune **orchestrator**. Not full Python Axolotl parity.
See [README.md](README.md) capability matrix for what works today.

CPU E2E LoRA, adapter merge, checkpoint I/O, and Hub download are **shipped**
(1.2.0+; see the 1.2.0 list above and the README matrix). Do not re-open those
as unchecked near-term items.

## Near-term (gap-close)
- [x] Docs honesty / retire COMPLETE fiction (PR-013)
- [x] `peft,qlora` features compile with aligned deps (PR-028)
- [x] Training YAML knobs honored + real norms (PR-029)
- [x] CLI merge/download unsupported errors (PR-030)
- [ ] GPU E2E production-complete (not blocked on nvcc `sm_120`)
- [ ] Unsloth RMS CustomOp on LoRA train graph (`RmsNormOp::bwd` exists; not wired)
- [ ] MLP SwiGLU / fused CE / attn

## Later
- Multi-GPU / packing / DPO / eval loop
- Broader architecture support beyond LLaMA-family loaders

## Non-goals (current)
- Drop-in replacement for Python Axolotl plugins ecosystem
- Claiming production SFT parity at version 1.1.x
