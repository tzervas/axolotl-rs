# Roadmap

## 1.2.0 (shipped)
- [x] CPU E2E LoRA train proof (tiny fixture)
- [x] Adapter merge into base weights
- [x] Embedded LoRA checkpoint save/load
- [x] Sharded safetensors + non-LLaMA hard refuse
- [x] Hub download CLI (minimal) + local path first-class
- [ ] GPU E2E on Blackwell — BLOCKED:env (nvcc lacks sm_120)

# Axolotl-RS Roadmap

## Honest positioning
Experimental YAML-driven Candle fine-tune **orchestrator**. Not full Python Axolotl parity.
See [README.md](README.md) capability matrix for what works today.

## Near-term (gap-close)
- [x] Docs honesty / retire COMPLETE fiction (PR-013)
- [x] `peft,qlora` features compile with aligned deps (PR-028)
- [x] Training YAML knobs honored + real norms (PR-029)
- [x] CLI merge/download unsupported errors (PR-030)
- [ ] CPU E2E LoRA train on tiny fixture (loss/weight proof)
- [ ] Adapter checkpoint I/O for embedded LoRA path
- [ ] Adapter merge implementation
- [ ] Optional Hub download (or keep unsupported + document)

## Later
- Multi-GPU / packing / DPO / eval loop
- Broader architecture support beyond LLaMA-family loaders

## Non-goals (current)
- Drop-in replacement for Python Axolotl plugins ecosystem
- Claiming production SFT parity at version 1.1.x
