# Archived — do not treat as current status

> **Honesty note (PR-013):** This document previously claimed “COMPLETE” / “production-ready”
> status that overstated the crate. Historical content lives at
> [`docs/archive/GPU_IMPLEMENTATION_COMPLETE.md`](docs/archive/GPU_IMPLEMENTATION_COMPLETE.md) and is retained only for archaeology.
>
> **Current status:** See [README.md](README.md) capability matrix.
> - Default features do **not** enable LoRA/QLoRA.
> - Real adapter paths require `--features peft` / `peft,qlora` and local model weights.
> - GPU E2E remains blocked by Candle CUDA RMSNorm gaps (see [CUDA_STATUS.md](CUDA_STATUS.md)).
> - Adapter merge and Hub download are **unsupported**.

