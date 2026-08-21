# Changelog

All notable changes to axolotl-rs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- BitNet b1.58 QAT recipe at `examples/configs/bitnet_qat_2b4t.yaml` (documentation
  only): intelligent conversion knobs (rewire SubLN + gated ReLU², STE QAT, KL
  distill, packed TwoBit, 5080 VRAM cap). `adapter: bitnet` is rejected until
  candle 0.9/0.11 align. No `bitnet-quantize` cargo dependency (axolotl-rs is
  0.11; quantize v0.5.1 is 0.9). AbsMean PTQ is forbidden.

### Fixed
- `reopen-issues-closed-off-main.yml` is valid YAML again: replace the
  column-0 `python3 <<'PY'` heredoc with `python3 -c` so GitHub stops
  reporting a startup_failure on every push.
- Self-hosted `fleet-ci` SIGKILL while compiling `candle-transformers` 0.11:
  repo-wide concurrency (PR and trunk no longer overlap), rustc codegen-units
  capped at 1, debuginfo off, and check/test sized to lib+bins+tests (benches
  stay on GitHub-hosted CI). Signal 9 on that crate is OOM, not a flake.

### Changed
- `.gitignore` covers env/key files, `.cargo/config.toml`, and crate artifacts
  (`Cargo.lock` remains tracked). `fleet-security.yml` gitleaks now requires
  `.gitleaks.toml` (`--config`).

## [1.4.0] - 2026-08-20

### Added
- CLI `axolotl export --format peft|hf|ollama-adapter|ollama-merged|gguf` (`src/export.rs`).
  GGUF is delegated to llama.cpp (`convert_hf_to_gguf.py` + `llama-quantize`); missing tools
  print the exact commands and exit 2. Never writes a custom `GGUF_TYPE_QLORA_NF4`.
- Hub-safe PEFT save: one `adapter_model.safetensors` with
  `{base_model.model}.{module}.lora_A.default.weight` / `lora_B.default.weight`
  (local writer; peft-rs 1.3.0 also ships `save_multi_module_pretrained_hf`).
- `adapter_config.json` fields: `peft_type`, `r`, `lora_alpha`, `target_modules`, `bias`,
  `task_type`, `base_model_name_or_path`, `lora_dropout`, `inference_mode`, `use_rslora`,
  `use_dora`.
- Merge copies non-weight HF **file** sidecars (`tokenizer.model`, `chat_template.jinja`, …);
  nested template directories are not copied. Applies rsLoRA scale when
  `adapter_config.use_rslora`, pairs native and HF LoRA keys, and looks up
  `{module}.weight` then `model.{module}.weight`.
- README Deploying section (vLLM / Ollama / llama.cpp).
- `.cz.toml` (commitizen conventional commits, semver 1.x; no `major_version_zero`).

### Changed
- `trainer::save_checkpoint` passes `Some(&self.config)` into `save_adapter_weights_hf`
  and does **not** overwrite `adapter_config.json` afterwards.
- Merge rejects packed U8/U32 (NF4) base or adapter tensors and writes dense F16/BF16/F32 `W`.

### Fixed
- README safetensors pin note now matches Cargo.toml (**0.8**, not 0.7).
- `export --format gguf` requires `llama-quantize` only when `--quantize` is not
  F16/BF16/F32/NONE; convert `--outtype` follows that flag.
- Ollama adapter export copies the PEFT dir next to the Modelfile (`ADAPTER ./adapter`).
- `save_adapter_weights` without a config no longer writes placeholder `adapter_config.json`.

## [1.3.0] - 2026-08-20

Candle types appear in the public API, so a candle minor bump is a **breaking
change for downstream**. Unpublished GitHub `1.2.0` (Candle 0.9) is superseded.

### Changed
- **candle-core / candle-nn / candle-transformers `0.9` → `0.11`**
- Optional sisters **peft-rs / qlora-rs / unsloth-rs `1.2`** (on crates.io)
- **MSRV 1.92 → 1.96** (`rust-toolchain.toml` 1.96.1). Self-hosted fleet-ci /
  publish / release install that toolchain instead of work-image rustc 1.88.
- Package version **1.3.0**. Cargo / GitHub description drops untested
  “VSA acceleration” claim (`vsa-optim` remains an optional, untested feature).

### Documentation
- Archived remaining COMPLETE/PHASE GPU banners at repo root to `docs/archive/`.
  `CUDA_STATUS.md` stays as the CUDA RMSNorm honesty note.

## [1.2.0] - 2026-07-22

### Added
- **CPU E2E LoRA train proof** on a tiny LLaMA-shaped fixture (`src/fixture.rs`,
  `tests/e2e_lora_cpu.rs`): finite loss, non-zero grads, checkpoint A/B present.
- **Adapter merge**: `merge_adapter` fuses LoRA ΔW into base linear weights
  (`W' = W + (B @ A) * (α/r)`), writes merged `model.safetensors` + `merge_info.json`.
  CLI `axolotl merge` succeeds on the fixture happy path.
- **Embedded LoRA checkpoint save/load**: round-trips A/B via trainable `VarMap`
  (`adapter_model.safetensors` + `adapter_config.json`).
- **Sharded safetensors load**: `model.safetensors.index.json` + shards; missing shard
  is a hard error (no silent stub).
- **Architecture honesty**: non-LLaMA families return `Unsupported model architecture`
  listing supported families (no 10×10 stub train).
- **Hub download**: minimal `reqwest` client (`axolotl download <model_id>`); local path
  remains first-class. Gated models need `HF_TOKEN` or `huggingface-cli`.
- **Optimizer init** on trainable adapter params at train start (was missing).
- Tiny fixture helpers: `write_tiny_llama_fixture`, `write_tiny_alpaca_jsonl`.

### Changed
- Version **1.2.0**; capability matrix documents green checks only for real features.
- **CI-safe deps:** peft/qlora/unsloth are crates.io optional versions (no committed path deps).
  Local SoT: `scripts/use-local-path-deps.sh` → gitignored `.cargo/config.toml` paths.
  After peft/qlora **1.1.0** publish, bump optional floors to `1.1` / `1.1` / `1.0.3`.
- reqwest gains `blocking` for Hub download.
- CLI merge/download docs no longer claim `UNSUPPORTED` for happy paths.
- README + `docs/DEPENDENCIES.md` describe the DAG and fleet override policy.
### Fixed
- LoRA A/B capture reads real VarMap values (not empty placeholders).
- Checkpoint path saves embedded LoRA even when `adapter_layers` is `None`.

### Notes / GPU
- `cargo test --features peft,cuda` **BLOCKED:env** on this host: RTX 5080 (sm_120)
  but installed `nvcc` max arch is 90. CPU gates remain green with `AXOLOTL_FORCE_CPU=1`.

## [1.1.1] - 2026-01-24

### Added
- CUDA-first device selection with explicit CPU fallback warnings
- Environment overrides: `AXOLOTL_FORCE_CPU`, `AXOLOTL_CUDA_DEVICE`

### Changed
- Bumped minimum Rust version to 1.92
- README badge alignment cleanup

## [1.1.0] - 2026-01-27

### Added
- **VSA-Accelerated Training**: Integrated `vsa-optim-rs` for deterministic gradient prediction
- `VSAAccelerator` wrapper with configurable training phases (WARMUP → FULL → PREDICT → CORRECT)
- Deterministic phase training with closed-form weighted least squares gradient prediction
- `VSAConfig` for fine-grained control over VSA dimensions, prediction windows, and memory budgets
- Ternary gradient accumulation using balanced `{-1, 0, +1}` representation
- Hyperdimensional bind/bundle/unbind operations for gradient compression
- Comprehensive integration tests for VSA acceleration
- Documentation for `vsa_accel` module with architecture overview

### Changed
- Improved memory efficiency through VSA gradient compression (experimental `vsa-optim` feature)

### Notes (honesty, PR-013)
- `TrainingConfig` does **not** expose a `vsa_config` field; VSA is configured via
  `VSAAcceleratorConfig` under the optional `vsa-optim` feature only.
- Version 1.1.x remains an orchestrator scaffold; see README capability matrix.

## [1.0.1] - 2026-01-24

### Fixed
- Fixed `std::path::Path` import missing when `peft` feature enabled
- Fixed `lora_params` variable reference in feature-gated code block
- Compilation now succeeds with `--features "peft,qlora,unsloth"`

## [1.0.0] - 2026-01-24

### Added
- Dynamic CI dependency configuration for sister projects (peft-rs, qlora-rs, unsloth-rs)
- GitHub-based dependency strategy with branch pinning for CI builds
- Comprehensive LoRA target injection tests (per-layer configuration)
- QLoRA training integration tests
- GPU checkpoint save/load tests

### Changed
- Resolved all clippy warnings for production quality
- Updated dependencies to use GitHub branches by default for development
- Improved code organization with dead code annotations for future use

### Fixed
- Unused import and variable warnings cleaned up
- All compilation warnings resolved

---

### Added (from 0.1.0-dev)
- Initial project scaffold with Rust port of Axolotl
