# Changelog

All notable changes to axolotl-rs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### CI (pending operator merge of prerequisite PRs into `dev`)

These entries describe open work that is **not yet on `dev`**. They will move under
the next published section only after the listed PRs land.

- **#40** — Fix invalid YAML in `reopen-issues-closed-off-main.yml`: the embedded
  Python heredoc was at column 0 and terminated the `|` block scalar, causing a
  permanent `startup_failure` on every push. Indent the heredoc so YAML stays valid
  and the shell still receives the intended script.
- **#44** — Call centralized `mycelium-workflows` `reusable-rust-ci.yml@v1` for the
  core gate (`fmt` / `clippy` / `check` / `test` / docs / benches). Keep coverage,
  cargo-audit/deny, and MSRV jobs local. Drop PR-body dependency override injection
  (script-injection channel). **Blocked on `@v1` tag existing upstream.**
- **#43** — Replace local `fleet-ci.yml` / `fleet-security.yml` with thin callers into
  mycelium fleet reusables with branch-tier strictness (dev vs main). **Blocked on
  fleet reusable workflows shipping under `@v1` (currently only on feature branches).**
- **#39** — Event-driven auto-merge for Jules PRs (`jules-*` branches / known authors),
  gated on every non-skipped check concluding success **and** at least one success
  (empty check list refuses). **Operator judgement: do not land until this repo has
  real required status-check contexts on the ruleset** (today `protec-main` requires
  a PR but zero status checks — contract §6).

### Closed without merge (superseded / duplicate)

- **#42** — Duplicate of #40 (same heredoc fix; #40 kept the explanatory comment).
- **#36** — Bot “comprehensive maintenance” PR. Unique useful work (binary imports
  from the library crate, cargo-deny v2, clippy cleanup) already on `dev` via
  `85aff79` / `5c03a8e` / `0b5d004` and on `main` via #37/#38. Remaining tip-only
  deltas were lint suppressions, not fixes.

## [1.2.0] - 2026-07-22

### Version reconciliation

- **crates.io published:** 1.1.1
- **in-tree `Cargo.toml`:** 1.2.0 (already bumped; never published)
- **No further bump in this release PR.** The 1.2.0 MINOR correctly describes the
  feature set below (merge, E2E LoRA, checkpoints, shards, download). Consolidated
  open PRs are CI-only and do not warrant 1.2.1 or 1.3.0 under conventional-commit
  rules. Grandfathered 1.x — do not renumber to 0.x or cut 2.0.0.

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
- Commitizen config (`.cz.toml`) with `major_version_zero = false` (grandfathered 1.x)
  and `version_files = ["Cargo.toml:version"]`.

### Changed
- Version **1.2.0**; capability matrix documents green checks only for real features.
- **CI-safe deps:** peft/qlora/unsloth are crates.io optional versions (no committed path deps).
  Local SoT: `scripts/use-local-path-deps.sh` → gitignored `.cargo/config.toml` paths.
  After peft/qlora **1.1.0** publish, bump optional floors to `1.1` / `1.1` / `1.0.3`.
- reqwest gains `blocking` for Hub download.
- CLI merge/download docs no longer claim `UNSUPPORTED` for happy paths.
- README + `docs/DEPENDENCIES.md` describe the DAG and fleet override policy.
- Binary entrypoint imports from the library crate instead of re-declaring modules
  (dedupe on `dev` cleanup series).

### Fixed
- LoRA A/B capture reads real VarMap values (not empty placeholders).
- Checkpoint path saves embedded LoRA even when `adapter_layers` is `None`.
- Training honors `gradient_accumulation_steps`, `lr_scheduler`, `warmup_ratio`,
  and `max_grad_norm`; grad/param norms are real L2 values.

### Notes / GPU
- `cargo test --features peft,cuda` **BLOCKED:env** on hosts without matching CUDA
  arch support. CPU gates remain green with `AXOLOTL_FORCE_CPU=1`.

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
