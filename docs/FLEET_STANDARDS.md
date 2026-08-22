# Fleet standards (tzervas)

Applied from the workstation pack under `plans/fleet-standards/pack/`.

## Workflows

| Workflow | When | Runner |
|----------|------|--------|
| `fleet-ci.yml` | push/PR to main|dev | detect + OOM-skip policy on GitHub-hosted (do not queue the snapshot on homelab — that blocked GPU kitchen-sink); kitchen-sink `cargo check/test` on **akula-prime** (`gpu` + `rust` + `scribe-cpu-build`; RAM not CUDA). GHCR `scribe-cpu-build` has 1.96.1 baked. Not the autodetect caller — detect and kitchen-sink must not share a runner. |
| `fleet-security.yml` | push/PR + weekly | `[self-hosted, linux, x64, podman, rust]` |
| `close-issues-on-main.yml` | PR closed→main | GitHub-hosted (API-only) |
| `reopen-issues-closed-off-main.yml` | PR merged off-main with Closes | same |

Action pins follow `tzervas/ap-workflows` `pins/actions.yml` (`actions/checkout@v7`, `actions/upload-artifact@v7`, `astral-sh/setup-uv@v9`). Do **not** `--all-features` (pulls `cuda`).

## License

MIT (`LICENSE`). Third-party crates and Apache-2.0 inspirations: `NOTICE`.
`fleet-security.yml` job `cargo deny licenses` (GitHub-hosted) is fail-closed.
Allow-list is permissive only (no GPL / AGPL / MPL).

## Issue close policy

- **`dev` / feature merges:** `Refs #n` only — issues stay open
- **`main` merges:** `Closes #n` / `Fixes #n`
- **Epics:** close only when delivery PR to main includes `Closes #<epic>`

## Badges

README badges use GitHub Actions SVG for **trunk** branch — live status, not static green.

## Copilot

Automatic Copilot code reviews are **disabled** for fleet-managed repos. Do not request Copilot on PRs.

## Gitleaks / gitignore

- **Local pre-commit is the real gate.** `bash scripts/install-hooks.sh` sets
  `core.hooksPath=.githooks`. That hook runs `gitleaks protect --staged`
  (`scripts/gitleaks-staged.sh`). Missing gitleaks **fails the commit** (not a
  skip). A finding in staged files: unstage it. A finding that already hit a
  remote: **rotate the credential** — rewriting history does not un-leak it.
  `git commit --no-verify` is how secrets land in git.
- `fleet-security.yml` is defense-in-depth after push. It **must** pass
  `--config .gitleaks.toml`.
- `.gitignore` must cover `/target/`, `.env*`, keys/PEMs, `.cargo/config.toml` (local path overrides), and `*.crate`.
- This binary crate **tracks** `Cargo.lock` (do not gitignore it).

## Self-hosted rustc memory (fleet-ci)

SIGKILL (signal 9) on `rustc` compiling `candle-transformers` 0.11 on the
homelab podman runner is **OOM**, not a flake. CGU=1 / `CARGO_BUILD_JOBS=1`
did not save it (2026-08-21, PR #83).

axolotl-rs depends on `candle-transformers` 0.11 with **no per-model features**
(~49k LOC). This repo only uses LLaMA. That crate is the RSS peak.

`fleet-ci.yml` therefore:

- **Does not rustc `candle-transformers` on homelab CPU.** Job
  `self-hosted memory (no kitchen-sink rustc)` prints `HONEST_CI class=OOM_SKIP`
  on GitHub-hosted so a down homelab runner cannot hold the concurrency group.
- Runs `cargo check/test` (`--lib --bins --tests`) on **akula-prime**
  (RTX 5080 desktop) via `gha-runner-ctl` GPU jobs:
  `runs-on: [self-hosted, linux, x64, podman, gpu, rust, scribe-cpu-build]`.
  `gpu` routes here. It does **not** enable CUDA. `rust` selects
  `ghcr.io/tzervas/ap-fleet-work-images/scribe-cpu-build:dev` (Rust 1.96.1
  + clippy baked; no `dtolnay/rust-toolchain` at job time). The worker is
  8c/16 GiB (`GHA_MEMORY=16g`). Do not add `large` — that label is not
  registered on the GPU runner, so GitHub would never assign the job.
  runner-ctl pulls the GHCR ref on demand (`GHA_PULL_POLICY=missing`); do
  not stash `localhost/*` copies as the source of truth.
- Keeps repo-wide concurrency `axolotl-rs-fleet-ci` (`cancel-in-progress: false`).
- Benches stay on GitHub-hosted `ci.yml`. GitHub-hosted Test Suite /
  `adapter-features` remain the public product gates.

Product feature coverage (`peft` / `qlora` / `unsloth`) is hosted
`ci.yml` `adapter-features`.

## Permissions

Workflows use minimum `permissions:` blocks (contents read; issues write only for close/reopen jobs).
