# Fleet standards (tzervas)

Applied from the workstation pack under `plans/fleet-standards/pack/`.

## Workflows

| Workflow | When | Runner |
|----------|------|--------|
| `fleet-ci.yml` | push/PR to main|dev | detect + memory snapshot on self-hosted; **kitchen-sink `cargo check/test` on `ubuntu-latest`** |
| `fleet-security.yml` | push/PR + weekly | same |
| `close-issues-on-main.yml` | PR closed→main | same |
| `reopen-issues-closed-off-main.yml` | PR merged off-main with Closes | same |

## Issue close policy

- **`dev` / feature merges:** `Refs #n` only — issues stay open
- **`main` merges:** `Closes #n` / `Fixes #n`
- **Epics:** close only when delivery PR to main includes `Closes #<epic>`

## Badges

README badges use GitHub Actions SVG for **trunk** branch — live status, not static green.

## Copilot

Automatic Copilot code reviews are **disabled** for fleet-managed repos. Do not request Copilot on PRs.

## Gitleaks / gitignore

- `fleet-security.yml` **must** pass `--config .gitleaks.toml` (native, docker, and podman).
- `.gitignore` must cover `/target/`, `.env*`, keys/PEMs, `.cargo/config.toml` (local path overrides), and `*.crate`.
- This binary crate **tracks** `Cargo.lock` (do not gitignore it).

## Self-hosted rustc memory (fleet-ci)

SIGKILL (signal 9) on `rustc` compiling `candle-transformers` 0.11 on the
homelab podman runner is **OOM**, not a flake. CGU=1 / `CARGO_BUILD_JOBS=1`
did not save it (2026-08-21, PR #83).

axolotl-rs depends on `candle-transformers` 0.11 with **no per-model features**
(~49k LOC). This repo only uses LLaMA. That crate is the RSS peak.

`fleet-ci.yml` therefore:

- **Does not rustc `candle-transformers` on self-hosted.** Job
  `self-hosted memory (no kitchen-sink rustc)` snapshots `free -h` and prints
  `HONEST_CI class=OOM_SKIP`.
- Runs `cargo check/test` (`--lib --bins --tests`) on **GitHub-hosted**
  `ubuntu-latest` (the same graph hosted `ci.yml` already compiles).
- Keeps repo-wide concurrency `axolotl-rs-fleet-ci` (`cancel-in-progress: false`).
- Benches stay on GitHub-hosted `ci.yml`.

Product feature coverage (`peft` / `qlora` / `unsloth`) is hosted
`ci.yml` `adapter-features`.

## Permissions

Workflows use minimum `permissions:` blocks (contents read; issues write only for close/reopen jobs).
