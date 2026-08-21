# Fleet standards (tzervas)

Applied from the workstation pack under `plans/fleet-standards/pack/`.

## Workflows

| Workflow | When | Runner |
|----------|------|--------|
| `fleet-ci.yml` | push/PR to main|dev | self-hosted linux x64 podman |
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

`fleet-ci` `cargo check/test` runs on the podman runner. SIGKILL (signal 9) on
`rustc` compiling `candle-transformers` is **OOM**, not a flake.

axolotl-rs is the fleet crate that depends on `candle-transformers` 0.11. That
crate has **no per-model features** (~49k LOC, every architecture) while this
repo only uses LLaMA (`models::llama` + `utils::repeat_kv`). Compiling it is
the RSS peak of this job.

`fleet-ci.yml` therefore:

- Uses a **repo-wide** concurrency group (`axolotl-rs-fleet-ci`) with
  `cancel-in-progress: false` so a PR and a `main` push cannot compile that
  crate at the same time (per-ref groups allowed that; both rustc processes
  died together).
- Caps rustc with `CARGO_BUILD_JOBS=1`, `CARGO_INCREMENTAL=0`,
  `CARGO_PROFILE_{DEV,TEST}_{DEBUG,CODEGEN_UNITS}` = `0` / `1`, and
  `RUSTFLAGS=-C codegen-units=1 -C debuginfo=0` (profile.dev defaults to 256
  codegen-units even with `CARGO_BUILD_JOBS=1`).
- Checks/tests **lib + bins + tests** only. Benches (`criterion` / plotters)
  stay on GitHub-hosted `ci.yml`.
- Snapshots `free -h` before cargo so a future OOM is diagnosable.

GitHub-hosted `ci.yml` (Test Suite) remains the full coverage gate.

## Permissions

Workflows use minimum `permissions:` blocks (contents read; issues write only for close/reopen jobs).
