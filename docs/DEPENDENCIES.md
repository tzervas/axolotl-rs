# axolotl-rs dependency graph

## DAG (no cycles)

```text
candle-*, tokenizers, safetensors, …
        │
        ▼
   ┌──────────┐     optional features
   │axolotl-rs│─────────────────────────────┐
   └────┬─────┘                             │
        │ peft / qlora / unsloth            │
        ▼                                   │
   ┌─────────┐    ┌─────────┐    ┌──────────┐
   │ peft-rs │◄───│qlora-rs │    │unsloth-rs│
   └─────────┘    └────┬────┘    └──────────┘
                       │ peft-rs only
                       ▼
                  (no back-edge to axolotl)
```

**Rule:** axolotl is a **leaf**. peft and unsloth never depend on axolotl or qlora.
qlora may depend on peft only.

## Committed Cargo.toml

Optional deps use **crates.io versions only** so CI works without sister checkouts:

```toml
peft-rs = { version = "1.2", optional = true }
qlora-rs = { version = "1.0", optional = true }  # bump to 1.2 after crates.io 1.2.0
unsloth-rs = { version = "1.0", optional = true }  # bump to 1.2 after crates.io 1.2.0
```

After peft-rs **1.1.0** and qlora-rs **1.1.0** are published, bump floors to `1.1` /
`1.1` / `1.0.3`.

## Local SoT (fleet)

```bash
bash scripts/use-local-path-deps.sh        # enable
bash scripts/use-local-path-deps.sh --clear  # disable
```

Writes gitignored `.cargo/config.toml` with `paths = [...]` — does **not** change
committed `Cargo.toml`.

## Features

| Feature | Pulls |
|---------|--------|
| `download` (default) | `reqwest` |
| `peft` | `peft-rs` |
| `qlora` | `qlora-rs` + `peft` |
| `unsloth` | `unsloth-rs` |
| `cuda` | `candle-core/cuda` |
