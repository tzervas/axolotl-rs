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
peft-rs = { version = "1.0", optional = true }
qlora-rs = { version = "1.0", optional = true }
unsloth-rs = { version = "1.0", optional = true }
```

After peft-rs **1.1.0** and qlora-rs **1.1.0** are published, bump those floors.
unsloth-rs **1.0.4** is the CustomOp CE/RoPE/RMSNorm release (see
`docs/UNSLOTH_KERNEL_WIRING.md`). Until it is on crates.io, local SoT:

```bash
bash scripts/use-local-path-deps.sh
```


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
