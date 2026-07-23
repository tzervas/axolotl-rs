#!/bin/bash
# Ensure Cargo.toml is CI-safe: no path deps to sister crates, optional git ref overrides.
#
# Strategy:
# 1. Rewrite any path= peft-rs/qlora-rs/unsloth-rs/vsa-optim-rs lines to crates.io versions
# 2. Strip [patch.crates-io] blocks that point at ../sister paths
# 3. If deps are already git-based, retarget branch from PEFT_RS_REF / PR body markers
#
# PR description markers (optional):
#   peft-rs: branch-or-tag
#   qlora-rs: branch-or-tag
#   unsloth-rs: branch-or-tag

set -euo pipefail

PEFT_REF="${PEFT_RS_REF:-main}"
QLORA_REF="${QLORA_RS_REF:-main}"
UNSLOTH_REF="${UNSLOTH_RS_REF:-main}"

if [ -n "${PR_BODY:-}" ]; then
    echo "Checking PR description for dependency overrides..."
    if echo "$PR_BODY" | grep -q "peft-rs:"; then
        PEFT_REF=$(echo "$PR_BODY" | grep -oP "peft-rs:\s*\K\S+" | head -1)
        echo "  Found peft-rs override: $PEFT_REF"
    fi
    if echo "$PR_BODY" | grep -q "qlora-rs:"; then
        QLORA_REF=$(echo "$PR_BODY" | grep -oP "qlora-rs:\s*\K\S+" | head -1)
        echo "  Found qlora-rs override: $QLORA_REF"
    fi
    if echo "$PR_BODY" | grep -q "unsloth-rs:"; then
        UNSLOTH_REF=$(echo "$PR_BODY" | grep -oP "unsloth-rs:\s*\K\S+" | head -1)
        echo "  Found unsloth-rs override: $UNSLOTH_REF"
    fi
fi

echo ""
echo "CI dependency patch (refs for optional git retarget):"
echo "  peft-rs:    $PEFT_REF"
echo "  qlora-rs:   $QLORA_REF"
echo "  unsloth-rs: $UNSLOTH_REF"
echo ""

# 1) Path deps → crates.io versions (must not require sister checkouts)
# Matches: peft-rs = { path = "../peft-rs", version = "1.1", optional = true }
#      and: peft-rs = { path = "../peft-rs", optional = true }
python3 - <<'PY'
from pathlib import Path
import re

p = Path("Cargo.toml")
text = p.read_text()
original = text

# Remove [patch.crates-io] sections that only exist for local path SoTs
text = re.sub(
    r"\n\[patch\.crates-io\][^\[]*",
    "\n",
    text,
    flags=re.MULTILINE,
)

replacements = {
    "peft-rs": 'peft-rs = { version = "1.0", optional = true }',
    "qlora-rs": 'qlora-rs = { version = "1.0", optional = true }',
    "unsloth-rs": 'unsloth-rs = { version = "1.0", optional = true }',
    "vsa-optim-rs": 'vsa-optim-rs = { version = "0.1", optional = true }',
}

for name, line in replacements.items():
    # path-based optional deps
    text = re.sub(
        rf'^{re.escape(name)}\s*=\s*\{{[^}}]*path\s*=[^}}]*\}}',
        line,
        text,
        flags=re.MULTILINE,
    )

if text != original:
    p.write_text(text)
    print("Rewrote path deps / patch.crates-io for CI (crates.io versions).")
else:
    print("No path deps or path patches found (already CI-safe).")
PY

# 2) Optional: retarget git branch deps if present (legacy dual strategy)
cat > /tmp/patch.sed <<EOF
s|git = "https://github.com/tzervas/peft-rs", branch = "[^"]*"|git = "https://github.com/tzervas/peft-rs", branch = "$PEFT_REF"|g
s|git = "https://github.com/tzervas/qlora-rs", branch = "[^"]*"|git = "https://github.com/tzervas/qlora-rs", branch = "$QLORA_REF"|g
s|git = "https://github.com/tzervas/unsloth-rs", branch = "[^"]*"|git = "https://github.com/tzervas/unsloth-rs", branch = "$UNSLOTH_REF"|g
EOF
sed -i -f /tmp/patch.sed Cargo.toml

echo "Sister-related dependency lines:"
grep -E 'peft-rs|qlora-rs|unsloth-rs|vsa-optim-rs' Cargo.toml || true
echo ""
echo "Cargo.toml patched successfully for CI."
