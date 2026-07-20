#!/bin/bash
# Script to dynamically patch Cargo.toml dependencies based on CI inputs
# This allows PRs to specify custom sister project versions via:
# 1. Workflow dispatch inputs
# 2. PR description markers (e.g., `peft-rs: feature-branch`)
# 3. Environment variables

set -e

# Ensure local path dependency for vsa-optim-rs is satisfied in CI
if [ ! -d "../vsa-optim-rs" ]; then
    echo "Creating dummy vsa-optim-rs dependency at ../vsa-optim-rs..."
    mkdir -p ../vsa-optim-rs/src
    cat > ../vsa-optim-rs/Cargo.toml <<EOF
[package]
name = "vsa-optim-rs"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.9"
thiserror = "1.0"
EOF
    cat > ../vsa-optim-rs/src/lib.rs <<EOF
use std::collections::HashMap;
use candle_core::{Device, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeterministicPhase {
    Warmup,
    Full,
    Predict,
    Correct,
}

#[derive(Debug, Clone)]
pub struct DeterministicPhaseConfig {
    pub warmup_steps: usize,
    pub full_steps: usize,
    pub predict_steps: usize,
    pub correct_every: usize,
    pub adaptive_phases: bool,
    pub history_window: usize,
    pub loss_threshold: f32,
    pub max_grad_norm: f32,
}

#[derive(Debug, Clone)]
pub struct StepInfo {
    pub total_step: usize,
    pub phase: DeterministicPhase,
}

#[derive(Debug, Clone)]
pub struct Stats {
    pub total_steps: usize,
    pub full_steps: usize,
    pub predict_steps: usize,
    pub speedup: f32,
}

pub struct DeterministicPhaseTrainer {
    current_step: usize,
    config: DeterministicPhaseConfig,
}

impl DeterministicPhaseTrainer {
    pub fn new(
        _shapes: &[(String, Vec<usize>)],
        config: DeterministicPhaseConfig,
        _device: &Device,
    ) -> Result<Self, String> {
        Ok(Self {
            current_step: 0,
            config,
        })
    }

    pub fn begin_step(&mut self) -> Result<StepInfo, String> {
        self.current_step += 1;
        Ok(StepInfo {
            total_step: self.current_step,
            phase: self.current_phase(),
        })
    }

    pub fn needs_backward(&self) -> bool {
        matches!(
            self.current_phase(),
            DeterministicPhase::Warmup | DeterministicPhase::Full | DeterministicPhase::Correct
        )
    }

    pub fn record_full_gradients(&mut self, _gradients: &HashMap<String, Tensor>) -> Result<(), String> {
        Ok(())
    }

    pub fn get_predicted_gradients(&mut self) -> Result<HashMap<String, Tensor>, String> {
        Ok(HashMap::new())
    }

    pub fn end_step(&mut self, _loss: f32) -> Result<(), String> {
        Ok(())
    }

    pub fn current_phase(&self) -> DeterministicPhase {
        if self.current_step <= self.config.warmup_steps {
            DeterministicPhase::Warmup
        } else {
            DeterministicPhase::Full
        }
    }

    pub fn get_stats(&self) -> Stats {
        Stats {
            total_steps: self.current_step,
            full_steps: self.current_step,
            predict_steps: 0,
            speedup: 1.0,
        }
    }
}
EOF
fi

PEFT_REF="${PEFT_RS_REF:-main}"
QLORA_REF="${QLORA_RS_REF:-main}"
UNSLOTH_REF="${UNSLOTH_RS_REF:-main}"

# Check if PR description contains dependency overrides
# Format: `peft-rs: branch-name` or `qlora-rs: v1.0.0` or `unsloth-rs: commit-sha`
if [ -n "$PR_BODY" ]; then
    echo "Checking PR description for dependency overrides..."
    
    # Extract peft-rs ref if specified
    if echo "$PR_BODY" | grep -q "peft-rs:"; then
        PEFT_REF=$(echo "$PR_BODY" | grep -oP "peft-rs:\s*\K\S+" | head -1)
        echo "  Found peft-rs override: $PEFT_REF"
    fi
    
    # Extract qlora-rs ref if specified
    if echo "$PR_BODY" | grep -q "qlora-rs:"; then
        QLORA_REF=$(echo "$PR_BODY" | grep -oP "qlora-rs:\s*\K\S+" | head -1)
        echo "  Found qlora-rs override: $QLORA_REF"
    fi
    
    # Extract unsloth-rs ref if specified
    if echo "$PR_BODY" | grep -q "unsloth-rs:"; then
        UNSLOTH_REF=$(echo "$PR_BODY" | grep -oP "unsloth-rs:\s*\K\S+" | head -1)
        echo "  Found unsloth-rs override: $UNSLOTH_REF"
    fi
fi

echo ""
echo "Patching Cargo.toml with sister project refs:"
echo "  peft-rs:    $PEFT_REF"
echo "  qlora-rs:   $QLORA_REF"
echo "  unsloth-rs: $UNSLOTH_REF"
echo ""

# Create a temporary sed script
cat > /tmp/patch.sed <<EOF
s|git = "https://github.com/tzervas/peft-rs", branch = "[^"]*"|git = "https://github.com/tzervas/peft-rs", branch = "$PEFT_REF"|g
s|git = "https://github.com/tzervas/qlora-rs", branch = "[^"]*"|git = "https://github.com/tzervas/qlora-rs", branch = "$QLORA_REF"|g
s|git = "https://github.com/tzervas/unsloth-rs", branch = "[^"]*"|git = "https://github.com/tzervas/unsloth-rs", branch = "$UNSLOTH_REF"|g
EOF

# Apply patches to Cargo.toml
sed -i -f /tmp/patch.sed Cargo.toml

echo "Cargo.toml patched successfully!"
echo ""
echo "Sister project dependencies in Cargo.toml:"
grep -A 1 "github.com/tzervas/peft-rs\|github.com/tzervas/qlora-rs\|github.com/tzervas/unsloth-rs" Cargo.toml || true
