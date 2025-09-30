#!/bin/bash

# =============================================================================
# SPA Training Script
# =============================================================================
# This script sets up the environment and launches training for the SPA model.
# It handles environment configuration, checkpoint discovery, and training execution.

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# Environment Configuration
# =============================================================================

# Ray configuration
export RAY_object_spilling_threshold=0.99
export RAY_BACKEND_LOG_LEVEL=FATAL
# Reproducibility
export PYTHONHASHSEED=10000


export JAVA_HOME=/home/aiops/zhuty/ragen-dev/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH
# ensure that the JAVA_HOME exists
if [ ! -d "$JAVA_HOME" ]; then
    echo "JAVA_HOME: $JAVA_HOME does not exist"
    exit 1
fi

# =============================================================================
# Training Configuration
# =============================================================================

# Training parameters
MODE="add_worldmodel"
MODEL="1.5B"
RENDER_MODE="text_with_coordinates"
BSZ_NUM=5
CONFIG_NAME="_2_sokoban"

# Derived paths
OUTPUT_DIR="./sftdata/${CONFIG_NAME}-${MODEL}-${RENDER_MODE}"
CHECKPOINT_DIR="./sftckpt/checkpoints${CONFIG_NAME}-${MODEL}-${RENDER_MODE}-qwen/"

# Export training variables
export MODE="$MODE"
export MODEL="$MODEL"
export PENALTY_VALUE=0.0
export RENDER_MODE="$RENDER_MODE"
export BT_NUM="$BSZ_NUM"
export CONFIG_NAME="$CONFIG_NAME"
export OUTPUT_DIR="$OUTPUT_DIR"

# =============================================================================
# Validation and Setup
# =============================================================================


# =============================================================================
# Training Pipeline
# =============================================================================

echo "Starting training pipeline..."

# Step 1: Generate SFT data (commented out - uncomment if needed)
echo "Step 1: Generating SFT data..."
# python -m SPA_agent.generate_sft_data --config-name "$CONFIG_NAME"

# Step 2: Fine-tuning (commented out - uncomment if needed)
echo "Step 2: Fine-tuning..."
# bash sft/finetune_ft.sh "$CONFIG_NAME" 4 "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$MODEL"

# Validate checkpoint directory exists
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: Checkpoint directory '$CHECKPOINT_DIR' does not exist!"
    echo "Please ensure the SFT checkpoint has been created first."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Checkpoint Discovery
# =============================================================================

echo "Searching for latest checkpoint in: $CHECKPOINT_DIR"

# Find the latest checkpoint
LATEST_CKPT=$(ls -d "${CHECKPOINT_DIR%/}"/*/ 2>/dev/null | sort -V | tail -n 1)
if [[ -z "$LATEST_CKPT" ]]; then
    echo "Error: No checkpoints found in '$CHECKPOINT_DIR'"
    exit 1
fi

LATEST_CKPT=${LATEST_CKPT%/}
echo "Latest checkpoint found: $LATEST_CKPT"

# Validate checkpoint directory
if [[ ! -d "$LATEST_CKPT" ]]; then
    echo "Error: Latest checkpoint directory '$LATEST_CKPT' is not accessible"
    exit 1
fi


# Step 3: PPO Training
echo "Step 3: Starting PPO training..."
EXPERIMENT_NAME="sokoban-${MODEL}-RENDER_MODE${RENDER_MODE}-spa"

echo "Training configuration:"
echo "  - Config: $CONFIG_NAME"
echo "  - Checkpoint: $LATEST_CKPT"
echo "  - Experiment: $EXPERIMENT_NAME"
echo "  - Model: $MODEL"
echo "  - Render Mode: $RENDER_MODE"

# Launch training
bash ./train_ppo_sfted.sh "$CONFIG_NAME" "$LATEST_CKPT" "$EXPERIMENT_NAME"

echo "Training pipeline completed successfully!"


