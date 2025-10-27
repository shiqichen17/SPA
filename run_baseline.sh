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
CONFIG_NAME=$1
# CONFIG_NAME="_2_sokoban"
# CONFIG_NAME="_10_sudoku"
# ensure that the CONFIG_NAME is set, and it is one of the following: _2_sokoban, _10_sudoku
if [ -z "$CONFIG_NAME" ]; then
    echo "CONFIG_NAME is not set"
    exit 1
fi
if [ "$CONFIG_NAME" != "_2_sokoban" ] && [ "$CONFIG_NAME" != "_10_sudoku" ]; then
    echo "CONFIG_NAME is not one of the following: _2_sokoban, _10_sudoku"
    exit 1
fi


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



# Step 3: PPO Training
echo "Step 3: Starting PPO training..."
if [ "$CONFIG_NAME" == "_2_sokoban" ]; then
    EXPERIMENT_NAME="sokoban-${MODEL}-RENDER_MODE${RENDER_MODE}-baseline"
elif [ "$CONFIG_NAME" == "_3_frozen_lake" ]; then
    EXPERIMENT_NAME="frozen_lake-${MODEL}-RENDER_MODE${RENDER_MODE}-baseline"
elif [ "$CONFIG_NAME" == "_10_sudoku" ]; then
    EXPERIMENT_NAME="sudoku-${MODEL}-RENDER_MODE${RENDER_MODE}-baseline"
fi  

echo "Training configuration:"
echo "  - Config: $CONFIG_NAME"
echo "  - Experiment: $EXPERIMENT_NAME"
echo "  - Model: $MODEL"
echo "  - Render Mode: $RENDER_MODE"

# Launch training
bash ./train_ppo_sfted.sh "$CONFIG_NAME" "Qwen/Qwen2.5-1.5B-Instruct" "$EXPERIMENT_NAME"

echo "Training pipeline completed successfully!"


