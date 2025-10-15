#!/bin/bash

export PYTHONHASHSEED=10000

# Script arguments
CONFIG_NAME=$1
ckpt=$2
name=$3

# Validate arguments
if [ -z "$CONFIG_NAME" ] || [ -z "$ckpt" ] || [ -z "$name" ]; then
    echo "Usage: $0 <config_name> <checkpoint_path> <experiment_name>"
    echo "Example: $0 _2_sokoban /path/to/checkpoint experiment_name"
    exit 1
fi

# Create log directory
mkdir -p log

# Training parameters
MODEL="1.5B"
MODE="base"
REWARD=0.0
COS=False
RENDER_MODE="text_with_coordinates"

# Export training variables
export MODEL=$MODEL
export MODE=$MODE
export REWARD=$REWARD
export COS=$COS
export RENDER_MODE=$RENDER_MODE

echo "Starting PPO training with:"
echo "  - Config: $CONFIG_NAME"
echo "  - Checkpoint: $ckpt"
echo "  - Experiment: $name"
echo "  - Model: $MODEL"
echo "  - Mode: $MODE"
echo "  - Render Mode: $RENDER_MODE"
echo "WANDB_ENTITY: $WANDB_ENTITY"

if [ -z "$BASE_DIR" ]; then
    echo "BASE_DIR is not set"
    BASE_DIR=/home/aiops/zhuty/ragen/checkpoints
    echo "BASE_DIR is not set, using default: $BASE_DIR"
fi

# Run training
CUDA_VISIBLE_DEVICES='0,1,2,3' python ../train.py \
    --config-path=SPA/config \
    --config-name $CONFIG_NAME \
    model_path=$ckpt \
    system.CUDA_VISIBLE_DEVICES=\'0,1,2,3\' \
    trainer.n_gpus_per_node=4 \
    trainer.total_training_steps=1000 \
    trainer.experiment_name=${name} \
    trainer.save_freq=100 \
    trainer.default_local_dir=$BASE_DIR/${CONFIG_NAME}/$ckpt \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    +actor_rollout_ref.rollout.tp_size_check=True \
    +algorithm.bi_level_gae=False 

echo "Training completed!"
