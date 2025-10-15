# SPA: Self-Play with World Model for LLM Agents

SPA (Self-Play with World Model) is a reinforcement learning framework that addresses the challenge of training Large Language Models (LLMs) as agents in out-of-distribution (OOD) scenarios. This recipe implements the approach described in our paper, where we equip LLM agents with an internal world model to better align reasoning with environmental dynamics and improve decision-making.

## Overview

Large Language Models (LLMs) as agents often struggle in out-of-distribution (OOD) scenarios. Real-world environments are complex and dynamic, governed by task-specific rules and stochasticity, which makes it difficult for LLMs to ground their internal knowledge in those dynamics. Under such OOD conditions, vanilla RL training often fails to scale; we observe Pass@k–the probability that at least one of k sampled trajectories succeeds–drops markedly across training steps, indicating brittle exploration and limited generalization.

Inspired by model-based reinforcement learning, we hypothesize that equipping LLM agents with an internal world model can better align reasoning with environmental dynamics and improve decision-making. We show how to encode this world model by decomposing it into two components: state representation and transition modeling.

Building on this, we introduce SPA, a simple reinforcement learning framework that cold-starts the policy via a Self-Play supervised finetuning (SFT) stage to learn the world model by interacting with the environment, then uses it to simulate future states prior to policy optimization. This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-based agent training performance.

## Key Results

SPA demonstrates significant performance improvements across three supported environments:

- **Sokoban** (`_2_sokoban`): Success rate improved from 25.6% to 59.8%
- **FrozenLake** (`_3_frozen_lake`): Score improved from 22.1% to 70.9%
- **Sudoku** (`_10_sudoku`): Enhanced performance on 4x4 Sudoku puzzles

All results are achieved using the Qwen2.5-1.5B-Instruct model.

## Framework Architecture

SPA implements a world model approach with three main components:

1. **State Representation**: Learning to encode environment states
2. **Transition Modeling**: Predicting future states after actions
3. **Self-Play SFT**: Cold-starting the policy through supervised learning

The framework consists of three main stages:

1. **Data Generation**: Generate training trajectories with world model predictions
2. **Supervised Fine-tuning (SFT)**: Train the model on the generated data
3. **PPO Training**: Use reinforcement learning to improve the agent's performance

## Directory Structure

```
SPA/
├── config/                    # Configuration files
│   ├── base.yaml             # Main configuration
│   ├── _2_sokoban.yaml       # Sokoban-specific config
│   ├── _3_frozen_lake.yaml   # FrozenLake config
│   ├── _10_sudoku.yaml       # Sudoku config
│   └── envs.yaml             # Environment configurations
├── SPA_agent/                # Core agent components
│   ├── agent_proxy.py        # Main agent proxy for LLM interactions
│   ├── ctx_manager.py        # Context manager for conversation handling
│   ├── es_manager.py         # Environment state manager
│   ├── generate_sft_data.py  # Script to generate SFT training data
│   └── base_llm.py          # Base LLM wrapper
├── sft/                      # Supervised fine-tuning components
│   ├── spa_sft_trainer.py    # SFT trainer implementation
│   ├── spa_sft_dataset.py    # SFT dataset implementation
│   ├── finetune_ft.sh        # SFT training script
│   └── config/
│       └── sft_trainer.yaml  # SFT trainer configuration
├── run_spa.sh                  # Main training pipeline script
├── train_ppo_sfted.sh        # PPO training script
├── run_spa.sh               # Alternative run script
└── README.md                 # This file
```

## Prerequisites

### System Requirements

- **Java Development Kit (JDK)**: Version 21 or higher is required
  - Download from: https://www.oracle.com/java/technologies/downloads/
  - Update the `JAVA_HOME` path in `run_spa.sh` to point to your JDK installation

### Dependencies

Make sure you have all required dependencies installed. The main requirements are:

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
vllm==0.8.2
ray>=2.10
hydra-core
flash-attn==2.7.4.post1
tensordict>=0.8.0,<0.9.0
wandb
gymnasium
gym_sokoban
peft
accelerate
```

### Environment Setup

1. Set up Java environment (required):
```bash
export JAVA_HOME=/path/to/your/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH
```

2. Set up the required environment variables:
```bash
export CUDA_VISIBLE_DEVICES="0,1"  # Adjust based on your GPU setup
export TRANSFORMERS_CACHE='/ssddata/model_hub'
export HF_HOME='/ssddata/model_hub'
export WANDB_ENTITY=your_wandb_entity
```

3. Ensure the RAGEN framework is properly installed and accessible.

## Quick Start

The easiest way to run the complete SPA pipeline is using the main training script:

```bash
# Navigate to the SPA directory
cd SPA

# Full pipeline: Generate data, train SFT, and run PPO
bash run_spa.sh <CONFIG_NAME> [CKPT] [GENERATE_DATA]

# Example: Run complete pipeline for Sokoban
bash run_spa.sh _2_sokoban last True

# Example: Run PPO training only with existing checkpoint
bash run_spa.sh _2_sokoban last False
```

### Command-Line Arguments

- `CONFIG_NAME` (required): Environment configuration name. Must be one of:
  - `_2_sokoban`: Sokoban environment
  - `_10_sudoku`: Sudoku environment  
  - `_3_frozen_lake`: FrozenLake environment
  
- `CKPT` (optional, default: `last`): Checkpoint to use
  - `last`: Use the latest checkpoint found in the checkpoint directory
  - `<step_number>`: Use checkpoint at specific global step (e.g., `1000`)
  
- `GENERATE_DATA` (optional, default: `False`): Whether to generate SFT data and train
  - `True`: Execute the complete pipeline (data generation → SFT → PPO)
  - `False`: Skip data generation and SFT, run PPO training only

### Pipeline Stages

Depending on the `GENERATE_DATA` flag:

**With GENERATE_DATA=True:**
1. Generate SFT data with world model predictions
2. Fine-tune the model on the generated data
3. Train the agent using PPO

**With GENERATE_DATA=False:**
1. Use existing checkpoint for PPO training (skips data generation and SFT)

## Typical Workflows

### Workflow 1: Complete Training from Scratch

When starting a new experiment:

```bash
# Step 1: Generate data, train SFT, and run PPO
bash run_spa.sh _2_sokoban last True

# This will:
# 1. Generate SFT training data with world model predictions
# 2. Fine-tune the base model on this data
# 3. Run PPO training using the fine-tuned checkpoint
```

### Workflow 2: Continue Training with Existing Checkpoint

When you already have a fine-tuned checkpoint:

```bash
# Use existing checkpoint for PPO training
bash run_spa.sh _2_sokoban last False

# This will:
# 1. Find the latest checkpoint in the checkpoint directory
# 2. Run PPO training using that checkpoint
```

### Workflow 3: Use Specific Checkpoint

When you want to use a specific checkpoint step:

```bash
# Use checkpoint at global step 2000
bash run_spa.sh _2_sokoban 2000 False

# This will use: ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/global_step_2000/
```

### Workflow 4: Train Multiple Environments

Train on different environments sequentially:

```bash
# Train Sokoban
bash run_spa.sh _2_sokoban last True

# Train Sudoku
bash run_spa.sh _10_sudoku last True

# Train FrozenLake
bash run_spa.sh _3_frozen_lake last True
```

## Step-by-Step Usage

If you prefer to run each stage manually instead of using `run_spa.sh`:

### 1. Data Generation

Generate training trajectories with world model predictions:

```bash
# Navigate to SPA directory
cd SPA

# Set environment variables
export MODE=add_worldmodel
export MODEL=1.5B
export RENDER_MODE=text_with_coordinates
export CONFIG_NAME=_2_sokoban
export BT_NUM=5
export OUTPUT_DIR=./sftdata/${CONFIG_NAME}-${MODEL}-${RENDER_MODE}
export PENALTY_VALUE=0.0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate SFT data
python -m SPA_agent.generate_sft_data --config-name $CONFIG_NAME
```

### 2. Supervised Fine-tuning

Fine-tune the model on the generated data:

```bash
# Set checkpoint directory
CHECKPOINT_DIR=./sftckpt/checkpoints${CONFIG_NAME}-${MODEL}-${RENDER_MODE}-qwen/
mkdir -p "$CHECKPOINT_DIR"

# Run fine-tuning
bash sft/finetune_ft.sh "$CONFIG_NAME" 4 "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$MODEL"
```

### 3. PPO Training

Train the agent using PPO:

```bash
# Find the latest checkpoint
LATEST_CKPT=$(ls -d "${CHECKPOINT_DIR%/}"/*/ 2>/dev/null | sort -V | tail -n 1)
LATEST_CKPT=${LATEST_CKPT%/}

# Set experiment name
EXPERIMENT_NAME="${CONFIG_NAME}-${MODEL}-RENDER_MODE${RENDER_MODE}-spa-${LATEST_CKPT}"

# Run PPO training
bash train_ppo_sfted.sh "$CONFIG_NAME" "$LATEST_CKPT" "$EXPERIMENT_NAME"
```

## Supported Environments

The `run_spa.sh` script currently supports three environments:

### 1. Sokoban (`_2_sokoban`)
A puzzle game where the player pushes boxes to target locations:
- **SimpleSokoban**: Basic Sokoban with 1 box (6x6 grid)
- **HarderSokoban**: More complex Sokoban with 2 boxes (10x10 grid)
- **Sokoban2Boxes**: Alternative 2-box configuration
- **SokobanDifferentGridVocab**: Different grid vocabulary
- **SokobanDifferentActionVocab**: Different action vocabulary

### 2. Sudoku (`_10_sudoku`)
4x4 Sudoku puzzle solving environment

### 3. FrozenLake (`_3_frozen_lake`)
Navigation puzzle with slippery ice where the agent must reach the goal while avoiding holes

**Note**: While the RAGEN framework supports additional environments (Countdown, MetaMathQA, WebShop), these are not currently validated or configured for use with the `run_spa.sh` script. To use other environments, you would need to update the validation check in `run_spa.sh` and ensure proper configuration files exist.

## Directory Structure and Paths

The `run_spa.sh` script automatically manages the following directory structure:

- **SFT Data**: `./sftdata/${CONFIG_NAME}-${MODEL}-${RENDER_MODE}/`
  - Example: `./sftdata/_2_sokoban-1.5B-text_with_coordinates/`
  
- **SFT Checkpoints**: `./sftckpt/checkpoints${CONFIG_NAME}-${MODEL}-${RENDER_MODE}-qwen/`
  - Example: `./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/`
  - Contains subdirectories for each checkpoint (e.g., `global_step_1000/`)

The script includes automatic checkpoint discovery that finds and uses the latest checkpoint when `CKPT=last` is specified.

## Configuration

### Script Configuration (`run_spa.sh`)

The `run_spa.sh` script sets the following key parameters:

- **MODE**: `add_worldmodel` - Enables world model predictions
- **MODEL**: `1.5B` - Model size (Qwen2.5-1.5B-Instruct)
- **RENDER_MODE**: `text_with_coordinates` - How environment states are rendered
- **BSZ_NUM**: `5` - Batch size number for data generation
- **PENALTY_VALUE**: `0.0` - Penalty value for the reward function

### Environment Variables

The script also configures:
- **Ray Configuration**: Object spilling threshold and backend logging
- **PYTHONHASHSEED**: Set to `10000` for reproducibility
- **JAVA_HOME**: Required for certain environment operations

### Main Configuration (`config/base.yaml`)

Key configuration parameters:

- **Model Settings**:
  - `model_path`: Base model path (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
  - `micro_batch_size_per_gpu`: Batch size per GPU
  - `ppo_mini_batch_size`: PPO mini-batch size

- **Training Settings**:
  - `trainer.total_training_steps`: Total training steps
  - `trainer.n_gpus_per_node`: Number of GPUs per node
  - `trainer.experiment_name`: Experiment name for logging

- **Agent Settings**:
  - `agent_proxy.max_turn`: Maximum number of turns per episode
  - `agent_proxy.max_actions_per_turn`: Maximum actions per turn
  - `agent_proxy.enable_think`: Enable thinking mode

- **Environment Settings**:
  - `es_manager.train.env_groups`: Number of environment groups
  - `es_manager.train.group_size`: Size of each group
  - `es_manager.train.env_configs.tags`: Environment types

### Environment-Specific Configuration

Each of the three supported environments has its own configuration file that extends the base configuration with environment-specific settings:
- `_2_sokoban.yaml`: Sokoban configuration
- `_3_frozen_lake.yaml`: FrozenLake configuration  
- `_10_sudoku.yaml`: Sudoku configuration

## World Model Implementation

The SPA recipe implements a world model approach where:

1. **Observation**: The agent observes the current state with coordinates
2. **Prediction**: The agent predicts the next state after taking actions
3. **Planning**: The agent uses these predictions for planning
4. **Action**: The agent executes the planned actions

The world model is learned through:
- **State Prediction**: Learning to predict future states
- **Action Planning**: Using predictions for action selection
- **Reward Optimization**: Optimizing for task completion

### Example World Model Output

For Sokoban, the agent generates outputs like:

```
<think>
<observation>
######
#_####
#_P###
#_X#_#
#__O_#
######
Player (P) is at (2,2); box (X) is at (3,2); target (O) is at (4,3).
</observation>
1 Down – I push box to (4,2).
2 Left – I step to (3,1).
3 Down – I stand left of box, ready to push it Right onto target.
<prediction>
######
#_####
#__###
#__#_#
#PXO_#
######
</prediction>
</think>
<answer>Down || Left || Down</answer>
```

## Key Components

### Agent Proxy (`SPA_agent/agent_proxy.py`)

The main component that handles LLM interactions:
- **VllmWrapperWg**: Wrapper for VLLM-based generation
- **ApiCallingWrapperWg**: Wrapper for API-based LLM calls
- **RandomActionWrapperWg**: Wrapper for random action generation
- **LLMAgentProxy**: Main proxy that coordinates between context manager and environment state manager

### Context Manager (`SPA_agent/ctx_manager.py`)

Handles conversation context and message formatting:
- Manages conversation history
- Formats messages for LLM input
- Parses LLM responses
- Handles world model predictions

### Environment State Manager (`SPA_agent/es_manager.py`)

Manages environment states and transitions:
- Initializes and resets environments
- Executes actions in environments
- Tracks environment status and metrics
- Handles environment-specific configurations

### SFT Components

- **spa_sft_trainer.py**: Implements FSDP-based SFT training
- **spa_sft_dataset.py**: Handles SFT dataset loading and preprocessing
- **generate_sft_data.py**: Generates training data with world model predictions

## Monitoring and Logging

The training process includes comprehensive logging:

- **Wandb Integration**: Automatic logging to Weights & Biases
- **Console Logging**: Real-time progress updates
- **File Logging**: Detailed logs saved to files
- **Metrics Tracking**: Success rates, rewards, and other metrics

## Validation and Error Handling

The `run_spa.sh` script includes several validation checks:

1. **CONFIG_NAME Validation**: Ensures the provided config name is one of the supported environments
2. **JAVA_HOME Validation**: Verifies that the Java installation exists at the specified path
3. **Checkpoint Directory Validation**: Checks that checkpoint directories exist before training
4. **Automatic Checkpoint Discovery**: Finds the latest checkpoint if none is specified
5. **Error on Missing Checkpoints**: Exits with a clear error message if no checkpoints are found

If validation fails, the script will exit with a descriptive error message.

## Troubleshooting

### Common Issues

1. **CONFIG_NAME Error**: 
   ```
   Error: CONFIG_NAME is not one of the following: _2_sokoban, _10_sudoku, _3_frozen_lake
   ```
   Solution: Use one of the supported environment configurations.

2. **JAVA_HOME Error**:
   ```
   Error: JAVA_HOME: /path/to/jdk does not exist
   ```
   Solution: Update the `JAVA_HOME` path in `run_spa.sh` or set it correctly in your environment.

3. **Checkpoint Not Found**:
   ```
   Error: Checkpoint directory '...' does not exist!
   ```
   Solution: Run with `GENERATE_DATA=True` first to generate data and train the SFT model.

4. **Import Errors**: Make sure the RAGEN framework is properly installed and the Python path includes the RAGEN directory

5. **CUDA Issues**: Ensure proper GPU setup and CUDA_VISIBLE_DEVICES configuration

6. **Memory Issues**: Adjust batch sizes (`BSZ_NUM`) and model configurations based on available GPU memory

7. **Path Issues**: Verify all file paths are correct and accessible

### Debug Mode

Enable debug logging by setting:
```bash
export VERL_SFT_LOGGING_LEVEL=DEBUG
```

### Performance Optimization

- Use appropriate batch sizes for your hardware
- Enable gradient checkpointing for memory efficiency
- Use mixed precision training (bfloat16)
- Optimize sequence parallel settings

## Advanced Usage

### Customizing Training Parameters

To modify training parameters, edit the `run_spa.sh` script:

```bash
# In run_spa.sh, you can modify:
MODE="add_worldmodel"          # Training mode
MODEL="1.5B"                   # Model size
RENDER_MODE="text_with_coordinates"  # State representation
BSZ_NUM=5                      # Batch size for data generation
PENALTY_VALUE=0.0              # Reward penalty value
```

### Custom Environments

To add custom environments:
1. Add environment configuration to `config/envs.yaml`
2. Create a new config file (e.g., `config/_11_myenv.yaml`)
3. Update the validation check in `run_spa.sh` to include your new config name
4. Run: `bash run_spa.sh _11_myenv last True`

### Custom Models

To use different base models:
1. Update the `model_path` in `config/base.yaml`
2. Optionally modify the `MODEL` variable in `run_spa.sh` to reflect the model size
3. Ensure the model is compatible with the framework (supports the same tokenizer interface)

### Customizing Directory Paths

By default, `run_spa.sh` uses:
- SFT data: `./sftdata/${CONFIG_NAME}-${MODEL}-${RENDER_MODE}/`
- Checkpoints: `./sftckpt/checkpoints${CONFIG_NAME}-${MODEL}-${RENDER_MODE}-qwen/`

To use custom paths, modify the `OUTPUT_DIR` and `CHECKPOINT_DIR` variables in `run_spa.sh`.

### Distributed Training

For multi-node training, configure the appropriate distributed settings in the training scripts and ensure proper network configuration between nodes.

## Citation

If you use SPA in your research, please cite our paper:

```bibtex
@article{spa2024,
  title={SPA: Self-Play with World Model for LLM Agents},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgments

This work is part of the RAGEN framework for training agents with reinforcement learning and generative models.