# SPA: Self-Play with World Model for LLM Agents

SPA (Self-Play with World Model) is a reinforcement learning framework that addresses the challenge of training Large Language Models (LLMs) as agents in out-of-distribution (OOD) scenarios. This recipe implements the approach described in our paper, where we equip LLM agents with an internal world model to better align reasoning with environmental dynamics and improve decision-making.

## Overview

Large Language Models (LLMs) as agents often struggle in out-of-distribution (OOD) scenarios. Real-world environments are complex and dynamic, governed by task-specific rules and stochasticity, which makes it difficult for LLMs to ground their internal knowledge in those dynamics. Under such OOD conditions, vanilla RL training often fails to scale; we observe Pass@k–the probability that at least one of k sampled trajectories succeeds–drops markedly across training steps, indicating brittle exploration and limited generalization.

Inspired by model-based reinforcement learning, we hypothesize that equipping LLM agents with an internal world model can better align reasoning with environmental dynamics and improve decision-making. We show how to encode this world model by decomposing it into two components: state representation and transition modeling.

Building on this, we introduce SPA, a simple reinforcement learning framework that cold-starts the policy via a Self-Play supervised finetuning (SFT) stage to learn the world model by interacting with the environment, then uses it to simulate future states prior to policy optimization. This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-based agent training performance.

## Key Results

Experiments across diverse environments show that our approach significantly improves performance:

- **Sokoban**: Success rate improved from 25.6% to 59.8%
- **FrozenLake**: Score improved from 22.1% to 70.9%
- **Sudoku**: Enhanced performance on 4x4 Sudoku puzzles

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
│   ├── _4_countdown.yaml     # Countdown config
│   ├── _5_metamathqa.yaml    # MetaMathQA config
│   ├── _6_webshop.yaml       # WebShop config
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

1. Set up the required environment variables:
```bash
export CUDA_VISIBLE_DEVICES="0,1"  # Adjust based on your GPU setup
export TRANSFORMERS_CACHE='/ssddata/model_hub'
export HF_HOME='/ssddata/model_hub'
export WANDB_ENTITY=your_wandb_entity
```

2. Ensure the RAGEN framework is properly installed and accessible.

## Quick Start

The easiest way to run the complete SPA pipeline is using the main training script:

```bash
# Navigate to the SPA directory
cd /ssddata/shiqi/RAGEN/SPA
bash run_spa.sh
```

This will execute the complete pipeline:
1. Generate SFT data with world model predictions
2. Fine-tune the model on the generated data
3. Train the agent using PPO

## Step-by-Step Usage

### 1. Data Generation

Generate training trajectories with world model predictions:

```bash
# Navigate to SPA directory
cd /ssddata/shiqi/RAGEN/SPA

# Set environment variables
export MODE=add_worldmodel
export MODEL=1.5B
export OUTPUT_DIR=./sftdata/sokoban-data
export BT_NUM=5
export CONFIG_NAME=_2_sokoban

# Generate SFT data
python -m SPA_agent.generate_sft_data --config-name $CONFIG_NAME
```

### 2. Supervised Fine-tuning

Fine-tune the model on the generated data:

```bash
cd sft
bash finetune_ft.sh $CONFIG_NAME 4 $CHECKPOINT_DIR $OUTPUT_DIR $MODEL
```

### 3. PPO Training

Train the agent using PPO:

```bash
cd ..
bash train_ppo_sfted.sh "$CONFIG_NAME" "$latest_ckpt" "sokoban-${MODEL}-sft"
```

## Supported Environments

SPA supports multiple environments through the RAGEN framework:

### Sokoban
- **SimpleSokoban**: Basic Sokoban with 1 box (6x6 grid)
- **HarderSokoban**: More complex Sokoban with 2 boxes (10x10 grid)
- **Sokoban2Boxes**: Alternative 2-box configuration
- **SokobanDifferentGridVocab**: Different grid vocabulary
- **SokobanDifferentActionVocab**: Different action vocabulary

### Other Environments
- **FrozenLake**: Navigation puzzle with slippery ice
- **Countdown**: Mathematical equation solving
- **MetaMathQA**: Mathematical reasoning
- **Sudoku**: 4x4 Sudoku puzzle solving
- **WebShop**: Online shopping simulation

## Configuration

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

Each environment has its own configuration file (e.g., `_2_sokoban.yaml`, `_3_frozen_lake.yaml`) that extends the base configuration with environment-specific settings.

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

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the RAGEN framework is properly installed and the Python path includes `/ssddata/shiqi/RAGEN`

2. **CUDA Issues**: Ensure proper GPU setup and CUDA_VISIBLE_DEVICES configuration

3. **Memory Issues**: Adjust batch sizes and model configurations based on available GPU memory

4. **Path Issues**: Verify all file paths are correct and accessible

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

### Custom Environments

To add custom environments, modify `config/envs.yaml` and add your environment configuration following the existing patterns.

### Custom Models

To use different base models, update the `model_path` in `config/base.yaml` and ensure the model is compatible with the framework.

### Distributed Training

For multi-node training, configure the appropriate distributed settings in the training scripts.

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

This work is part of the RAGEN framework for training agents with reinforcement learning and generative models.# SPA
