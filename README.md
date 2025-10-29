# SPA: Self-Play with World Model for LLM Agents

<p align="center">
  <a href="https://spa-ai.github.io"><img src="https://img.shields.io/badge/Homepage-orange?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2510.15047"><img src="https://img.shields.io/badge/Paper-red?style=for-the-badge"></a>
  <a href="#post"><img src="https://img.shields.io/badge/Post-green?style=for-the-badge"></a>
  <a href="#experiment-log"><img src="https://img.shields.io/badge/Experiment%20Log-purple?style=for-the-badge"></a>
</p>

SPA (Self-Play Agent) is a reinforcement learning recipe for training Large Language Model (LLM) agents in **out-of-distribution (OOD) environments**. By equipping agents with an **internal world model** through self-play supervised finetuning (SFT), SPA enables better grounding, broader exploration, and more reliable generalization.

---

## Overview

LLM agents often struggle when deployed in environments that differ from their pre-training distribution. Standard reinforcement learning tends to overfit to narrow solution paths, improving **Pass@1** slightly but causing **Pass@k** to degrade. This reflects brittle exploration and weak generalization.

SPA addresses this by introducing a **world model** with two key components:

* **State Representation**: structured abstractions (e.g., symbolic coordinates in Sokoban) that lower perplexity and make spatial relations explicit.
* **Transition Modeling**: predicting next states during self-play, enabling the agent to internalize environment dynamics before policy optimization.

This initialization makes subsequent PPO training more stable and effective.

---

## Key Results

SPA significantly improves performance across challenging environments:

* **Sokoban**: Pass@1 success rate from **25.6% → 59.8%**
* **FrozenLake**: Pass@1 success rate from **22.1% → 70.9%**
* **Sudoku**: Pass@1 success rate from **0.0% → 59.6%**

These improvements are consistent across different LLM families, including **Qwen** and **LLaMA** models.

---

## Framework

SPA training consists of three stages:

1. **Data Generation**: Collect self-play trajectories with `<observation>` and `<prediction>` states.
2. **Supervised Finetuning (SFT)**: Train the agent to predict next states and actions.
3. **PPO Optimization**: Reinforce policies initialized with the learned world model.

This exploration-before-exploitation process allows agents to first **learn environment rules**, then optimize for rewards.

---

## Repository Setup

Clone **RAGEN** and place SPA inside:

```bash
git clone git@github.com:RAGEN-AI/RAGEN.git
cd RAGEN
git clone git@github.com:shiqichen17/SPA.git
```

---

## Environment Setup

From the RAGEN root directory:

```bash
bash scripts/setup_ragen.sh
pip uninstall -y torch torchvision torchaudio && pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y vllm flash-attn flash_attn
pip install vllm==0.8.5.post1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
python -c "import torch; import flash_attn; import vllm; print('✅ All modules loaded successfully.')"
```

> **Note**: Use the versions above exactly to avoid runtime errors.

---

## Quick Start

From the SPA directory:

```bash
cd SPA
bash run_spa.sh <CONFIG_NAME> [CKPT] [GENERATE_DATA]
```

**Arguments:**

* `CONFIG_NAME` (required): Environment config - `_2_sokoban`, `_10_sudoku`, or `_3_frozen_lake`
* `CKPT` (optional, default: `last`): Checkpoint to use (`last` for latest, or step number like `1000`)
* `GENERATE_DATA` (optional, default: `False`): Set to `True` to run full pipeline, `False` for PPO only

**Examples:**

```bash
# Full pipeline (generate data → SFT → PPO)
bash run_spa.sh _2_sokoban last True

# PPO training only with existing checkpoint
bash run_spa.sh _2_sokoban last False

# Use specific checkpoint step
bash run_spa.sh _10_sudoku 2000 False
```

This script runs the **full pipeline** (when `GENERATE_DATA=True`):

* Generate self-play training data
* Perform SFT world-model training
* Run PPO policy optimization

---

## Pretrained Models and Datasets

We provide pretrained models and training datasets for all three environments on Hugging Face:

| Environment | 📊 SFT Training Data | 🤖 Model (after self-play finetuning) |
|------------|------------|------------|
| **Sokoban** | [SPA-sokoban-data](https://huggingface.co/datasets/tyzhu/SPA-sokoban-data) | [SPA-sokoban-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct) |
| **FrozenLake** | [SPA-frozenlake-data](https://huggingface.co/datasets/tyzhu/SPA-frozenlake-data) | [SPA-frozenlake-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-frozenlake-qwen2.5-1.5b-instruct) |
| **Sudoku** | [SPA-sudoku-data](https://huggingface.co/datasets/tyzhu/SPA-sudoku-data) | [SPA-sudoku-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-sudoku-qwen2.5-1.5b-instruct) |

These resources allow you to:
- **Use the pretrained models** directly for inference or further finetuning
- **Reproduce the SFT stage** using the provided training data
- **Skip data generation** and start from the SFT or PPO stages

> **Note**: The FrozenLake and Sudoku datasets include trajectory filtering to remove trajectories not following the format, while the Sokoban dataset contains unfiltered raw trajectories from self-play data generation.

---

## Supported Environments

SPA supports a variety of environments integrated through RAGEN:

* **Sokoban** (grid-based spatial puzzles)
* **FrozenLake** (navigation under stochastic transitions)
* **Sudoku** (4×4 logical puzzles)

---

## Example World Model Trace

For Sokoban, SPA generates structured reasoning traces:

```
<think>
<observation>
######
#___O#
#__X_#
###P_#
###__#
######
Player (P) at (3,3); box (X) at (2,3); goal at (1,4).
</observation>
<prediction>
######
#___O#
#____#
###X_#
###P_#
######
</prediction>
</think>
<answer>Up</answer>
```

This explicit **observation → prediction → action** format grounds decision-making in environment dynamics.

---

## Configuration

Key configuration files are located in `config/`:

* `base.yaml`: core training settings
* `_2_sokoban.yaml`, `_3_frozen_lake.yaml`, etc.: environment-specific configs
* `envs.yaml`: environment registry

Important parameters:

* `model_path`: base model (e.g., `Qwen/Qwen2.5-1.5B-Instruct`)
* `trainer.total_training_steps`: PPO steps
* `agent_proxy.max_turn`: max turns per episode
* `es_manager.train.env_groups`: number of environment groups

---

## Citation

If you use SPA in your work, please cite:

```bibtex
@misc{chen2025spa,
      title={Internalizing World Models via Self-Play Finetuning for Agentic RL}, 
      author={Shiqi Chen and Tongyao Zhu and Zian Wang and Jinghan Zhang and Kangrui Wang and Siyang Gao and Teng Xiao and Yee Whye Teh and Junxian He and Manling Li},
      year={2025},
      eprint={2510.15047},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.15047}, 
}

```

---

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

---

## Acknowledgments

SPA is built on top of the [RAGEN](https://github.com/RAGEN-AI/RAGEN) framework, extending it with explicit world-model pretraining for improved RL scalability.
