#!/bin/bash
export TRITON_CACHE_DIR='/ssddata/model_hub'
export TRANSFORMERS_CACHE='/ssddata/model_hub'
export HF_DATASETS_CACHE='/ssddata/model_hub'
export TRANSFORMERS_CACHE='/ssddata/model_hub'
export ALFWORLD_DATA='/ssddata/shiqi/ETO/eval_agent/data/alfworld'
export VLLM_CONFIG_ROOT='/ssddata/model_hub'
export WANDB_ARTIFACT_DIR='/ssddata/model_hub'
export WANDB_CACHE_DIR='/ssddata/model_hub'
export WANDB_DIR='/ssddata/model_hub'
export RAY_TMPDIR='/ssddata/model_hub'
export TMPDIR='/ssddata/model_hub'
export WANDB_ARTIFACT_DIR='/ssddata/model_hub'
export WANDB_CACHE_DIR='/ssddata/model_hub'
export WANDB_DIR='/ssddata/model_hub'
export WANDB_TMP_DIR='/ssddata/model_hub'
export WANDB_DATA_DIR='/ssddata/model_hub'
export TMPDIR='/ssddata/model_hub'
export TEMP='/ssddata/model_hub'
export TMP='/ssddata/model_hub'
export NLTK_DATA='/ssddata/model_hub'
export PYTHONHASHSEED=10000
export WANDB_ENTITY=1430411375


CONFIG_NAME=$1
# _1_bandit _6_webshop _2_sokoban_base _4_countdown _5metamathqa _3_frozen_lake
ckpt=$2
name=$3
mkdir -p log


LOWC_LIST=(0.2)
HIGHC_LIST=(0.28)
MODE_LIST=('base')
REWARD_LIST=(0.0)
COS_LIST=(False)
MODEL_LIST=("1.5B")
RENDER_MODE_LIST=('text')

# Create GPU usage tracking array for GPU pairs
# GPU pairs: 0,1 and 2,3
declare -A gpu_pair_in_use
for gpu_pair in "6,7"; do
    gpu_pair_in_use[$gpu_pair]=0
    rm -f /tmp/gpu_pair_${gpu_pair}_busy
done

# Function to check if GPU pair is available
check_gpu_pair_available() {
    local gpu_pair=$1
    # Check if GPU pair is in use
    if [ -f /tmp/gpu_pair_${gpu_pair}_busy ]; then
        return 1  # GPU pair is in use
    fi
    return 0  # GPU pair is available
}

# Function to get available GPU pair
get_available_gpu_pair() {
    for gpu_pair in "6,7"; do
        if check_gpu_pair_available $gpu_pair; then
            echo $gpu_pair
            return 0
        fi
    done
    return 1  // No available GPU pair
}

# Create task queue
declare -A task_queue
queue_index=0

for MODEL in "${MODEL_LIST[@]}"; do
    for MODE in "${MODE_LIST[@]}"; do
        for REWARD in "${REWARD_LIST[@]}"; do
            for COS in "${COS_LIST[@]}"; do
                for RENDER_MODE in "${RENDER_MODE_LIST[@]}"; do
                    task_queue[$queue_index]="$LOWC $HIGHC $MODEL $MODE $REWARD $COS $RENDER_MODE"
                    queue_index=$((queue_index + 1))
                done
            done
        done
    done
done

# Current queue position
current_task=0
total_tasks=$queue_index

# Execute tasks concurrently
while [ $current_task -lt $total_tasks ]; do
    # Check if there is an available GPU pair
    GPU_PAIR=$(get_available_gpu_pair)
    if [ $? -eq 0 ]; then
        # Get task parameters from queue
        read MODEL MODE REWARD COS RENDER_MODE<<< "${task_queue[$current_task]}"
        
        echo "Launching task $((current_task + 1))/$total_tasks: MODEL=$MODEL MODE=$MODE REWARD=$REWARD COS=$COS RENDER_MODE=$RENDER_MODE on GPU pair $GPU_PAIR"
        
        export MODEL=$MODEL
        export MODE=$MODE
        export REWARD=$REWARD
        export COS=$COS
        export RENDER_MODE=$RENDER_MODE
        
        # Mark GPU pair as in use
        touch /tmp/gpu_pair_${GPU_PAIR}_busy
        
        # Start training task in the background
        (
            CUDA_VISIBLE_DEVICES='6' python ../train.py \
                --config-name $CONFIG_NAME\
                model_path=$ckpt \
                trainer.n_gpus_per_node=1 \
                 trainer.total_training_steps=1000 trainer.experiment_name=${name}  
                 
                # > "log/${CONFIG_NAME}_${MODEL}_${MODE}_$(date +"%Y-%m-%d_%H-%M-%S")_REWARD_${REWARD}_COS_${COS}_RENDER_MODE_${RENDER_MODE}.log" 2>&1
            # After task completion, mark GPU pair as available
            rm -f /tmp/gpu_pair_${GPU_PAIR}_busy
        ) &
        
        current_task=$((current_task + 1))
        # Add a brief delay to avoid starting tasks simultaneously
        sleep 5
    else
        echo "Waiting for available GPU pair..."
        sleep 5
    fi
done

# Wait for all tasks to complete
wait

# Clean up temporary files
for gpu_pair in "0,1" "2,3"; do
    rm -f /tmp/gpu_pair_${gpu_pair}_busy
done

# Wait for the last batch of tasks to complete
wait
