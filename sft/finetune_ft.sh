# NOTE only tested with 1 GPU
set -x
export TRANSFORMERS_CACHE='/ssddata/shiqi/model_hub/hub'
export HF_HOME='/ssddata/shiqi/model_hub/hub'


env_type=$1
nproc_per_node=$2
save_path=$3
data_path=$4
model=$5

shift 5


if [ ! -d $save_path ]; then
    mkdir -p $save_path
fi



export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Sanity check: ensure nproc_per_node does not exceed visible GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    num_visible_gpus=${#gpu_array[@]}
else
    # Fallback to nvidia-smi if CUDA_VISIBLE_DEVICES is not set
    if command -v nvidia-smi >/dev/null 2>&1; then
        num_visible_gpus=$(nvidia-smi --list-gpus | wc -l)
    else
        num_visible_gpus=0
    fi
fi

if [ "$nproc_per_node" -gt "$num_visible_gpus" ]; then
    echo "Error: Requested nproc_per_node=$nproc_per_node but only $num_visible_gpus GPUs are visible (CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES')." >&2
    echo "Please reduce nproc_per_node or expose more GPUs." >&2
    exit 1
fi


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
 -m sft.spa_sft_trainer \
    data.train_files=$data_path/wm_train.parquet \
    data.val_files=$data_path/wm_val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    optim.lr=1e-4 \
    data.train_batch_size=16 \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-$model \
    trainer.default_local_dir=$save_path \
    trainer.experiment_name=test_zpy_${env_type}-sft-qwen-2.5-3b-base \
    trainer.logger=['console'] \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null \
    +trainer.max_ckpt_to_keep=2  \
    model.target_modules=all-linear \
     trainer.project_name=gsm8k-sft \
    model.enable_gradient_checkpointing=False $@ \
    2>&1 | tee  $save_path/train.log

    
