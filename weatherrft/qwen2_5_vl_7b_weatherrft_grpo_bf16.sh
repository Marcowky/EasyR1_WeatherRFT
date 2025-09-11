#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export WANDB_BASE_URL="https://api.bandw.top"
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export CUDA_VISIBLE_DEVICES=1,2,3,4
NUMA_NODE=0

DATE=$(date +%Y%m%d_%H%M%S)
GPU_NUMS=4
MODEL_PATH=/home/kaiyu/Model/Qwen/Qwen2.5-VL-7B-Instruct

PROJECT_NAME=weatherrft
EXPERIMENT_NAME=qwen2_5_7b_weatherrft_grpo_500hpa_situation

CONFIG_PATH=weatherrft/config.yaml

TRAIN_FILE=/home/kaiyu/Project/ICASSP_weatherrft/WeatherRFT/data/dataset/WeatherCQ/EasyR1/WeatherCQ_dataset_deepseek_v3_en/split/train_500hpa_situation.json
VAL_FILE=/home/kaiyu/Project/ICASSP_weatherrft/WeatherRFT/data/dataset/WeatherCQ/EasyR1/WeatherCQ_dataset_deepseek_v3_en/val_all.json

# REWARD_PATH=./weatherrft/weatherrft.py
# REWARD_PATH=./weatherrft/weatherrft_with_logic_reward.py
REWARD_PATH=./weatherrft/weatherrft_with_logic_reward_val_acc.py

# FORMAT_PROMPT=./weatherrft/weatherrft.jinja
FORMAT_PROMPT=./weatherrft/weatherrft_seperate_choice_prompt.jinja

SAVE_CHECKPOINT_PATH=./checkpoints/${PROJECT_NAME}/${DATE}-${EXPERIMENT_NAME}

ROLLOUT_NUM=5

# 复制一份配置文件、当前脚本、当前 reward、当前 prompt 到保存路径
mkdir -p ${SAVE_CHECKPOINT_PATH}
cp ${CONFIG_PATH} ${SAVE_CHECKPOINT_PATH}/config.yaml
cp $0 ${SAVE_CHECKPOINT_PATH}/run.sh
cp ${REWARD_PATH} ${SAVE_CHECKPOINT_PATH}/reward.py
cp ${FORMAT_PROMPT} ${SAVE_CHECKPOINT_PATH}/prompt.jinja

# 这里可选是否绑定 numa 节点，注意，需要注释掉之前的 P2P 设置
python3 -m verl.trainer.main \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    config=${CONFIG_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=${ROLLOUT_NUM} \
    worker.reward.reward_function=${REWARD_PATH}:compute_score \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${GPU_NUMS} \
    trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.format_prompt=${FORMAT_PROMPT} \
    | tee ${SAVE_CHECKPOINT_PATH}/train.log