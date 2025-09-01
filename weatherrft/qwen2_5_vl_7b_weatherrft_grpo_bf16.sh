#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export WANDB_BASE_URL="https://api.bandw.top"

MODEL_PATH=/home/kaiyu/Model/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=weatherrft/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16
