# 设置 conda 环境
source /home/kaiyu/miniconda3/etc/profile.d/conda.sh
conda activate vllm_0901

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export CUDA_VISIBLE_DEVICES=0

PORT=50907
# MODEL_PATH=/home/kaiyu/Model/Qwen/Qwen3-32B
MODEL_PATH=/home/kaiyu/Model/openai/gpt-oss-20b/
TENSOR_PARALLEL_SIZE=2

vllm serve ${MODEL_PATH} --port ${PORT}
# vllm serve ${MODEL_PATH} --port ${PORT} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}