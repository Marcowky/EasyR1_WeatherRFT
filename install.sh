PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip config set global.index-url "${PIP_INDEX}"
pip config set global.extra-index-url "${PIP_INDEX}"
python -m pip install --upgrade pip

pip uninstall -y torch torchvision torchaudio pytorch-quantization pytorch-triton torch-tensorrt transformer-engine flash-attn apex megatron-core xgboost opencv grpcio

pip install "vllm==0.9.1" "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" tensordict torchdata "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" "grpcio>=1.62.1" "optree>=0.13.0" pandas ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb liger-kernel mathruler pytest yapf py-spy pyext pre-commit ruff

ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl" && wget -nv -P . "${URL}" && pip install --no-cache-dir "./$(basename ${URL})"

pip config unset global.index-url
pip config unset global.extra-index-url