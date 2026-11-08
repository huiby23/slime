#!/bin/bash
set -eo pipefail
WORKSPACE=${SLIME_WORKSPACE:-/root/hby}
source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORKSPACE/envs/slime"
if [[ -f "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh" ]]; then
    source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"
fi
if [[ -f "$WORKSPACE/Ascend/nnal/atb/set_env.sh" ]]; then
    source "$WORKSPACE/Ascend/nnal/atb/set_env.sh" --cxx_abi=1
fi
set -u
export SLIME_ACCELERATOR=npu
export ASCEND_RT_VISIBLE_DEVICES=${SMOKE_NPU_DEVICES:-1,2}
export SLIME_HOST_IP=127.0.0.1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HF_HOME="$WORKSPACE/.cache/huggingface"
export XDG_CACHE_HOME="$WORKSPACE/cache"
export TORCH_HOME="$WORKSPACE/cache/torch"
export TRITON_CACHE_DIR="$WORKSPACE/cache/triton"
export TORCH_EXTENSIONS_DIR="$WORKSPACE/cache/torch-extensions"
export TMPDIR="$WORKSPACE/tmp"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"
export PYTHONPATH="$WORKSPACE/slime:$WORKSPACE/deps/Megatron-LM:$WORKSPACE/deps/MindSpeed:$WORKSPACE/deps/mbridge:$WORKSPACE/deps/Megatron-Bridge/src:${PYTHONPATH:-}"
cd "$WORKSPACE/slime"
exec python scripts/npu/smoke_qwen3.py "$@"
