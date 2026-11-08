#!/bin/bash
set -eo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_WARM_POOL=0
export TE_PARALLEL_COMPILER=2
export MAX_COMPILE_CORE_NUMBER=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=64
export SLIME_SGLANG_NPU_ALLOC_CONF="expandable_segments:False"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-128}"
export SLIME_NPU_TRIM_UNUSED_CPU_HEAP="${SLIME_NPU_TRIM_UNUSED_CPU_HEAP:-1}"
WORKSPACE=${SLIME_WORKSPACE:-/root/hby}
source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORKSPACE/envs/slime"
source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"
source "$WORKSPACE/Ascend/nnal/atb/set_env.sh" --cxx_abi=1
export SLIME_ACCELERATOR=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP="${HCCL_IF_IP:?Set HCCL_IF_IP to the NPU host communication address}"
export HCCL_IF_BASE_PORT=32100
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MAX_JOBS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export XDG_CACHE_HOME="$WORKSPACE/cache"
export TORCH_HOME="$WORKSPACE/cache/torch"
export TRITON_CACHE_DIR="$WORKSPACE/cache/triton"
export TORCH_EXTENSIONS_DIR="$WORKSPACE/cache/torch-extensions"
export TMPDIR="$WORKSPACE/tmp"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PYTHONPATH="$WORKSPACE/slime:$WORKSPACE/deps/Megatron-LM:$WORKSPACE/deps/MindSpeed:$WORKSPACE/deps/mbridge:$WORKSPACE/deps/Megatron-Bridge/src:${PYTHONPATH:-}"
if [ -n "${SLIME_NPU_REPLAY0_BACKUP:-}" ]; then
    export PYTHONPATH="$WORKSPACE/setup:$PYTHONPATH"
fi
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
cd "$WORKSPACE/slime"
exec python scripts/npu/qwen30b_dapo.py "$@"
