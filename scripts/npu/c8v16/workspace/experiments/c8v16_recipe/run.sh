#!/usr/bin/env bash
# Default mode builds the command and manifest. --run selects full training.
set -eo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="${SLIME_WORKSPACE:-/root/hby}"
source "$workspace/miniconda3/etc/profile.d/conda.sh"
conda activate "$workspace/envs/slime"
source "$workspace/Ascend/ascend-toolkit/set_env.sh"
source "$workspace/Ascend/nnal/atb/set_env.sh" --cxx_abi=1
# Older Ray versions import from the current directory before PYTHONPATH.
repo="$(cd "$recipe_dir/../../../../../.." && pwd -P)"
cd "$repo"
export SLIME_ACCELERATOR=npu ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP="${HCCL_IF_IP:?Set HCCL_IF_IP to the NPU host communication address}"
export HCCL_BUFFSIZE=128 HCCL_IF_BASE_PORT=32100 HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
export SLIME_NPU_PREALLOCATE_ADAM=1 SLIME_NPU_TRIM_UNUSED_CPU_HEAP=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SLIME_SGLANG_NPU_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCH_WARM_POOL=0 TE_PARALLEL_COMPILER=2 MAX_COMPILE_CORE_NUMBER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$workspace/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false RAY_USAGE_STATS_ENABLED=0 RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MAX_JOBS=2 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$workspace/cache" TORCH_HOME="$workspace/cache/torch" TRITON_CACHE_DIR="${SLIME_FA_TRITON_CACHE:-$workspace/cache/triton-c8v16}"
export TORCH_EXTENSIONS_DIR="$workspace/cache/torch-extensions" TMPDIR="$workspace/tmp"
export PYTHONPATH="$repo:$workspace/deps/Megatron-LM:$workspace/deps/MindSpeed:$workspace/deps/mbridge:$workspace/deps/Megatron-Bridge/src:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
exec python "$recipe_dir/train_c8v16.py" --workspace "$workspace" "$@"
