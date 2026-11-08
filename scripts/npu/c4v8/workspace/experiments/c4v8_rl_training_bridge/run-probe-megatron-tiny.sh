#!/bin/bash
# Default is inspect-only. Parent must explicitly reserve an idle NPU before
# passing --execute-npu and HBY_C4V8_PROBE_NPU_RESERVED=<physical card>.
set -eo pipefail
source /root/hby/miniconda3/etc/profile.d/conda.sh
conda activate /root/hby/envs/slime
source /root/hby/Ascend/ascend-toolkit/set_env.sh
source /root/hby/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export SLIME_ACCELERATOR=npu
export MAX_COMPILE_CORE_NUMBER=1
export MAX_JOBS=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HCCL_BUFFSIZE=128
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=/root/hby/cache
export TORCH_HOME=/root/hby/cache/torch
export TORCH_EXTENSIONS_DIR=/root/hby/cache/c4v8-tiny-probe-extensions
export TRITON_CACHE_DIR=/root/hby/cache/triton-c4v8
export TMPDIR=/root/hby/tmp
export PYTHONPATH=/root/hby/experiments/c4v8_rl_training_bridge:/root/hby/experiments/c4v8_device_core:/root/hby/slime:/root/hby/deps/Megatron-LM:/root/hby/deps/MindSpeed:/root/hby/deps/mbridge:/root/hby/deps/Megatron-Bridge/src:${PYTHONPATH:-}
cd /root/hby/experiments/c4v8_rl_training_bridge
exec python probe_megatron_tiny.py "$@"
