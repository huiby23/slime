#!/bin/bash
set -eo pipefail
source /root/hby/activate-slime.sh
export TMPDIR=/root/hby/tmp XDG_CACHE_HOME=/root/hby/cache TORCH_HOME=/root/hby/cache/torch
export TRITON_CACHE_DIR=/root/hby/cache/triton-c4v8
export MAX_COMPILE_CORE_NUMBER=1 MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HCCL_BUFFSIZE=128 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HBY_C4V8_ALLOW_UNVALIDATED_FUSION=1
exec /root/hby/envs/slime/bin/python -u /root/hby/experiments/c4v8_device_core/validate_equal_length_batch_npu.py "$@"
