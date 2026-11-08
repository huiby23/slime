#!/bin/bash
# Used only through the owned bounded controller after its all-eight idle gate.
set -eo pipefail
[[ "$1" == /root/hby/experiments/* && -f "$1" ]] || exit 2
source /root/hby/activate-slime.sh
export TMPDIR=/root/hby/tmp XDG_CACHE_HOME=/root/hby/cache TORCH_HOME=/root/hby/cache/torch
export TRITON_CACHE_DIR=/root/hby/cache/triton-c4v8
export MAX_COMPILE_CORE_NUMBER=1 MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HCCL_BUFFSIZE=128 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HBY_C4V8_ALLOW_UNVALIDATED_FUSION=1
mkdir -p /root/hby/tmp /root/hby/cache/triton-c4v8
exec /root/hby/envs/slime/bin/python "$@"
