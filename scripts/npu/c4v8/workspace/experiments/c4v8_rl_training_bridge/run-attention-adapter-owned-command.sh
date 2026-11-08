#!/bin/bash
set -eo pipefail
source /root/hby/activate-slime.sh
export TMPDIR=/root/hby/tmp TRITON_CACHE_DIR=/root/hby/cache/triton-c4v8
export MAX_COMPILE_CORE_NUMBER=1 MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HCCL_BUFFSIZE=128
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HBY_C4V8_PROBE_NPU_RESERVED=1 HBY_C4V8_ALLOW_UNVALIDATED_FUSION=1
exec /root/hby/envs/slime/bin/python /root/hby/experiments/c4v8_rl_training_bridge/validate_attention_adapters.py --execute-npu --policy /root/hby/experiments/c4v8_rl_training_bridge/role-probe-policy-fast-v2.json --output "$1"
