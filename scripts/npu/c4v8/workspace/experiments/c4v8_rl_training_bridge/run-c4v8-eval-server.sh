#!/bin/bash
# Called only by eval-c4v8-trajectory.py with an owned new process group and
# stdin JSON. The dedicated shim, not generic sglang.launch_server, installs
# the identical numerical backend in parent and every spawn scheduler.
set -eo pipefail
source /root/hby/activate-slime.sh
export TRITON_CACHE_DIR=/root/hby/cache/triton-c4v8
export SLIME_ACCELERATOR=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCH_WARM_POOL=0 TE_PARALLEL_COMPILER=2 MAX_COMPILE_CORE_NUMBER=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1 SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=64
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export HCCL_BUFFSIZE=128 HCCL_IF_IP="${HCCL_IF_IP:?Set HCCL_IF_IP to the NPU host communication address}" HCCL_IF_BASE_PORT=34350
export HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MAX_JOBS=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset USE_TORCH USE_TF TORCH_DEVICE_BACKEND_AUTOLOAD HBY_C4V8_CPU_SPAWN_PROBE
export PYTHONPATH=/root/hby/experiments/c4v8_rl_training_bridge:/root/hby/experiments/c4v8_device_core:/root/hby/experiments/c4v8_rollout_bridge:${PYTHONPATH:-}
if [[ "${HBY_C4V8_CORE_MODULE:-}" == fast_attention ]]; then
    export PYTHONPATH=/root/hby/experiments/c4v8_fast_attention:/root/hby/experiments/c4v8_fused_nvfp4:/root/hby/experiments/c4v8_fused_qdq:/root/hby/experiments/c4v8_staged_decode:$PYTHONPATH
    [[ "${HBY_C4V8_ALLOW_UNVALIDATED_FUSION:-}" == 1 ]]
fi
exec /root/hby/experiments/c4v8_rollout_bridge/python-c4v8 -m slime.backends.sglang_utils.runtime
