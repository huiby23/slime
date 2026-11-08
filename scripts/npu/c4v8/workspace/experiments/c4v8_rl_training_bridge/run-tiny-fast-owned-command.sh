#!/bin/bash
# Called only by run-owned-probe.py after its all-eight idle gate.
set -eo pipefail
export HBY_C4V8_ALLOW_UNVALIDATED_FUSION=1 HBY_C4V8_PROBE_NPU_RESERVED=1
export ASCEND_RT_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1
exec /bin/bash /root/hby/experiments/c4v8_rl_training_bridge/run-probe-megatron-tiny.sh \
 --execute-npu --physical-device 1 --core-module fast_attention --sequence-length 128 \
 --policy /root/hby/experiments/c4v8_rl_training_bridge/role-probe-policy-fast-v2.json \
 --output-dir "$1"
