#!/usr/bin/env bash
# Prepared C4V8 forward QDQ + STE experiment. No production job is launched by preparation.
# Original Qwen3-30B-A3B, nonthinking, DAPO real rewards, 100 updates, every5 HF snapshots.
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLIME_NPU_PREALLOCATE_ADAM=1 SLIME_NPU_TRIM_UNUSED_CPU_HEAP=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=128 MAX_COMPILE_CORE_NUMBER=1
export HBY_C4V8_CORE_MODULE=fast_attention
export HBY_C4V8_IMPLEMENTATION_ID=fast_attention_fused_forward_ste_staged_decode_v2
export HBY_C4V8_ALLOW_UNVALIDATED_FUSION=1
export HBY_C4V8_POLICY_FILE='/root/hby/experiments/c4v8_rl_training_bridge/prepared-fast-v3/precision-policy.json'
export HBY_C4V8_LAUNCH_GATE="${HBY_C4V8_LAUNCH_GATE:-/root/hby/experiments/c4v8_rl_training_bridge/prepared-fast-v3/launch-gate.pending.json}"
unset SLIME_NPU_REPLAY0_BACKUP SLIME_NPU_REPLAY0_SOURCE_MANIFEST SLIME_NPU_REPLAY0_ALLOW_CURSOR_RESTART
unset SLIME_NPU_REPLAY0_SHA256 SLIME_NPU_REPLAY0_AUDIT
exec /bin/bash '/root/hby/experiments/c4v8_rl_training_bridge/prepared-fast-v3/run-c4v8-runtime.sh' \
  --version c4v8-fast-forward-qdq-ste-base100-every5 --model /root/hby/models/Qwen3-30B-A3B \
  --parent-completed-optimizer-steps 0 --lineage-target-optimizer-steps 100 \
  --rollouts 5 --prompts 5 --samples 16 --minibatch 4 \
  --validation-trajectory-dir /root/hby/evaluations/qwen3-30b-c4v8-fast-forward-qdq-ste-every5 \
  --offline-checkpoint-validation --keep-all-weight-versions --hf-checkpoint-every-steps 5 \
  --data /root/hby/data/dapo-math-17k/train.jsonl \
  --eval-data /root/hby/data/aime-2024/eval.jsonl --eval-name aime2024 \
  --offload-fraction 0.35 --response-length 4096 --sequence-length 8192 \
  --sglang-mem-fraction 0.135 --sglang-context-length 8192 \
  --sglang-max-total-tokens 12288 --sglang-max-running-requests 4 \
  --sglang-chunked-prefill-size 256 "$@" -- \
  --rollout-shuffle --log-probs-chunk-size 1024
