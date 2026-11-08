#!/usr/bin/env bash
# Environment template only. Inspect paths before sourcing; launches no job.
WORKSPACE="${SLIME_WORKSPACE:-/root/hby}"
source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORKSPACE/envs/slime"
source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"
source "$WORKSPACE/Ascend/nnal/atb/set_env.sh" --cxx_abi=1
export SLIME_ACCELERATOR=npu
export PYTHONPATH="$WORKSPACE/slime:$WORKSPACE/deps/Megatron-LM:$WORKSPACE/deps/MindSpeed:$WORKSPACE/deps/mbridge:$WORKSPACE/deps/Megatron-Bridge/src:${PYTHONPATH:-}"
# Set HCCL_IF_IP in the deployment shell. No host address or credential is included.
