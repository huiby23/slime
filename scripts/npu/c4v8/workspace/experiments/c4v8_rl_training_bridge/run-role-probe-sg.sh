#!/bin/bash
# No device test without root's explicit --execute-after-resource-gate.
set -eo pipefail
source /root/hby/miniconda3/etc/profile.d/conda.sh
conda activate /root/hby/envs/slime
source /root/hby/Ascend/ascend-toolkit/set_env.sh
source /root/hby/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export SLIME_ACCELERATOR=npu
export PYTHONPATH=/root/hby/slime:/root/hby/deps/Megatron-LM:/root/hby/deps/MindSpeed:/root/hby/deps/mbridge:/root/hby/deps/Megatron-Bridge/src:${PYTHONPATH:-}
exec python /root/hby/experiments/c4v8-integration-probes/sg_smoke.py "$@"
