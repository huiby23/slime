#!/bin/bash
# Execute only through run-owned-probe.py with a root-granted all-eight lease.
set -eo pipefail
source /root/hby/activate-slime.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TMPDIR=/root/hby/tmp XDG_CACHE_HOME=/root/hby/cache
export TRITON_CACHE_DIR=/root/hby/cache/triton-c4v8
export MAX_COMPILE_CORE_NUMBER=1 OMP_NUM_THREADS=2 HCCL_BUFFSIZE=128
exec /root/hby/envs/slime/bin/python -u /root/hby/experiments/c4v8_30b_sg_pilot/pilot.py "$@"
