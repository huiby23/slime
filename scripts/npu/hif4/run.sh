#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export SLIME_FA_CUBE_FORMAT=hif4
export SLIME_FA_TRITON_CACHE="${SLIME_WORKSPACE:-/root/hby}/cache/triton-hif4"
exec bash "$here/../c8v16/workspace/experiments/c8v16_recipe/run.sh" "$@"
