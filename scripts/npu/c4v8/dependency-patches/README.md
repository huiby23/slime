# Pinned Ascend dependency snapshots

These patches are complete `git diff HEAD --binary` snapshots exported read-only
from the five dependency checkouts used by the C4V8 experiment. They include the
already-applied Slime integration patches, Ascend adaptations, and subsequent
runtime fixes. They are **not** overlays on top of the repository's Docker/NPU
patch series.

Apply exactly one setup method: either these complete snapshots on clean pinned
HEADs, or a separately reconstructed patch series. Do not apply
`docker/npu_patch/*`, `docker/patch/*`, or `scripts/npu/*.patch` in addition to
these snapshots. The SGLang patch already includes the NPU pyproject selection.

`dependency-lock.json` records each exact HEAD, patch SHA-256, byte count,
diff statistics, and the result of `git ls-files --others --exclude-standard`.
All five checkouts had zero non-ignored untracked files at capture. Ignored
build products, caches, installed wheels, credentials, and environment files
are outside these source snapshots.

## Source preparation order

1. Create separate clean checkouts at each HEAD in the lock file. Keep the
   dependency directory names `sglang`, `Megatron-LM`, `MindSpeed`,
   `mbridge`, and `Megatron-Bridge`.
2. Verify the checkout HEAD and each patch SHA-256 against the lock file.
3. In each nonempty-patch checkout, run `git apply --check /path/to/NAME.patch`,
   then `git apply /path/to/NAME.patch`. The empty `mbridge.patch` means that
   checkout needs no source modification; do not invoke git apply on it.
4. Install into the already selected Python 3.11 / Torch 2.8 / torch-npu 2.8 /
   CANN 8.5 environment, using the pinned NPU wheels and requirements recorded
   with the experiment. Install patched SGLang's `python[srt_npu]`, then the
   editable Megatron-LM, MindSpeed, mbridge and Megatron-Bridge sources, and
   finally Slime. Resolve the locked package versions without upgrading the
   pinned stack. Source patches alone do not reproduce compiled vendor wheels.
5. Use the explicit C4V8 role bootstrap and recipe. Their standalone attention
   modules, process-specific backend registration and shared numerical-policy
   identity are stored in the Slime experiment source; these external patches
   do not globally enable C4V8.

## What the snapshots retain

- SGLang: existing Slime and Ascend integration, plus NPU memory-saver cache-clear
  deferral, symmetric idempotent release/resume, and synchronization around
  static-buffer copying and physical unmap. These lifecycle fixes are needed by
  the selected standard rollout offload/reload path.
- Megatron-LM: existing Slime and Ascend compatibility, ModelOpt lazy imports,
  scalar optimizer-step comparison, and the CPU-offload optimizer allocation
  and foreach fixes used by the 30B run.
- MindSpeed: existing Ascend adaptation, NumPy 1.26-series requirement, and
  tensor-parallel metadata when HF loading skips random initialization.
- Megatron-Bridge: existing MindSpeed parameter mapping and Qwen-VL adaptation;
  the latter is retained as part of the exact checkout snapshot, not claimed to
  be exercised by this text-only Qwen3-MoE experiment.
- mbridge: clean pinned source; no patch.

The full snapshots intentionally retain existing, inactive general-purpose
branches (including Megatron's opt-in INT4 weight-QAT code). Their presence does
not mean this experiment quantizes backbone weights. The requested experiment
uses a BF16 backbone and a separately selected C4V8 attention simulator.

No server code was changed, no NPU test was run, and no job was started during
this export. Historical run/test evidence is separate from source reproduction.
