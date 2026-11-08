# Ascend baseline in /root/hby

The user fork remains at main commit
`540ea919e45b8b929198abce008a103be9cfc779`, with local compatibility work on
`codex/ascend-environment`. The original main branch and origin are preserved.

This setup uses the pinned dependencies from `docker/npu_patch/README.md`:
SGLang `dce8b0606c06d3a191a24c7b8cbe8e238ab316c9`,
Megatron-LM `3714d81d418c9f1bca4594fc35f9e8289f652862`,
MindSpeed `fc63de5c48426dd019c3b3f39e65f5bdf56e4086`,
mbridge `89eb10887887bc74853f89a4de258c0702932a1c`, and
Megatron-Bridge `35b4ebfc486fb15dcc0273ceea804c3606be948a`.
The dependency sources and applied upstream patches are under `/root/hby/deps`;
their source provenance is in `/root/hby/setup/dependency-sources.json`.
The NPU guide originally targets slime v0.2.2; the local integration preserves
this fork's newer main and adapts its scheduling and bootstrap code.

Run the BF16 installation smoke test after the environment is installed:

```bash
bash /root/hby/slime/scripts/npu/run_smoke_qwen3.sh
```

The default uses physical NPUs 1 and 2, one Megatron training worker and one
SGLang rollout engine, Qwen3-0.6B, two rollouts with two samples each, and one
optimizer step per rollout. Thinking is disabled and each response is limited
to 64 tokens. The independent prompts in
`/root/hby/data/smoke-open-prompts.jsonl` request varied two-sentence stories
at temperature 1.0 so synthetic sample-index rewards do not cancel on
identical arithmetic answers. Set `SMOKE_NPU_DEVICES=3,4` to choose another healthy pair.
Additional slime CLI options can follow `--`.

The launcher creates a private local Ray cluster with its own namespace,
temporary directory, and 2 GiB object store. It shuts down only that cluster.
It does not stop existing Ray services or kill unrelated Python processes.

The smoke reward alternates by sample index so the infrastructure can produce
nonzero GRPO advantages without an external judge. This is a synthetic
installation test, not a math accuracy, convergence, or performance benchmark.
It calls this repository's `train.main`: RayTrainGroup, RolloutManager,
Megatron optimizer, and full-weight synchronization all run normally.
Weight synchronization uses full Hugging Face safetensors on disk rather than
a device-specific transport.

Outputs are in the printed `SMOKE_OUTPUT` directory under `/root/hby/logs`.
The runner writes `success.json` only after training returns and verifies:
two rollout artifacts, two finite positive optimizer gradient norms, and three
full weight versions (initial synchronization plus two updated policies).
The JSON records those norms, versions, and package versions.

This baseline does not implement NVIDIA NVFP4 or validate W4A8C8V16 RL.
CUDA-specific low precision kernels require a separate Ascend implementation.

The optional `examples/coding_agent_rl` Sunabako sandbox currently supports
Linux x86_64 only. Its dependency is guarded by a platform marker in the main
requirements; the example cannot run on this ARM64 host. Core GRPO does not
import Sunabako. The sandbox implementation and its isolation settings are
unchanged.

The pinned MindSpeed requirement `numpy<=1.26.0` conflicts with the ARM64
Decord2 dependency pulled by SGLang (`numpy>=1.26.4`). The local dependency
source allows only the NumPy 1.26 patch series:
`numpy>=1.26.0,<1.27`; the environment pins 1.26.4. No implementation code
contains a NumPy 1.26.0-specific version check. This compatibility adjustment
must be validated by NPU computation, attention gradients, and the RL smoke;
it does not claim upstream certification. The exact dependency patch is
`scripts/npu/mindspeed-numpy.patch`.

The smoke launcher passes `--cxx_abi=1` when sourcing ATB, matching the installed
PyTorch 2.8 build and avoiding the vendor script's failed auto-detection under
`set -e`.

The smoke writes a private SGLang server-group configuration that binds its
HTTP server to `127.0.0.1`; its router and router metrics also bind loopback.
Production multi-node defaults are unchanged. Cached engine URLs follow an
explicit server-group host override so cleanup uses the registered address.

The official DeepEP NPU wheel 1.0.0+ba46a300.cann.8.5.0.b232 places `deep_ep_cpp` inside its `deep_ep`
package while importing it as a top-level module. In this environment the
wheel's own extension directory is exposed with an environment-local
`site-packages/ascend_deepep_extension.pth` containing the absolute path
`/root/hby/envs/slime/lib/python3.11/site-packages/deep_ep`. Reproduce this
after installing the official wheel; no system libraries need replacement.
Only the `deep_ep.Config` / `deep_ep.Buffer` import was checked; this does not
validate expert-parallel communication or a MoE training job.

Pinned Megatron's `has_modelopt_state` helper inspects checkpoint files
without needing ModelOpt itself. The local
`scripts/npu/megatron-modelopt-lazy.patch` moves the optional ModelOpt imports
from that helper module into `load_modelopt_state`, the actual restore path.
This avoids a ModelOpt 0.37.0 partial-import failure after the MindSpeed
bootstrap during ordinary BF16 HF model construction. Real ModelOpt restore
keeps its original imports and behavior; this does not implement NVIDIA
NVFP4 on Ascend.

The installed ARM64 baseline also pins PyArrow 20.0.0, Datasets 3.6.0 and
Pandas 2.2.3 alongside NumPy 1.26.4. PyArrow 26.0.1 required NumPy 2 at import
time on this host despite a satisfiable package metadata check.

After sourcing CANN 8.5, vendor `op-compile-tool` metadata declares the Python
standard-library modules `getopt`, `inspect` and `multiprocessing` as external
distributions. Their standard-library imports work; do not install third-party
packages with those names to silence the vendor metadata warnings. The conda
environment's own dependency check passes independently of that metadata.

Check conda-environment dependencies without CANN's injected Python path:

```bash
env -u PYTHONPATH /root/hby/envs/slime/bin/python -m pip check
```

The vendor metadata/import evidence is
`/root/hby/logs/cann-python-metadata-validation.json`. Resolved environment
versions and complete dependency diffs are saved under `/root/hby/logs`.

The local `scripts/npu/megatron-optimizer-step.patch` canonicalizes scalar
optimizer group step tensors to their numeric values before checking equality.
Equal-valued NPU tensors no longer compare as different Python objects.
Mismatched steps still fail validation, and checkpoint group steps remain
Python integers. CPU tensor fixtures validate equal steps, unequal rejection,
and the saved metadata format.

A smoke-only training hook records the actual MindSpeed FlashAttention module
and traverses its output autograd graph once per rollout. The
`attention_*.json` files and success JSON distinguish an observed
`NpuFusionAttentionBackward` node from module configuration alone. It does not
change attention flags or return a modified forward output. The legacy
`use_flash_attn=False` mask-generation warning alone does not determine the TE
attention backend.

Validated run (server UTC 2026-11-04):
`/root/hby/logs/slime-npu-smoke-20261104-180700-08903b8d/success.json`.

Qwen3-0.6B BF16 GRPO completed two rollouts and two real Megatron optimizer
steps on physical NPUs 1 and 2. Gradient norms were 35.58595207875662 and
28.308171966724665. Both model/optimizer checkpoints saved successfully, and
full policy weights were synchronized into SGLang as versions 1, 2 and 3.
Each rollout recorded the actual MindSpeed FlashAttention module and
`NpuFusionAttentionBackward0` in its output graph; both
`fused_attention_backward_in_graph` values are true.

The canonical log is `/root/hby/logs/slime-rl-smoke.log`; its
`slime-rl-smoke.exit` marker is 0. Post-run checks found this run's recorded
driver/worker PIDs gone and its loopback serving/router/metrics ports
15000, 3600 and 5675 closed.

This validates the small BF16 installation path with synthetic rewards. It is
not a Qwen3-30B-A3B training run, NVFP4/W4A8C8V16 validation, or a quality or
throughput benchmark.

## C4V8 forward simulation archive

See [C4V8 source, precision policy, and validation scope](c4v8/README.md) for the `fa/30B-W16A16C4V8` development snapshot. The experiment is paused by request; the archive is not a completed 100-step training result.
