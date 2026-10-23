# NVIDIA NVFP4 training recipe simulation

This implements the default **linear-layer training recipe**, including backward,
for the single-GPU Qwen3-0.6B BF16 GRPO baseline. It is software simulation for
A100, using ordinary PyTorch operations and FP32 GEMMs. It does not require native
FP4 or FP8 instructions.

| GEMM | Left operand | Right operand (transposed by GEMM) |
| --- | --- | --- |
| Forward | NVFP4(X), 1x16 blocks | NVFP4(W), 16x16 blocks |
| Input gradient | NVFP4(G), stochastic rounding, 1x16 blocks | NVFP4(W.T), 16x16 blocks |
| Weight gradient | NVFP4(RHT(G.T)), stochastic rounding, 1x16 blocks | NVFP4(RHT(X.T)), 1x16 blocks |

Every quantized operand has E2M1 values, rounded E4M3 block scales, and one FP32
tensor scale derived from the current amax. Row and column operands come from
the original high precision tensor, with independent amax after rotation where
appropriate. RHT uses TE's fixed sign vector and a shared normalized 16x16
Hadamard matrix. Token padding is retained through RHT and WGRAD.

GEMMs compute FP32 inner products of 16-element FP4 payloads, apply their block
scales, accumulate in FP32, and apply the global scales at the epilogue. This
avoids an extra BF16 rounding of dequantized operands before multiplication.
Gradient rounding uses unbiased software stochastic rounding, with PyTorch's RNG.
The original BF16 parameters and baseline optimizer/master-state precision remain
in use; quantization never overwrites a master parameter.

All four transformer projections per layer and the output projection are covered
(113 linear modules for the 28-layer baseline). Megatron's fused norm/linear
interfaces and its native `ColumnParallelLinear` output interface are supported.
The tied embedding/output weight retains both gradient contributions; SGLang's
shared embedding keeps its original embedding lookup implementation. Attention,
normalization and embedding lookup retain their baseline precision.

The adapter requires dense Qwen3 and TP=PP=CP=EP=1. It fails on unsupported layers
or incomplete projection coverage. Hardware packing, scale swizzles, hardware RNG
bit streams, and exact Tensor Core accumulation order are outside the simulator.
This does not establish bitwise equivalence or Blackwell speed/memory savings.

## Files

- `slime_plugins/nvfp4_sim/recipe.py`: quantizers, RHT, block GEMM, custom backward.
- `slime_plugins/nvfp4_sim/megatron.py`: Slime custom model provider.
- `slime_plugins/nvfp4_sim/sglang_models/qwen3.py`: matching rollout simulation.
- `prepare_baseline.py`: prepare an inspectable baseline-derived run and optional CPU preflight/launch.
- `validate_megatron.py`: CUDA integration check using real Megatron, TE,
  distributed optimizer, two updates, a model checkpoint round-trip and optimizer
  metadata restoration. This small check does not serialize the distributed
  optimizer's separate FP32 parameter/moment checkpoint.

## Use the original baseline

The completed BF16 study ran Slime commit
`0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e` from
`/data2/huibingyu/slime-v040/code/slime`, rather than this working checkout's newer
main branch. The launcher preserves that source and its recorded runtime image,
and overlays only `nvfp4_sim` through `PYTHONPATH`. It also preserves TIS settings.

From the repository root, prepare and validate without allocating a GPU:

```bash
python3 examples/nvfp4_sim/prepare_baseline.py \
  --output outputs/nvfp4-smoke --quick-smoke --preflight
```

This writes `config.json`, `mismatch.yaml`, `manifest.json`, `preflight.log`, and
`configuration-check.json`. The manifest includes the exact Docker command and
baseline/simulator revisions. Quick smoke uses two updates, 2 prompts x 2
responses, 64 response tokens, and 8 validation questions. Its sparse math rewards
are not a convergence or gradient-quality test; use the CUDA validation script
for the optimizer/gradient check.

To launch that smoke run when the chosen GPU is idle:

```bash
python3 examples/nvfp4_sim/prepare_baseline.py \
  --output outputs/nvfp4-smoke --quick-smoke --gpu 7 --run
```

The host launcher and original training supervisor check GPU availability before
initializing CUDA. `--run` also runs CPU preflight first. Use a fresh output
directory after an actual run. For a baseline-sized experiment, omit
`--quick-smoke` and set `--updates 100`; the original rollout/evaluation settings
are retained. Checkpoint interval is one update.

For manual integration, set:

```text
PYTHONPATH=<this checkout>/slime_plugins:<baseline checkout>:/root/Megatron-LM
SGLANG_EXTERNAL_MODEL_PACKAGE=nvfp4_sim.sglang_models
--custom-model-provider-path nvfp4_sim.megatron.model_provider
--sglang-json-model-override-args '{"architectures":["NVFP4SimQwen3ForCausalLM"]}'
```

Use BF16 and leave native `--fp4`/`--fp8` disabled. Keep SGLang prefill/decode CUDA
graphs disabled, as in the recorded baseline. Both the training worker and all
SGLang child processes must inherit the package path and external model setting.

## Wait for an idle GPU

After preparing a two-update quick smoke with `--preflight`, run the host-only
watcher in a persistent background process:

```bash
python3 -u examples/nvfp4_sim/watch_gpu.py \
  --template outputs/nvfp4-sim-preflight \
  --output outputs/nvfp4-gpu-watch-NEW
```

It polls all GPUs every five seconds. A card qualifies only after more than 300
observed seconds with no compute processes, memory usage at most 64 MiB, and 0%
utilization. Activity, failed queries, or long sampling gaps reset the timer.
Before launching, it checks the selected card again; the training supervisor
also checks before initializing CUDA. These checks do not reserve the GPU.

`status.json`, `watch.pid`, and `events.jsonl` record monitoring and launch state.
Each attempt gets its own configuration, launch log, and results directory.
If another process claims the selected GPU during the experiment, the watcher
stops only its own labeled container and waits for a new idle interval. Actual
training failures stop the watcher for inspection. Create `STOP` in the watch
directory to stop monitoring and its active experiment. Use a fresh directory
for a new watcher to preserve earlier evidence.

The scheduler's host-only checks can run without PyTorch:

```bash
python3 -m unittest discover -s tests -p 'test_nvfp4_gpu_watch.py' -v
```

## Validation

The CPU suite compares E2M1/E4M3 quantization and RHT with the NVIDIA reference
installed in the pinned baseline image (Transformer Engine 2.18.0). It also checks
ties, saturation, zero blocks, padding, transposes, stochastic rounding, both
backward GEMMs, FP32 main-grad accumulation, two FP32 master updates, checkpoint
restoration, tied weights, and rollout weight refresh:

```bash
CUDA_VISIBLE_DEVICES= python3 -m pytest \
  tests/test_nvfp4_sim.py tests/test_nvfp4_sim_adapters.py
```

TE's Python-interface CPU tests bypass only the constructor's CUDA availability
guard while explicitly creating CPU parameters; the guard is restored before
forward/backward. These are not native CUDA integration checks.

The separate `validate_megatron.py --output <directory>` must run in the baseline
runtime on an exclusively available CUDA GPU. It writes `validation.json` and a
checkpoint. On A100, the two-layer integration check completed two nonzero-gradient
optimizer updates and a model checkpoint round-trip; all seven recipe counters
were 18 (FPROP, DGRAD, WGRAD, both RHT operands, both stochastic gradient casts).
This validates CUDA execution with real Megatron/TE and a distributed optimizer.

The full Qwen3-0.6B GRPO smoke passed CPU configuration preflight and started on
GPU 7. A different user's process then claimed the same GPU, so our container was
stopped during startup, before its first update. Full-model rollout/training/weight-sync
and convergence validation remain pending; the small CUDA check is not an accuracy
experiment. Availability checks do not reserve a device against other users.
Artifacts from this checkout are under `outputs/nvfp4-cuda-validation`,
`outputs/nvfp4-sim-preflight`, and `outputs/nvfp4-sim-smoke-a100`.

References: [NVIDIA NVFP4 recipe documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html),
[recipe API](https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html),
and the baseline image's
`transformer_engine/pytorch/custom_recipes/quantization_ref_nvfp4.py` reference.
