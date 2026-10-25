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

The adapter requires dense Qwen3 and TP=PP=CP=EP=1; data parallelism can use
multiple full replicas. It fails on unsupported layers
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
directory after an actual run. The smoke checkpoint interval is one update.

To validate actual GRPO learning signals, use the two-update training smoke:

```bash
python3 examples/nvfp4_sim/prepare_baseline.py \
  --output outputs/nvfp4-training-preflight --training-smoke --preflight
```

This keeps the baseline's 8 prompts x 8 answers, 1024-token training/eval response
limits, and batch size 64. Only the number of updates and evaluation questions
are reduced to two and eight; evaluation/save intervals are one. The two-hour
wall-time limit accommodates software simulation. No reward shaping, extra loss
term, or sampling filter is introduced.

The original supervisor runs unchanged through `run_validated.py`. A passive
before-step hook writes `optimizer-validation.jsonl`: it observes an entire FP32
optimizer master tensor before/after each real optimizer call, one projection's
FP32 main gradient, the global gradient norm, and all seven recipe counters. It
does not change the optimizer's inputs, RNG, gradients, or return value.

`training-validation.json` requires both updates to have mixed-reward prompt
groups, finite nonzero global/projection gradients, an FP32 master change, and
execution of FPROP/DGRAD/WGRAD, both RHT operands and stochastic gradient casts.
It also requires both checkpoints, completed weight synchronizations and an
evaluation. `successful_training` is false if this evidence is missing, even if
the original supervisor reports a clean exit. A passing smoke is integration
evidence, not a convergence or accuracy result.

## Formal experiment after validation

```bash
python3 examples/nvfp4_sim/prepare_baseline.py \
  --output <fresh-formal-template> --formal --updates 100 --preflight
python3 -u examples/nvfp4_sim/watch_gpu.py \
  --template <fresh-formal-template> --output <fresh-formal-watch> \
  --after-validation <training-smoke-watch-directory>
```

The formal watcher monitors both validation completion and idle GPUs. It launches
only after the smoke watcher exits successfully and all seven validation checks
pass, including both nonzero-gradient optimizer updates. It also verifies the
same baseline commit, runtime image, quantizer configuration, training dimensions,
and unchanged simulator plugin source. Failed/stopped validation prevents launch.

The formal arm starts fresh from the original HF initialization and seed 42, with
100 updates, 8 prompts x 8 responses, 1024-token limits, 512 evaluation questions,
evaluation interval 25 and checkpoint interval 50. A comparison artifact verifies
the baseline CLI options are retained except for quantization integration and
relocated output/configuration paths. The passive optimizer observer remains
enabled. The runtime limit is 96 hours to accommodate software simulation; this
does not change the number of updates. Keep output/cache/checkpoints on `/data2`.

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

After preparing a two-update smoke with `--preflight`, run the host-only
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
For the short smoke, a competing process causes the watcher to stop its own
labeled container and wait again. During a formal experiment, GPU sharing is
recorded and the existing training job keeps running. Another process appearing
alone must not discard hours of training. Actual training failures stop the
watcher for inspection. Create `STOP` in the watch
directory to stop monitoring and its active experiment. Use a fresh directory
for a new watcher to preserve earlier evidence.

`--adopt-running` replaces a dead host monitor while retaining its owned running
container, attempt directory and progress. It refuses a live previous monitor or
a container with a different ownership label.

To resume a formal job, pass `--resume-from <previous-attempt-directory>` with a
fresh watch directory. This requires a complete model/optimizer checkpoint and
the corresponding saved dataset cursor. It removes the HF finetuning and
optimizer/RNG-reset flags, mounts the prior checkpoint read-only, and starts at
the saved rollout plus one. Resume restores training RNG, optimizer, scheduler
and dataset position; a restarted SGLang worker is not guaranteed to reproduce
the same sampling RNG stream bit for bit. The formal save interval remains 50;
progress before the first checkpoint cannot be restored after process loss.

The scheduler's host-only checks can run without PyTorch:

```bash
python3 -m unittest discover -s tests -p 'test_nvfp4_gpu_watch.py' -v
```

## Four-GPU continuation and best/last

`prepare_parallel.py` prepares a continuation on GPUs 0, 1, 6, and 7: Megatron
DP=4 with TP/PP/CP/EP=1, and four independent TP=1 SGLang engines. Global batch
64, 8 prompts x 8 answers, 1024-token limits, the optimizer, and the NVFP4
arithmetic stay unchanged. The checkpoint includes FP32 optimizer state, training
RNG and the dataset cursor. The distributed optimizer checkpoint is resharded by
Megatron when loading onto four replicas. Sampling RNG streams can change after
restarting engines or changing replica count.

The rollout adapter can cache deterministic weight casts between updates. Both
the baseline's regular and flattened-bucket update APIs call `model.load_weights`,
which invalidates every cache; tensor identity/version changes also invalidate a
cast. Activation casts and all NVFP4 GEMM arithmetic are unchanged. The cache is
optional and is not intended for an external direct-copy loader that bypasses
`model.load_weights`.

`train_best_last.py` runs the pinned Slime driver and replaces only its save
operation. Every optimizer update writes the native model, optimizer, RNG and
dataset cursor to a staging directory, then atomically publishes `checkpoints/last`.
The evaluation log hook publishes `checkpoints/best` when the 512-question GSM8K
score improves. Only snapshots referenced by best or last remain; both names may
point to the same snapshot. A partial write never replaces the previous last.
Best becomes available after the first evaluation of a saved trained checkpoint.
This continuation enables native TensorBoard logging.

`checkpoint_live.py` can safely export the existing single-GPU job before this
migration. It pauses the recorded CPU driver during an in-flight training RPC,
waits for that RPC to finish, and saves the matching model and source boundary.
On failure it resumes the original driver. `--hold-driver` retains the pause
only after the checkpoint is complete, allowing the launch controller to migrate
without losing an update. The existing GPU actors are not killed to make a save.

## Validation

### Quantization and mismatch diagnostics

`export_diagnostic_hf.py` reads native distributed checkpoints on CPU and exports
their trained model tensors with the pinned Qwen mapping. It requires exact HF
parameter coverage, shapes, BF16 dtypes and tied embedding/head consistency; it
does not fill missing trainable parameters from the initialization.

`evaluate_diagnostic.py` evaluates frozen exports on the same 512 validation
questions with thinking disabled, greedy top-k=1, a 1024-token response limit,
TP=1 and at most 16 active requests. Cross BF16-trained and NVFP4-trained weights
with BF16 and NVFP4 inference to separate training effects from inference effects.
An initial BF16 control checks reproducibility against the existing baseline.

`replay_diagnostic.py` reconstructs the saved 64-answer batch from four DP shards
and performs no optimizer updates. It compares baseline gradients, a repeat with
identical stochastic-rounding RNG, a repeat with different RNG, and TIS with the
baseline RNG. Weights are hashed before and after. Gradient differences alone do
not establish an accuracy benefit; an online TIS A/B remains a separate experiment.

`watch_diagnostics.py <run-directory>` runs CPU-preflighted jobs on individually
idle GPUs, checks occupancy again inside each job, and accepts only explicitly
owned reservation PIDs during a reservation handoff. Containers use isolated
network namespaces. Each job writes its own status, predictions and results.
Pending final-checkpoint jobs wait until a complete step-100 snapshot is frozen
and exported; they never modify the ongoing formal run.

The second diagnostic round is prepared with `prepare_diagnostic_phase2.py
<new-directory> --prior <first-round-directory>`. It reuses frozen artifacts
through a read-only mount. Five inference arms share 64 predeclared validation
indices: full W4A4, BF16, weights only, activations only, and W4A4 with a BF16
output projection. The highest-accuracy ablation is automatically evaluated on
all 512 rows; this includes the screening rows and is not an independent holdout.
Completed-run diagnostic arms explicitly pin their arithmetic version so the
paired comparisons remain valid.

The eight-answer gradient probe compares resetting only the default seed,
restoring Megatron's named RNG tracker, using a separate stochastic-rounding
generator, and deterministic rounding as a diagnostic control. The subsequent
deterministic probe also sets `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and PyTorch deterministic algorithms before
initializing CUDA/backends. The 64-answer TIS replay restores the tracker and independent SR stream and requires
the identical-RNG gradient repeat to agree within relative L2 1e-5. A failed
reproducibility check is a failed diagnostic, never evidence that TIS helps.

`alignment_diagnostic.py` compares fixed step-75 weights and teacher-forced
tokens across Megatron and SGLang. It canonicalizes grouped QKV ordering,
checks weight hashes, records selected projections in layers 0/13/27, compares
weight caching and serial/batched requests, and probes growing cached prefixes.
Quantization of sampled real activations and weights is checked against the
pinned NVIDIA software reference. Prefix-extension checks are not a full
free-running decode accuracy experiment. All these jobs perform zero optimizer
updates. One failed phase-two job does not stop independent jobs; dependent jobs
are blocked visibly. Jobs can be added only after CPU preflight has passed.

The real-activation audit found a division-order difference in the original
simulator: scalar reverse division can become reciprocal then multiplication.
For global amax 10, its encode scale is 268.8000183105469 rather than the
reference's 268.79998779296875. A block amax of .9375 then rounds to E4M3 scale
44 rather than 40. The corrected default uses explicit `torch.div`, matching
TE's software-reference operation order. Set `nvfp4_sim_reference_arithmetic:
false` (or `NVFP4Recipe(reference_arithmetic=False)`) only to replay the original
experiment. Frozen phase-two legacy comparisons set this flag explicitly;
separate corrected inference arms use true. The existing step-100 weights were
trained before this correction; corrected inference does not retroactively make
them a corrected-training run. Matching sampled software-reference tensors does
not establish native hardware bitwise equivalence.

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

The first full Qwen3-0.6B attempt stopped during startup after another user
claimed the same GPU. A subsequent automatically scheduled quick smoke on GPU 7
completed both steps, generation, weight synchronization, evaluation and checkpoint
saves, then released the GPU. All its 64-token responses were truncated and all
rewards/gradients were zero. This proves process connectivity, not active learning.
The stronger training smoke above addresses this validation gap. Convergence
validation remains pending. Availability checks do not reserve a GPU against
other users.
Artifacts from this checkout are under `outputs/nvfp4-cuda-validation`,
`outputs/nvfp4-sim-preflight`, and `outputs/nvfp4-sim-smoke-a100`.

References: [NVIDIA NVFP4 recipe documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html),
[recipe API](https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html),
and the baseline image's
`transformer_engine/pytorch/custom_recipes/quantization_ref_nvfp4.py` reference.
