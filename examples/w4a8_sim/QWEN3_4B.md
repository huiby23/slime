# Qwen3-4B BF16 and NVFP4 W4A8 experiment

This branch records the completed 100-update Qwen3-4B comparison. The full
Slime source, quantization implementation, adapters, launch tools and tests are
versioned here. The exact formal commands, environment, data hashes, source
revisions and results are in
[experiments/qwen3_4b_20261030.json](experiments/qwen3_4b_20261030.json).

## Precision and scope

- Initialization: original `Qwen/Qwen3-4B`, 4,022,468,096 parameters.
- W4A8 linear weights: NVFP4 E2M1, E4M3 block scales, FP32 tensor scale,
  with 16-by-16 weight tiles.
- Linear activations: FP8 E4M3; backward output gradients: FP8 E5M2.
  FP8 uses per-row blocks of 64 and FP32 scales.
- Matrix products use FP32 PyTorch GEMMs with TF32 disabled; outputs return
  in BF16. Master parameters, main gradients and Adam moments remain FP32.
- Attention projections are quantized. FA QK/softmax/PV, embeddings, norms
  and the LM head retain their baseline precision.

This is an A100 numerical simulation of a custom linear W4A8 recipe. NVIDIA's
default W4A4 recipe, FP4 gradients, Hadamard transforms, packed FP4 checkpoints
and native FP4 performance are outside this version's implemented scope.

## Code entry points

| Purpose | Path relative to repository root |
| --- | --- |
| Quantization arithmetic | `slime_plugins/w4a8_sim/recipe.py` |
| Megatron training adapter | `slime_plugins/w4a8_sim/megatron.py` |
| Shared Dense/MoE SGLang linear interface | `slime_plugins/w4a8_sim/sglang_linear.py` |
| SGLang rollout adapter | `slime_plugins/w4a8_sim/sglang_models/qwen3.py` |
| Atomic best/last policy | `slime_plugins/w4a8_sim/checkpoint_policy.py` |
| Matched BF16/W4A8 plan | `examples/w4a8_sim/prepare_qwen3_4b.py` |
| Full-size validation and formal controller | `examples/w4a8_sim/run_dense_experiment.py` |
| Training launcher | `examples/w4a8_sim/train_best_last.py` |
| Plan, gradient gate and resume checks | `tests/test_w4a8_dense_plan.py` |

## Completed formal experiment

Both arms used seed 42, TP=1 and DP=2, with two colocated TP=1 rollout engines.
BF16 used physical GPUs 0/6 and W4A8 used 2/6, with matching parallel topology.
Each update sampled eight prompts with
eight answers each, for a global batch of 64. Each arm completed 100 optimizer
updates. The learning rate was a constant 1e-6, Adam betas were 0.9/0.98,
weight decay was 0.1, and the maximum response length was 1024 tokens.

| Own-mode GSM8K validation, 512 questions | Initial | Best | Final |
| --- | ---: | ---: | ---: |
| BF16 | 92.578125% | 92.968750% at update 75 | 91.796875% |
| W4A8 | 90.8203125% | 92.578125% at update 100 | 92.578125% |

Both formal validations passed with successful optimizer records, finite
sampled master parameters and nonzero gradients observed during training.
These are single-seed, own-precision evaluation results; they do not establish
a consistent accuracy advantage for either precision.

The BF16 training source was commit `6eb03e4c`; W4A8 used `a7bf5a76`.
The latter adds initial-evaluation monitoring/adoption handling. Frozen sources
and original launch manifests remain beside the experiment artifacts.

Later local cleanup separates the shared rollout linear adapter from the MoE
model module and formats the W4A8 source with Black at 119 columns. The
quantization and optimizer computations retain the same Python AST, and the
moved linear method retains the same class body. The result index's source
hashes identify the original formal commits; checkout the baseline tag for
those exact source files.

## Version management

The experiment branch is `research/qwen3-4b-w4a8`; the completed baseline tag
is `qwen3-4b-w4a8-v1-20261030`. Run Git commands from your repository root:

```bash
git status
git log --oneline -10
```

Commit the code/configuration paths related to each change and check
`git remote -v` before publishing the branch.

## Checkpoints and reuse

Each complete resumable last is about 52.45 GiB. Each HF BF16 weights-only
best is about 7.51 GiB including tokenizer/configuration files. A best export
stores BF16 weights; W4A8 evaluation requires explicitly enabling the
quantization adapters when loading it.

The result index records paths used by the completed experiment for provenance.
New runs choose their own model, output and checkpoint locations through
`--model`, `--output` and `--checkpoints`.

The completed experiment's smoke/test artifacts, caches and containers were
cleaned after use. Its historical resume-after-validation shortcuts therefore
cannot reuse those deleted reports. For a fresh experiment, choose a new run
directory, prepare its plan and rerun the required validation gates. Preserve
the existing completed run.

For example, planning a new two-GPU run does not start training:

```bash
python3 examples/w4a8_sim/prepare_qwen3_4b.py \
  --model /path/to/Qwen3-4B \
  --output /path/to/experiment \
  --checkpoints /path/to/checkpoints \
  --gpus 1 2
```

GPU indices in this planning example are placeholders. Launching also requires
current idle-GPU observations, checksum-verified inputs, frozen sources, enough
storage and passed full-size update/checkpoint/resume gates.
