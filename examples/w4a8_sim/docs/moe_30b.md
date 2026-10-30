# Qwen3-30B-A3B MoE preparation

This document preserves the earlier 30B preparation and small-model validation
notes. Full-size 30B formal training was not completed. The current completed
Dense experiment is described in [the 4B guide](../QWEN3_4B.md).

This experiment is separate from the completed Qwen3-0.6B W4A4 experiment.
It loads original BF16 `Qwen/Qwen3-30B-A3B` initialization, not an INT4/FP8
checkpoint and not the differently configured Instruct-2507 model.

The requested scope is **linear layers**. FlashAttention, softmax and attention
KV caches retain their baseline precision. This is not the contract's FA-core
C4V16 implementation and does not simulate HiF4. It does not provide native
low-bit speed or memory savings on A100.

## Precision policy

| Operation | Left operand | Right operand | Accumulation/output |
| --- | --- | --- | --- |
| FPROP | FP8 E4M3 activation | NVFP4 weight | FP32 / BF16 |
| DGRAD | FP8 E5M2 output gradient | NVFP4 transposed weight | FP32 / BF16 |
| WGRAD | FP8 E5M2 transposed gradient | FP8 E4M3 activation | FP32 main gradient |

NVFP4 weights use reference E2M1 payloads, E4M3 block scales, an FP32 tensor
scale, and 16x16 weight tiles. FP8 uses per-row blocks of **64** with FP32
current scales and nearest-even rounding. This block size divides the expert
width 768 both unsharded and after rollout TP=4 (192), avoiding changed
activation grouping between trainer and rollout. Inference reduces weight
tensor amax across the TP group before quantization. Cached amax values are
invalidated after the stock `load_weights` update path, including `.data` writes.

This is a **custom W4A8 recipe**, not NVIDIA's default NVFP4 W4A4 training
recipe. Backward uses FP8 gradients, not FP4 gradient stochastic rounding; no
Hadamard transform is applied. Payloads are unpacked BF16 and products use FP32
PyTorch GEMMs with TF32 disabled. Native block accumulation order is not emulated.

Attention QKV/output projections and every routed expert's gate/up and down
projections are quantized. The router computes in FP32 in both backends. RMSNorm,
embedding lookup and LM head stay high precision; original Parameters and native
checkpoint keys are retained. Native Megatron expert dispatch, weighted
combination, distributed gradients and Adam updates remain in charge.

## Four-GPU plan

- Trainer: TP=PP=CP=ETP=1, EP=4, bias-free TE grouped experts with separate weights.
- Rollout: TP=4, EP=PP=1, Triton MoE backend with standard top-k, eager execution.
- Full-parameter Adam: FP32 master weights, main gradients and both moments.
- Explicit CPU optimizer offload fraction **1.0** and precision-aware optimizer.
- Training/rollout use native Slime CPU offload between phases, not simultaneous
  permanent residency. Microbatch 1, context 2048, response 1024, full recompute.
- GRPO, seed 42, 8 prompts x 8 answers, LR 1e-6, 100 optimizer updates; thinking
  disabled. Initial compatibility data is the previous GSM8K train/val split.
  This is not the harder MMLU/GPQA/AIME evaluation suite.

From the published BF16 index: 30,532,122,624 parameters, 61,064,245,248 weight
bytes. EP=4 gives an estimated **49.1 GiB per GPU for BF16 weights plus FP32
main gradients**, before activations and buffers. FP32 master/Adam CPU state
is about **341.2 GiB** overall. These are capacity estimates; a complete 30B
smoke must measure actual peak memory before formal training.

A complete checkpoint is approximately **398.1 GiB** before metadata. Save after
every optimizer update, retaining only best and last. Put their owned physical
snapshots on two filesystems; write and rename each pending checkpoint on its
own filesystem, then atomically replace logical symlinks. Prune only owned,
unreferenced complete snapshots. Incomplete writes never replace last.

## Preparation and validation

Download original initialization (the downloader pins file revisions and verifies
published SHA256 values when supplied; partial downloads are resumable):

```bash
python3 examples/w4a8_sim/download_model.py --source modelscope \
  --destination /data3/huibingyu/models/Qwen3-30B-A3B \
  --report /data2/huibingyu/nvfp4-sim/qwen3-30b-w4a8-20261028/model-download.json
```

Prepare an inspectable plan without claiming GPU validation:

```bash
python3 examples/w4a8_sim/prepare_qwen3_30b.py \
  --model /data3/huibingyu/models/Qwen3-30B-A3B \
  --output /data2/huibingyu/nvfp4-sim/qwen3-30b-w4a8-20261028 \
  --checkpoint-primary /data4/huibingyu/nvfp4-sim/qwen3-30b-w4a8-20261028 \
  --checkpoint-overflow /data3/huibingyu/nvfp4-sim/qwen3-30b-w4a8-20261028 \
  --gpus 0 3 4 7
```

Use the original Slime runtime and `PYTHONPATH` from the plan. CPU preflight
parses the real backend CLI with the planned world size and a mocked A100
capability query; it creates no CUDA context and does not validate GPU kernels.
The standalone `validate_megatron.py` separately exercises real expert parallel
training, CPU optimizer state, full DCP serialization and repeated post-resume
updates on idle CUDA devices. It uses a small randomly initialized MoE, not 30B.

Before formal launch require: numeric/backend tests, CUDA EP=4 integration,
rollout integration, complete/hash-verified weights, free-storage checks, and a
real full-size smoke including optimizer updates, native weight synchronization,
complete last checkpoint and resume. A CLI plan or small-model test alone must
never mark the 30B experiment running or validated.

`observe_idle.py` only records GPU resources. It neither reserves devices nor
stops any process. Require continuous observed idle time >300 seconds before a
new experiment. Observe later foreign contention without killing existing jobs.

`queue_validation.py --work <evidence-directory> --tokenizer <original-model>`
queues actual rollout TP=2, training EP=4 and rollout TP=4 checks for up to 24
hours. It records its live state in `validation-queue.json`, checks GPU process
ownership again immediately before launch, and preserves container logs. GPUs
released by its own successful stage may continue the next stage after a fresh
idle check. It does **not** start full-size training after these small tests.

Validation evidence for the 2026-10-28 preparation: 45 CPU numerical/backend
tests passed; the real two-rank MoE run with fully CPU-offloaded Adam passed
complete distributed save/load and an exact repeated optimizer update. Four-rank
integration and the full 30B smoke remain separate launch prerequisites.
