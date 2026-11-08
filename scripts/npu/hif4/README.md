# HiF4 operands for BF16 FlashAttention

This research implementation converts BF16 inputs to HiFloat4 values using groups of 64, an E6M2 base scale, 8- and 4-element subgroups with binary scale exponents, and signed S1P2 payloads. It simulates the operands of QK and PV. Q/K group along head features and V groups along logical key positions. A causal query only uses visible V values when computing the scale for its final partial group. The existing BF16 Vector boundaries, FP32 accumulation and straight-through backward are shared with the C8V16 experiment.

The experimental probability rule must be supplied explicitly to the attention configuration. The training recipe selects `single_hif4_qdq_before_pv`, rounds the online softmax tile's unnormalized weights once on their way to PV. The denominator sums the pre-HiF4 values at the C8V16 precision boundaries. The project-specific meaning of “first-order” remains open. The training and rollout actors load the same named HiF4 backend and record their policy and implementation hashes.

The converter follows Algorithm 1 using nearest-even BF16/E6M2 scaling and half-away S1P2 rounding as used in the public NumPy implementation. Training rejects nonfinite ordinary inputs. The format represents an all-zero group exactly. On this NumPy version the upstream scalar helper produces NaNs for an entirely zero group, so that edge case and the sign of zero are tested separately. The reference comparison covers 38,080 decoded finite values across 595 groups, after canonicalizing the signs of zero. See the [format specification](https://arxiv.org/html/2602.11287v1) and [pinned upstream reference](https://github.com/GCC-HiFloat/HiFloat4-Quantization_Library/blob/6d937b6fcf34f63b8fc563bd72e3aea0f44a46b4/hifx4_npu/HiF4_NVFP4_v14f16.py).

From the repository root, run `python scripts/npu/hif4/check_hif4.py --device cpu`. Pass `--upstream PATH` to recompute the golden values using the pinned reference. The same test on one idle Ascend device with `--device npu` will also compare device output and gradients against CPU. The 910B3 device tests in `evidence/` check rounding at 1,152 additional group maxima and exercise training forward/backward and cached decoding. An explicit BF16 rounding operation is needed before rounding the global scale to E6M2 on this toolchain.

## Run

Use the pinned Ascend environment from the sibling C8V16 recipe. Set `SLIME_WORKSPACE` to the model/data/environment root and `HCCL_IF_IP` to the current node communication address. The original Qwen3-30B-A3B model starts from step 0. Five fresh DAPO rollouts give 100 optimizer updates with five mixed-reward questions, sixteen answers per question and global batch four. The recipe saves full HF weights every five updates. It records an offline AIME2024 protocol for the 21 model points.

```sh
bash scripts/npu/hif4/run.sh --validate-arguments
bash scripts/npu/hif4/run.sh --run
```

The rollout backend batches key tiles for single-query decode and preserves the 128-key online update order. The 910B3 comparison covers prefixes from 1 through 8,192. In the 4,097-key case with three requests this changed the median single-layer time from 77.6 ms to 12.7 ms. These timings describe attention only. End-to-end training time includes prompt filtering, generation, full-model gradients and checkpoint I/O.
