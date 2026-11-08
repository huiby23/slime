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

Every layer reads its full visible logical prefix from the native BF16 KV pool and computes HiF4 operands anew. The serving bridge shares integer address metadata within one forward. Quantized K/V values and scales have no lifetime across forwards. P is quantized once before PV, and the softmax denominator uses the original probabilities. This simulates HiF4 numerical precision on the 910B3.

Every fifth successful update writes an immutable HF checkpoint plus a sharded, lossless restart bundle. The bundle contains the actual model tensors, FP32 master parameters, both Adam moments and step counters, scheduler, per-rank RNG, prepared on-policy minibatches, iterator offsets and rollout source state. Run with the same topology, code, data and base model to resume. The default `--restart-retention latest` advances `RUN/hybrid-restart/LATEST` after all ranks and HF weights are durable. Historical HF weights remain available for evaluation. `--restart-retention all` retains all recovery bundles and requires storage sized for the entire run. A fresh generation after a restart can differ due to asynchronous request scheduling.

```sh
store="$RUN/hybrid-restart"
bash scripts/npu/hif4/run.sh --run --resume-from "$store/$(cat "$store/LATEST")"
```

While historical retention is being decided, launching with `--restart-retention all` keeps each state written so far. The job records a storage limit and exits at a durable checkpoint if free space falls below the next HF plus Adam uncompressed bound. To switch a running lineage after a retention decision, atomically install `retention.json` under its recovery store with the manifest's exact `hybrid_restart_fingerprint` in `fingerprint` and `retention` set to `latest` or `all`. The next fully committed checkpoint applies the choice. Switching to `latest` prunes only older linked recovery bundles after the new pointer is durable.

On eight 910B3s, the Qwen3-30B-A3B TP8 benchmark with three concurrent requests measured 3.68 to 8.66 aggregate decode tokens/s at a 1,024-token prefix and 2.14 to 5.52 at 3,968. Each timing is the median of two warm repeats. With 12 generated tokens per request, throughput including prefill was 0.84 to 1.59 and 0.13 to 0.19 tokens/s respectively. Maximum sampled HBM on any card was 12,116 MiB for the control and 12,156 MiB for the candidate. The baseline memory reading was repeated with the same inputs after its initial sampler missed five-digit readings.

All 216 greedy output IDs matched. Independent serving runs yielded different logprob values, including repeats of the same implementation. In a separate paired pass through the real model, the two attention implementations received the same Q/K/V tensors. All 2,947,940,352 BF16 output values matched bitwise across the 48 layers and all eight ranks. A small Megatron checkpoint probe using the production TP4/PP2/EP4 topology also matched the next optimizer update and all 1,624 state comparisons after restarting every rank. The evidence directory contains the paired-output summary, speed and HBM measurements, and checkpoint restart results.
