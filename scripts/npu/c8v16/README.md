# W16A16C8V16 on Ascend 910B3

This experiment uses the original Qwen3-30B-A3B in non-thinking mode. Slime's Megatron actor and SGLang's prefill/decode backend call the same causal attention function.

| Path | Forward precision |
| --- | --- |
| Backbone, projections and stored KV cache | BF16 |
| QK | Q and K use MXFP8 E4M3FN elements, each group of 32 head features sharing an E8M0 power-of-two scale. The product accumulates in FP32. |
| PV | Unnormalized softmax probabilities and V use MXFP8 along the contracted key positions. Every query computes the final partial V group's scale from its visible prefix. The product accumulates in FP32. |
| Vector | Scores, max, exponential, tile sum, online softmax state, PV contribution and output are rounded at BF16 boundaries. Elementwise operations and the within-tile sum use FP32 before rounding. |

The E8M0 scale uses a ceiling for the block maximum divided by 448. The NPU path runs a Triton QDQ kernel and FP32 matmuls with HF32 disabled. Training replays query tiles for first-order straight-through gradients. Official format definition [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

The C4V8 comparison also changes the V grouping axis in PV. That difference affects how to interpret changes in accuracy.

## Train and inspect

Use the Ascend environment and pinned dependencies specified in the sibling C4V8 archive. Keep the entire Slime checkout together, and set `SLIME_WORKSPACE` to the directory containing the model, data and environment (deployed under `/root/hby`). Set `HCCL_IF_IP` to the communication address of this node. From the repository root:

```sh
recipe=scripts/npu/c8v16/workspace/experiments/c8v16_recipe/run.sh
bash "$recipe" --validate-arguments
bash "$recipe" --run
```

The recipe generates new DAPO-Math-17k responses, selects groups with nonzero reward variance, and targets 100 GRPO optimizer updates. Five questions × 16 samples per rollout and a global batch of 4 give 20 updates per rollout. There are five rollouts. Training uses TP4 × PP2, EP4, and a BF16 backbone with FP32 Adam. The recipe saves HF weights every five successful updates and records an offline evaluation protocol for 30 AIME2024 questions.

Each run records the policy and source hashes in its manifest. The `trainer-backend-audit/` and `backend-audit/` directories record the selected implementations for trainer ranks and SGLang workers respectively. The adapter tests compare packed training, cache writes, prefill and decode on matched inputs.

The single-device validation uses 133 tokens, four query heads, one KV head and a head dimension of 128. The saved report contains the actual forward and gradient differences against the CPU reference and prefill/decode errors at three prefix lengths. See [device test](verify_device.py) and [Ascend results](evidence/ascend910b3.json).
