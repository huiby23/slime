Portable NVFP4/MXFP8 attention numerical core — 2026-11-08
========================================================

This isolated experiment implements the previously defined CPU FA forward
quantization graph using portable torch FP32/integer operations. Its model
backbone is outside this module. It is not a native NVFP4/MXFP8 accelerator,
not a complete 30B RL integration, and not a low-precision backward kernel.

Public interfaces
-----------------
device_quant.py:
  nvfp4_qdq(x, check=True)
  mxfp8_qdq(x, allow_masked_negative_infinity=False, check=True)
  nvfp4_values / mxfp8_values optionally return logical payload/scale bytes.
  Quantizers use identity STE; structural mask gradients are zero.

device_attention.py:
  AttentionConfig(query_block=...,key_block=...,vector='mxfp8',
                  reduction='mxfp8_tree_reduction',quantize_operands=True)
  attention(q,k,v,config,query_offset=0)
  Inputs are B,H,T,D; GQA requires Hq divisible by Hkv. All inputs share a
  device/dtype. Core arithmetic casts inputs to FP32, then casts final O to
  input storage dtype. The training bridge can inject this function and use
  per-query-tile recomputation; this module itself uses ordinary autograd.

The numerical policy is unchanged from nvfp4_mxfp8_attention_sim:
  C4: E2M1 RNE; 16-feature blocks with E4M3FN scale; FP32 per-row upper scale.
      Q/K/V group along token features. P groups along the logical key tile,
      using unnormalized online-softmax probabilities.
  V8: E4M3FN/E8M0 block32; NVIDIA-documented ceil-amax power-of-two scaling;
      explicit boundary QDQ and experiment-defined adjacent binary max/sum
      tree with re-quantization at every level.
  Structural causal masks precede score quantization and remain out of band.
  Key tiles begin at logical token0, independent of physical cache pages.
  Matrix accumulation and elementary operations are FP32 numerical simulation.
  Backward is ordinary high-precision autograd with STE, not quantized GEMMs.

Actual NPU findings and repairs
------------------------------
All tests were authorized only after the prior evaluation released all8 NPUs.
Only NPU1 was used, at small tensor sizes, with a 5% per-process HBM limit.
No model/Ray job, driver modification, global stop or device reset was used.

1. Native torch.copysign fell back to CPU; this implementation now manipulates
   sign bits. Native signbit(-0) also lost the sign in the observed stack.
2. Native pow(2,e) was not exact for235/255 integer exponents. Such tiny scale
   errors changed121/758 E4M3 boundary encodings. IEEE754 integer-bit construction
   now produces exact powers, including exponent-127's subnormal encoding.
3. A complete stage trace found the next first divergence in P's C4 QDQ:
   identical P and identical block-decode scale, but native reciprocal produced
   a different block-encode value. A normalized value moved from CPU
   1.74999988079 to NPU1.75, changing E2M1 RNE's code. Newton/addcmul attempts
   were not reliable; their evidence is retained.
4. exact_fp32.py computes positive FP32 division using integer significands,
   integer quotient/remainder and explicit nearest-even rounding. It handles
   subnormals/overflow/zero without float64 or CPU tensor copies. It is used
   only for NVFP4 scale formation. This preserves the validated FP32 operation
   sequence rather than changing the scale recipe or hiding mismatch.

Evidence
--------
CPU v4: 12/12 tests, including600,000 full-exponent random division cases,
format values/bytes/scales against the original CPU reference, boundary cases,
causal prefix and decode, and forward/STE gradient parity.
NPU format diagnostic v2: exact powers255/255, E4M3 values and codes758/758,
40 dyadic MXFP8 values, and9 scale boundaries with both nextafter neighbors.
NPU v4:196,608 random positive FP32 division cases exactly match CPU RNE;
NVFP4/MXFP8 random payload, scale and reconstructed values are exact.
32/64/128-token FP32-input attention outputs are exact against same-process
server CPU; STE gradient relative L2 is about0.7e-7..2.2e-7.
NPU BF16 v5 deliberately retains its failed strict CPU-exact output gate:
1/32768 values differ (max0.0078125, relative L2=9.9319e-5) for Hq8/Hkv1,D128.
Stage trace v2 localizes the first difference to ordinary FP32 PV matmul
accumulation (input difference <=9.54e-7 crossing an MXFP8 midpoint), after all
preceding quantizer codes match. This is separate from the repaired same-input
quantizer bug. On the same NPU, full prefill vs token decode and head8 vs two
head4 partitions are bit-exact across all32768 outputs. The old strict failure
is preserved; this explanation does not pretend CPU and NPU BLAS are identical.
CPU Q64/K128 and Q128/K128 policies pass reference output/gradient checks;
changing the logical key tile changes the online numerical policy and must be
recorded in any experimental configuration.

The report describes what passed. Earlier failures remain in v1/v2/v3 and
the stage/format/division diagnostic directories; they are not overwritten.

Timing and production limits
----------------------------
The strict eager graph synchronizes on many validation assertions. CPU dispatch
counts before the exact-division repair were4,026/11,431 forward torch calls
and116/328 scalar reads for lengths32/64 (Q16/K32, Hq4/Hkv2,D32); these are not
NPU kernel counts. The repaired v4 NPU measured0.65s/2.05s forward for64/128
tokens in that configuration, excluding a separate first-shape compile cost.
These are small attention-layer timings, not full-model token throughput.
For4096 tokens with Q128/K128 and48 layers, static accounting gives roughly
686,000 Vector QDQ calls per model forward; each expands into many torch ops.
This eager implementation must not be blindly deployed for100-step30B RL.

A future optimization can aggregate device-invalid flags and check at explicit
boundaries, but must preserve failures rather than silently suppress them.
Kernel fusion must preserve the tested bit-level scale/RNE rules, operation
ordering, masks and layout equivalence. Source metadata reports npu_validated
false as a production capability flag; small-tensor evidence is scoped here.
Parent-run SG/Megatron tiny integrations and a retrospective outer-log audit
confirm actual aten::bitwise shift Tensor_out CPU fallbacks in BOTH isolated
v4/v5 tests and real runtime roles. The numerical reports had not incorporated
those outer-log warnings; no role-specific explanation is needed. Therefore
all-NPU execution is NOT established for this eager core; its numerical
validation must not be mistaken for no-fallback backend validation. Separate Triton fusion candidates are being tested; final
acceptance must include inference_mode and real train/rollout roles.

Actual packed Megatron/SGLang operation, low-memory backward integration,
train/rollout logprobs, full30B update and practical throughput remain gates.

How to validate
---------------
CPU (never initializes NPU):
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python validate_core.py --device cpu \
      --output evidence/NEW_CPU_RUN

Authorized NPU1 only, after verifying all8 cards idle:
  source /root/hby/activate-slime.sh
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_COMPILE_CORE_NUMBER=1 \
    python validate_core.py --device npu:1 --authorize-npu1 \
      --output evidence/NEW_NPU_RUN --lengths 32 64 128

Do not run this beside an existing full job. Diagnostics intentionally copy
small tensors to CPU for analysis; device_quant/device_attention do not use
CPU tensor copies. The strict checks do synchronize scalar validity results.
