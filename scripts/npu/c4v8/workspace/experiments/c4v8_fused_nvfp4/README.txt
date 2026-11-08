Independent Ascend Triton NVFP4 QDQ candidate
============================================
This directory is a numerical simulation performance candidate, not native
NVFP4 hardware and not an approved full-model training kernel. The portable
reference and its production guards remain unchanged.

nvfp4_triton.py preserves block16 E4M3 RNE scales, FP32 per-row upper scales,
and E2M1 RNE payload values. Its straight-through backward is identity; this
is not quantized backward compute. Compiler FP fusion is disabled deliberately.

Register-only API for larger fused kernels:
  nvfp4_value_2d(x) -> values, invalid
  x is a Triton FP32 tensor [ROWS, WIDTH], WIDTH a power of two >=16.
  WIDTH must contain the full row whose upper scale is shared; missing logical
  columns may be zero padded. Do not call independently on16-key pieces of a
 128-key P row, because that changes the upper-scale algorithm.
  The helper performs no loads/stores or host operations. invalid is [ROWS].
  Invalid rows produce NaNs and must be checked/propagated by the caller.

The positive FP32 division helper _divide_positive_rne uses IEEE754 integer
significands and int64 quotient/remainder with explicit ties-to-even rounding.
It handles zero, subnormal and overflow. Signed callers must apply the original
sign bit separately, preserving negative zero, and validate their own domain.

Actual evidence:
  npu-fused-v1: failed. Native tl.div_rn did not equal CPU FP32 RNE.
  npu-fused-v2: failed overall. Integer-significand division passed196608
  cases across the full positive finite exponent range. NVFP4 width16 passed
  payload/scales/values. Width35 failed payload/values with huge unexpected
  values despite exact scales; performance was not measured.
  cpu-padding-equivalence.json:22 cases prove physical zero padding to a
  power-of-two row leaves logical reference values, payload and scales exact.

npu-fused-v3: numerical pass (exit0,193.15 seconds including first-shape JIT).
  All196608 division cases,21 FP32/BF16 quantization cases (width1..129,
  zeros, subnormals, real P counterexample), inference_mode, invalid rejection
  and propagation, and STE identity passed. Physical power-of-two row padding
  fixes the width35 failure without altering logical reference values/codes.
  Warm512x128 strict-check QDQ measured portable15.535ms, fused0.621ms,
  ratio25.0x, peak allocator35.3MB. This is a microbenchmark, not model speedup.
  Eight NPUs were idle after process exit; release evidence is saved.

The combined v3 process log contains aten::bitwise shift CPU fallback warnings
near the portable-reference timing phase. Attribution was not instrumented,
so fused-only absence of fallback is not yet established. fallback-audit.json
records this limitation; validate_fused_only.py tests CPU reference followed
by only fused NPU operations with explicit phase markers and jit_compile=False.
It must be run under a new authorized NPU1 slot, not alongside another job.
Old failure reports and exact v2/v3 source snapshots are retained.

validate_fused.py only runs on NPU1 after an all8-idle gate; it limits process
HBM to5%, records source SHAs, tests format/value/scale equality, inference
mode, invalid-input propagation and STE, then benchmarks warmed512x128 QDQ.
Run only with explicit ownership of the NPU1 test slot and a bounded timeout.

Follow-up npu-fused-only-v4 (2026-11-08)
---------------------------------------
NROWS is now a runtime scalar with do_not_specialize, keeping physical width
and row/group tile sizes static. The standalone probe exits0 in33.78 seconds:
rows13->17->257->13 reuse the same JIT cache key (first call1 compilation,
subsequent calls0); values, logical codes and scales remain bit exact.
FP32[512,128] ordinary/inference-mode and STE identity pass with
jit_compile=False and allow_hf32=False. Its complete log contains no backend
CPU fallback warning; backend-log-audit.json preserves checked patterns and
log SHA. Unlike v3 this process never invokes the portable reference on NPU.
The10-repeat strict-check+logical-bytes average is0.672ms, peak34.8MB.
All8 NPUs are idle after exit, with snapshot saved. This scope is still a
quantizer microtest, not full FA, training update or production acceptance.
The preserved v3 source and its broad format tests remain separately recorded.
