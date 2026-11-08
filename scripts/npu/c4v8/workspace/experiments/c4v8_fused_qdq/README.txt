C4V8 fused Vector stage candidates (Triton-Ascend 3.2.0)
======================================================

Status: default forward components accepted on isolated NPU1 in60 test
comparisons/cases, with encoding/value comparisons bit-exact and invalid-row
rejection intact. Full attention, STE backward and model acceptance remain
outside this component result. Historical failure evidence is retained below.
No change to c4v8_device_core, production ascend backend, or memory-saver flow.
This is forward QDQ simulation, not native Blackwell/NVFP4 acceleration.

Pinned installation read directly from server:
/root/hby/envs/slime/lib/python3.11/site-packages/triton_ascend-3.2.0.dist-info
Name triton-ascend; Version3.2.0. Source snapshots are in pinned-source/.
Official repository consulted: https://github.com/triton-lang/triton-ascend
The current README targets newer3.6.0 and its install recommendations are NOT
applied. No dependency or CANN upgrade is proposed here.

Actual3.2 source confirms:
- tl.cast(... bitcast=True), integer bitwise ops, reshape, tl.max/sum/reduce.
- Ascend-specific gather is in triton.triton_patch.language.core. Imported
  explicitly rather than assuming current upstream tl.gather is exported.
- tl.exp emits create_exp. A real pre-P stage probe matches its CPU fixture,
  including exp(-inf)=0; this is not exhaustive transcendental parity.
- tl.div_rn emits create_precise_divf but peer NVFP4 tests found nonexact NPU
  rounding. Final normalization uses the peer IEEE integer-division helper.
- Compiler exposes enable_fp_fusion; candidate passes False to preserve written
  FP32 operation boundaries. Backend honoring this option needs actual review.
- libdevice.round calls __hmf_roundf; nearbyint has a separate expanded RNE
  implementation. Neither is used in E4M3 rounding. ilogb/ldexp/pow exist, but
  candidate scale formation does not rely on their numerical accuracy.

Quantization algorithm:
FiniteFP32 last-axis block32; E8M0 ceil-amax scale; E4M3FN saturation and RNE.
E8M0 powers are assembled as IEEE754 bits, including exponent-127 subnormal.
Scale exponent uses FP32 exponent bits and the exact448*2**e boundary, avoiding
log2/pow approximations. E4M3 rounding computes small positive mantissa units,
integer truncation plus explicit >half / tie-and-odd. Signed zero is restored
through sign bits. A zero block scale is1. No native float8 cast is used.
Structural-inf is excluded from amax and restored. Ordinary NaN/+inf are
flagged and produceNaN, not quietly saturated. Invalid reconstruction is flagged.
The caller must establish that -inf inputs are structural masks, not arithmetic
overflow; unmasked QK scores must be checked before inserting causal masks.

Fused regions:
- qdq_kernel: one row program; quantization plus logical bytes/scales/errorflag.
- pre_p_prepare_kernel + pre_p_denominator_kernel: score QDQ, adjacent-pair
  max tree with QDQ at EVERY level,
  running-max merge QDQ, alpha difference/exp QDQ, score difference/exp QDQ,
  adjacent-pair sum tree with QDQ at EVERY level, denominator multiply/add QDQ.
  Fixed-size gather buffers keep padded identities; partial odd tail behavior is
  checked against compact CPU tree. Scalar states have separate block32 scales.
  Outputs m,l,alpha,P. The current two-kernel split preserves every boundary
  while isolating P lifetime from the sum tree. With the finite-predicate fix,
  this defaultAPI passed width17/32/64/128 plus carry states. NVFP4 P QDQ
  and PV matmul remain mandatory externally.
- post_pv_kernel: previous-O rescale QDQ, PV contribution QDQ, O-add QDQ;
  optional final denominator division and QDQ. No P/vector boundary is removed.
forward_api.py requires explicit unvalidated opt-in and NPUFP32 input. It is
forward-only; requires_grad under enabled autograd is rejected. Training would
need an explicit recomputation/STE bridge, not implicit missing gradients.

Errors and validation:
Every kernel writes row error flags. The default wrapper synchronizes once to
reject ANY nonzero flag. check=False is only for a reviewed outer region that
combines flags and checks them before returning attention output.
Flags:1 ordinary nonfinite;2 reconstruction overflow;4 uninitialized/all-masked
maximum;8 ordinary max/score subtraction invalid;16 invalid denominator.
The reduced flag value need only be nonzero; exact bit attribution across
separate block reductions is not yet guaranteed. No failed row is accepted.

CPU validation:
.../hby/envs/c4v8-cpu/bin/python validate_cpu.py --output evidence/cpu-v1.json
Tests cover all E4M3 halfpoints and adjacent FP32 values; scale boundaries from
exponent-127 through119; approximately987k randomized finite FP32 bit patterns;
partial blocks, signed zeros, masked-inf and ordinary nonfinites; fixed-lane
tree regrouping versus compact tree; every pre-P/post-PV quantization boundary.
The test oracle is CPU only. Passing it is NOT evidence Triton kernels compile,
produce correct NPU codes, or accelerate a model.

Next acceptance (requires an exclusively authorized free NPU):
1. Same-input MXFP8 bytes/scales/reconstruction exact against validated core,
   including midpoint neighbors, every scale exponent, zeros and invalid flags.
2. Compare per-boundary pre-P/post-PV states. Separate exp/FP32 division changes
   from encoding changes. Explain threshold crossings from upstream numerical
   changes; do not silently accept unexplained percent-scale output drift.
3. Warm repeated calls without compilation; wall-time, device-time, flags and
   peak memory. Keep original negative evidence and source hashes.
4. Both Megatron and SGLang must use one frozen policy/implementation; no train
   versus rollout numerical split. Then actual isolated SG init/decode/prefill
   backend audit, followed by model startup approval from coordinating agent.

Kernel-count implications (estimates, not measured speed):
For keytileK, original Vector QDQ call count per tile is13+2*ceil(log2K):
K32=>23; K128=>27. Each currentQDQ itself expands to many torch ops/checks.
Independent fusedQDQ reduces operations inside each call but keeps23/27 launches.
The current split pre-P/post-PV uses3 Vector launches per Q/K tile. With QK, NVFP4-P,
PV still outside, at least6 operations/launch groups per tile remain, plus
initial Q/K/V quantization and final output. For a1024-token decode context,
K128 gives8tiles*48layers*6=2304 groups per request/token; separate4 requests
could give9216. K32 is4times more. Batching rows can share launches across
requests without mixing scales, but current bridge loops per request.

Whole decode GEMV follow-up:
One NPU program per row/head can loop all logical keytiles and retain m/l/O,
including ALL C4 operands/P and V8 tree/update boundaries. This could make one
main kernel per layer (48 perbatchedtoken) rather than per-tile Python loops.
BF16 KV can remain persistent; inline NVFP4 conversion has stable pertoken row
scales, but GQA sharing and exact scaling divisions must be designed carefully.
tl.sum-based GEMV changes FP32 accumulation order from current matmul and tl.exp
may differ from torch.exp. These need a documented new common reference and
bounded per-boundary numerical analysis, not an unsupported bit-equivalence
claim. whole_decode.py implements a candidate with runtime SEQ (no perlength
constexpr specialization), but it uses the currently failing monolithic tree
and has not been compiled/run on NPU after the finite-predicate fix. Runtime
binary reuse remains unverified. It stays unrouted from actual model execution.
Large prefill may benefit from keeping Cube matmuls and Vector kernels separate.

Actual NPU evidence through second bounded window (2026-11-08)
-----------------------------------------------------------
Every probe: isolated NPU1,5%HBM,HF32False,jit_compileFalse,CPUreference only,
no portable NPU fallback path, caches and temp under/root/hby. All8devices were
idle before probes and after each completed window. No full model was launched.
- evidence/npu-v1: standalone MXFP8 widths17/32/128 reconstruction,payload,scale
  exact; all247scale exponent boundaries exact; NaN/+inf flags and structural
  -inf correct.8x128 checked call0.447ms, not whole-attention/model throughput.
- evidence/vector-stage-diag-v1: through P generation each stage exact on128
  fixture. Adding sum tree to the original kernel caused canonicalNaN P values.
- npu-vector-v2: actual compiler rejection of0d block_type. Shape1 states are
  used now. npu-vector-v3 preserved original canonicalNaN/largestates failure.
- npu-compact-v1: compile timeout180s, NOT numerical evidence.
- npu-split-vector-v1: compact reshape/split tree compiler failed BiSheng
  storage-alignment propagation. It did not run; no numerical result exists.
- npu-split-gather-v1: two-kernel aligned fixed-gather tree compiled and ran.
  width17 m/l/alpha/P all exact;128 has flags0 but m/P/denominator large errors.
  Consequently zero flags alone are insufficient and this candidate is BLOCKED.
  postPV+exact signed normalization1024 outputs exact, checkedcall0.491ms.
  Peak allocated29184bytes, full source SHAs in report.json.
- npu-tree-diag-v1:69 real NPU comparisons all bit-exact. Independent gather,
  max/sum at every tree level, final shape1 directstore and aligned broadcast,
  and old-m/old-l merge pass for32/64/128. The standalone quantized tree is
  correct on these inputs. The larger pre-P fusion failure is still unresolved;
  compiler synchronization/layout is being isolated, not asserted as fact.

Model rollout and training must not select fast_attention until the parent
records an actual numerical/performance gate over the frozen implementation.
The opt-in environment means experimental execution, never a validation claim.

Further isolation evidence (all experiments remain unactivated):
- npu-split-barrier-v1 and npu-split-singlebuffer-v1 retain the same128-wide
  failures. Metadata confirms the requested compiler flags. Neither fixes it.
- npu-split-autowarps-v1 also fails identically. An early analysis incorrectly
  read CPUOptions.num_warps=-1 as the NPU default; actual NPUOptions default4
  is the same as the explicit argument. Reports contain the correction and
  original source snapshots. It was not a real arithmetic configuration change.
- npu-three-stage-v1 (maximum/probability/sum separated) also fails128.
- npu-state-stride-v1: input/output state stride1/32 and grid4/8/16 all fail
  the same maximum stage, including aligned broadcast. Pure scalar store or
  grid row count alone does not explain the failure.
- npu-max-flags-v1:MODE0 original standalone tree passes;MODE1 first scoreQDQ
  with unused initial flags passes;MODE2 retaining/ORing firstQDQ flags fails;
  MODE3 additional maximum-finite check also fails. InitialQDQ overflow checks
  are dead-code-eliminated when flags are unused, so this evidence does NOT
  prove the integer OR instruction itself is faulty.
- npu-flag-slots-v1 writes initial flags to an independent GM slot instead of
  keeping them across the tree; it still fails128. Flags are never discarded.
- npu-max-finite-v1 isolates _finite(x) implementation: integer bit predicate
  versus equivalent abs(x)<inf. The hypothesis concerns the firstQDQ internal
  overflow check and preservation of its source value, NOT only the later
  _finite(maximum) check. This minimalMODE2 now passes scalar+broadcast exactly
  for the same8rows/128 fixture. Full pre-P/quantization regression is still
  required before integration; this is not full-attention acceptance.

Frozen default forward acceptance,2026-11-08
------------------------------------------
Evidence:evidence/default-forward-components-acceptance.json and
evidence/npu-default-finite-v1/report.json. Exit0;60 cases/comparisons passed;
no CPUfallback log matches;all8NPUs released after the run. Actual default API,
NOT a monkeypatch candidate. Only accepted numerical-source change from the
previous two-kernel API:_finite(x) now uses abs(x)<inf. IEEE classification
equivalence passed1048586CPU bit patterns including NaN/Inf/subnormal/zero;
existing E4M3/E8M0 encoding and every specified QDQ/tree boundary are unchanged.
The minimal retained-flag negative/positive pair demonstrates an Ascend compiler
miscompile triggered by the integer finite predicate in this composed graph;
the exact compiler internal alias mechanism remains inferred, not proven.

Real NPU scope:MXFP8 logical bytes/scales/values,247scale-exponent boundaries;
preP17/32/64/128 with initial and finite carried state;postPV and independent
normalization;random structural masks with grids1/4/13/16 and score scales
.001/1/32/100. First score-QDQ overflow to-inf retains flag2 through subsequent
masked handling and is rejected;NaN,+Inf,and all-masked row are rejected with
no contamination of other rows. HF32False,jitFalse,NPU1,5%fraction,CPUreference.

Measured8x128 checked wall times:MXFP8 .490ms,preP(two kernels) .653ms,
postPV+normalize .463ms;three repetitions,host flag validation included. These
are component timings, not training/model throughput. Peak allocation222208B.

Frozen source SHA256:
fused_mxfp8.py e5a89e9ad647eba3928c8f18a70a09de6f6212668d7c5c05080a296021c65531
forward_api.py e5a6b9f6e6f2679d041985bccce0e74029a830c5a2c0aa91a0f2ec2c341c1cbf
whole_decode.py d58d7e879c5eb1422013c5c655daf89156f015f08f5d30454a3b9ace21627a8e

Root coordinates integratedattention+gradient andactualSG/Megatron acceptance.
Do not route whole_decode or remove the explicit experimental gate merely
because forward components passed. Keep this numerical source frozen while
integration evidence is generated.
