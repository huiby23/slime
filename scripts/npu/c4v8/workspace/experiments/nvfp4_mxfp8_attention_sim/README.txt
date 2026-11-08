NVFP4 C4 + MXFP8 V8 attention: isolated CPU numerical reference
================================================================

Scope and current experiment
----------------------------
This directory defines reproducible forward quantization boundaries, their
identity-STE surrogate gradients, and small CPU comparisons. It does not change
the BF16 backbone weights/activations outside attention. No production training,
rollout engine, CUDA/NPU kernel, or live evaluation is patched by these files.

The explicit experimental baseline is AttentionConfig(): C4 plus all specified
V8 forward boundaries, including the custom MXFP8 tree max/sum and online
maximum/denominator updates. The mixed_fp32_reduction arm is a named ablation
that keeps those sensitive states and reductions FP32. C4V16 and C4V32 controls
use the same tiles and C4 operands, with their reduction policy recorded.

"All specified forward boundaries" is not "all arithmetic instructions are
8-bit" and does not include quantized backward GEMMs. MXFP8 is a block format;
it does not prescribe the tree, exp, division, or device accumulator arithmetic.
The tree here is an experiment-defined numerical policy, not an OCP/NVIDIA
standard for summation and not a contract-specific requirement.

Format recipes
--------------
C4 nvfp4.py and test_nvfp4.py are unchanged copies of the prior independent
nvfp4_attention_sim CPU reference. Q, K, V use feature-axis block16, E2M1
payload, E4M3FN block scale and FP32 upper scale per token/head row; P uses the
same format within a query's key tile. No RHT or stochastic rounding is added.
V is scaled in token-feature layout; this is not native transposed-GEMM parity.
The P rule quantizes unnormalized online-softmax probabilities. It is not a
claim to implement an unavailable HiF4 first-order-quantization specification.

V8 has E4M3FN payload and one E8M0 power-of-two scale per 32 last-axis values:

  s = 2**ceil(log2(amax / 448)); y = s * RNE_E4M3FN(x / s)

The scale exponent is clamped to [-127,127]. E8M0 exponent bytes use bias127;
code255 is NaN and rejected. All-zero blocks use scale1. CPU tail padding uses
zeros to complete a block and never merges separate rows. The padding support
is a reference extension, not a claim that a given TE API accepts all lengths.
Scale estimation uses FP32 amax and FP64 frexp, avoiding tiny FP32 division
underflow and log2 rounding at powers of two. Reconstruction/cast overflow
raises; there is no silent Inf, format change, or higher-precision fallback.

This is NVIDIA's documented ceil-amax conversion, not OCP's suggested
floor(log2(amax))-8 conversion; both can describe MX-format payloads but differ
numerically. For amax480, this recipe uses scale2 and reconstructs480; the OCP
suggested scale1 conversion can saturate at448. A plain float8 cast without
block scales is not MXFP8.

Primary format sources (reviewed 2026-11-08):
https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Masking, axes and ordered forward boundaries
-------------------------------------------
All score/probability groups run along keys within one query row. O groups run
along features within one query row. Scalar states form their own one-element
padded block, never a block that mixes query positions. Causal -inf is applied
BEFORE score scale estimation. The sentinel is outside the payload format,
excluded from amax, preserved after QDQ, and has zero STE derivative. NaN/+inf
are errors. No future token may affect a prefix through a shared scale.

For each key tile, after Q/K/V C4 QDQ:
  1. FP32 reference QK accumulation and scalar 1/sqrt(d), then score Vector QDQ.
  2. Baseline: tree max QDQ; running maximum merge QDQ.
     Ablation: max reduction and running maximum remain FP32.
  3. QDQ(m_old - m_new); QDQ(exp(result)) gives alpha.
  4. QDQ(score - m_new); QDQ(exp(result)) gives unnormalized P.
  5. Baseline: tree sum(P) QDQ; QDQ(l_old * alpha); QDQ(sum of both).
     Ablation: FP32 sum(P), denominator multiply and addition.
  6. C4 QDQ(P), FP32 reference PV matmul, Vector QDQ(contribution).
  7. Vector QDQ(O_old * alpha), then Vector QDQ(rescaled O + contribution).
After all tiles: Vector QDQ(O/l), then cast to the input storage dtype.

The tree pairs adjacent [0,1], [2,3], etc. along the last axis. Odd levels append
zero for sum or structural -inf for max. Each level requantizes its partials in
fresh blocks32 before the next level. Even a singleton goes through QDQ. This
is not an FP32 sum followed by a terminal cast. It depends on tile size, tree
order and grouping; permutation/tile invariance is not promised.

Underlying single elementary operations (add/multiply/subtract/exp/divide),
dequantized Cube accumulations, scale computation and exact scalar constants
are host floating arithmetic in this simulation. The requested precision is
applied at the listed boundaries. No hardware transcendental approximation is
specified or emulated. Ordinary autograd and identity STE run in host/storage
precision, not an 8-bit backward implementation.

Numerical failures are part of the experiment
-------------------------------------------
The reference rejects nonfinite or nonpositive softmax denominators and invalid
uninitialized all-masked rows. It never rescues them by switching to FP32.
In a deliberately adversarial length64 max tree, 32 scores12288 followed by
32 scores near -1e9 can requantize the maximum to16384 during scale regrouping;
every exp then underflows and the denominator becomes0. The test requires an
explicit failure, while the FP32-reduction ablation survives that case.
This is a limitation of the chosen baseline algorithm and a potential subject
of subsequent compensation, not a hidden change to the baseline.

Validation and limitations
--------------------------
Run only on CPU, in this directory:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python validate_cpu.py --output-dir evidence/UNIQUE_ID

The runner disables backend autoload, rejects a loaded torch_npu module and
sets one CPU thread. It never probes/initializes an accelerator. Tests include
independent scalar/LUT format rounding, power-of-two transitions, tails, zeros,
mask exclusion, STE gradients, GQA, prefix causality and same-policy decode,
per-level tree comparison, numerical negative cases, and small random tensor
forward/gradient comparisons across baseline and controls. JSON includes source
SHA256, all policies and limitations; failures remain in their unique run dir.

Passing these tests is not evidence of 30B training convergence, RL reward or
accuracy, speed, a memory-efficient FlashAttention backward, native bit parity,
or a working train/rollout integration. Ordinary autograd retains every tile's
graph. A production implementation still needs bounded-memory forward/backward
kernels, explicit backward precision policy, train/rollout agreement, hardware
validation and controlled model/RL experiments after resources are available.
