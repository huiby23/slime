Independent staged C4V8 decode candidate
=======================================
No production backend is registered or modified. No NPU test has been run for
this directory. It is a forward-only numerical simulation, not native low-bit
hardware or a low-precision backward implementation.

The candidate handles Q[B,H,1,128] and BF16/FP32 K/V[B,HK,SEQ,128]. It retains
fixed logical128-key online boundaries and the original feature/key scale
axes. It performs a fixed number of major passes per call rather than a Python
kernel chain for each key tile:
 1. Existing exact NVFP4 QDQ on Q/K/V; torch FP32 full-sequence QK.
 2. Independent score MXFP8/local-max tree for each key tile.
 3. A small ordered m/alpha scan. Its recurrence is independent of L/O.
 4. Independent P8 per tile, separate exact NVFP4 P4, and sum-tree(P8).
 5. Torch FP32 batched PV using P4; MXFP8 contribution boundary.
 6. A small ordered L/O scan retaining every multiply/add QDQ boundary, then
    correctly rounded FP32 normalization and the final MXFP8 boundary.

P8 is used by the denominator; P4 is used only by PV. Do not sum P4 or replace
an ordered state scan by a single FP32/global reduction. QK/PV accumulation
remains ordinary FP32 simulation. Runtime SEQ/TILES are do_not_specialize
arguments, while head128/key128 stay static. Kernel identities can be recorded
through HBY_C4V8_AUDIT_KERNEL_CACHE=1.

Physical state rows are32elements/128bytes to reduce unaligned-store risks;
that precaution is not claimed to fix the currently investigated compiler
issue. Error checks use independent per-stage GM slots. Initial score flags
are stored before a complex tree, not OR-accumulated across it. Every written
slot is checked by the wrapper. The shared MXFP8 helpers still require their
own real-device gate. No checks were removed.

CPU evidence
------------
evidence/cpu-v2/report.json covers28 cases: FP32/BF16, lengths1/17/33/128/129/
255/256, single/multiple batch and GQA ratios8:1/2:1. Reordering the same tile
GEMMs into the staged schedule preserves outputs bit-for-bit. The proposed
batched GEMMs also yield exact final outputs in these cases, but4 cases have
raw QK/PV FP32 differences up to8.34e-7/9.54e-7. Those differences are recorded,
not hidden under a tolerance. They can cross later rounding boundaries on
other inputs or devices, so this is not NPU equivalence or speed evidence.

NPU probe prepared, NOT run
--------------------------
validate_npu.py requires an all8-idle gate and uses only NPU1 with5%HBM,
jit_compile=False and HF32=False. CPU reference never executes on NPU. It
saves every stage before rejecting flags, records source and binary identities,
compares several sequence lengths and head partitions, tests invalid input and
unimplemented-backward rejection, and logs phase markers. A completed probe
is distinct from production acceptance; complete stdout/stderr must also be
checked for backend CPU fallback. Run only after explicit NPU1 ownership.

Limitations
-----------
This initial candidate recomputes NVFP4 for all history K/V and expands GQA
K/V tensors. There is no persistent prequantized cache. Its temporary memory
is linear in sequence length, but GQA copies can be costly. A later exact
cache must live within SG's existing kv_cache/offload allocator, be invalidated
with page reuse/resume/weight updates, and must never re-QDQ cached dequantized
NVFP4 (CPU evidence shows repeated NVFP4 values are not bit-idempotent).
