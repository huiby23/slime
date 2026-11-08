C4V8 training bridge: query-tile recomputation and packed-layout adapters
======================================================================

This directory is an independent integration preparation, not an active model
or validated 30B RL recipe. It does not patch Megatron, MindSpeed, SGLang or the
BF16 backbone. Explicit parent-side activation and real hardware validation
are required before any production run.

Numerical contract
------------------
The supplied attention core defines all forward arithmetic. The initial core
is the independently tested C4V8 forward QDQ reference with identity-STE
surrogate gradients. Backward here replays exactly that operation graph, with
FP32 derivative arithmetic and FP32 cross-query gradient accumulation, finally
casting leaf gradients to their input storage type. There is NO new C4/V8
quantization of backward GEMMs/reductions. Do not call this full forward-and-
backward low-precision FA.

recomputed_attention(q,k,v,kernel=...,config=...,query_offset=0)
accepts BHND/GQA inputs, a deterministic core and the fixed query/key tile
policy. Forward saves Q/K/V only. Backward recomputes one query tile at a time
and frees that graph before the next. The replay K/V prefix is shortened only
to the exact causal visible end; groups remain tied to logical token indices.

This avoids retaining the whole sequence's attention graph, but is not a fused
FA backward. Peak replay storage still scales with query_tile * key_length,
plus Q/K/V and gradients. Strict core checks and Python loops can be extremely
slow on an NPU. Memory and speed claims require actual hardware measurements.
No dropout, stateful amax, stochastic rounding, higher-order derivatives or
autocast-dependent arithmetic is supported by this replay contract.

Layouts and pinned interfaces
-----------------------------
attention_sbhd returns [S,B,H*D]. attention_thd independently applies causal
self attention to each packed sample and returns [T,H,D]. Optional padded
cumulative offsets map valid lengths to physical segments, with zero outputs
and gradients in padding. Different valid Q/KV sequence lengths, CP>1 and
unrecognized masks are rejected rather than silently ignored.

Read-only server review on 2026-11-08 confirmed:
- slime/backends/megatron_utils/model_provider.py:162-183 invokes a callable
  spec with (args, config, vp_stage), otherwise builds get_gpt_decoder_block_spec.
- Megatron-LM/megatron/core/transformer/attention.py:196-205 passes config,
  layer_number, attn_mask_type, attention_type, cp_comm_type, softmax_scale and
  pg_collection when constructing core_attention.
- The same file:881-884 squeezes THD inputs; 965-973 calls the core with
  attention_mask, attn_mask_type, attention_bias and packed_seq_params;
  998-1003 reshapes THD output to [T,1,H*D].
- get_gpt_decoder_block_spec returns TransformerBlockSubmodules.layer_specs.
- PackedSeqParams contains compact and padded Q/KV cumulative lengths.

megatron_spec.get_spec deep-copies the ordinary MoE block spec, changes only
self_attention.submodules.core_attention, and leaves qkv_proj/o_proj/MoE/norm
choices intact. It requires explicit HBY_C4V8_CORE_MODULE and
HBY_C4V8_POLICY_FILE, then constructs that module's AttentionConfig from JSON.
Example prepared imports (not authorization to run an unverified model):
  PYTHONPATH=/root/hby/experiments/c4v8_rl_training_bridge:/root/hby/experiments/c4v8_device_core:...
  HBY_C4V8_CORE_MODULE=device_attention
  HBY_C4V8_POLICY_FILE=/root/hby/experiments/.../precision-policy.json
  --spec megatron_spec get_spec
The launcher must pass these values to every trainer process explicitly and
record the policy and source hashes; no installed framework source is changed.

The adapter refuses attention dropout, sliding windows, layer-dependent QK
scaling, a nonstandard softmax scale, CP>1, noncausal/cross attention, additive
attention bias and arbitrary masks. Compatibility with a new model family is
not implied by its total parameter count.

CPU verification
----------------
Run validate_cpu.py --output-dir evidence/NEW_UNIQUE_RUN with the shared CPU
Python. It records test outcomes and all bridge/reference/device-core hashes.
Tests compare values and gradients to the existing ordinary-autograd reference,
including BF16 leaf accumulation, selected requires_grad flags, query offsets,
noncausal core behavior, packed/GQA boundaries, padding gradients and RNG state.
A saved_tensors_hooks test verifies that forward saves only the three inputs.
The device-capable core is separately tested as an injected CPU callback.

The real installed MindSpeed/Megatron TransformerConfig and MoE block spec were
also inspected without constructing a model or a process group; see
evidence/pinned-spec-inspect-v2/report.json on the server. This is stronger than
the layout unit tests, but is not an NPU model-forward or optimizer test.

Dedicated 100-update comparison recipe
-------------------------------------
prepare_recipe.py creates a new recipe beside the frozen BF16 source snapshots.
It never modifies the baseline launcher or starts training. The selected
comparison is original Qwen3-30B-A3B, DAPO-Math-17k, nonthinking, fresh optimizer,
100 real updates, strict nonzero-group-variance math reward, every5 full-HF
snapshots, and all21 AIME2024 evaluations after training. FP32 master/Adam,
CPU offload0.35, TP4/PP2/EP4 training and TP8 rollout remain explicit.

prepared-v6 passed the actual installed Slime --validate-arguments path. This
returns before Ray initialization or model/NPU construction. Logs preserve the
earlier v4 sys.path failure and the successful fix; no failures were relabeled.
Later prepared versions must be validated again as source identities change.

The shared flat policy is injected into driver, trainer and SG runtime, with a
canonical JSON SHA. The SG executable is the separate python-c4v8 shim and all
three attention selectors must be hby_c4v8_sim. The ordinary BF16 main-model
weights and projection/MLP arithmetic are not globally quantized.

prepared-v8 additionally requires the simulator FP32 GEMMs to run with
torch.npu.matmul.allow_hf32=False. get_spec sets and reads this flag in each
trainer process and records trainer-policy-pid*.json; actual NPU forward rejects
an unexpectedly re-enabled flag. SG bootstrap separately applies the same
setting. Earlier numerical probes setting the flag do not establish that it
was set in earlier actual trainer/serving processes. The14-case CPU bridge
suite tests the control/audit guards but is not NPU hardware evidence.

The capacity guard retains20 every5 exports,6 rollout exports and1 temporary,
plus2% per-export margin and64GiB reserve. For the current HF model this needs
1,750,492,970,083 free bytes. It never deletes earlier models or assumes a
future deduplication gain. Run only after measured free space passes.

The launch evidence gate requires the exact current source inventory and
policy SHA, actual trainer/rollout agreement, real optimizer/update integration,
the same-policy evaluator, representative-length measured throughput, and no
unexpected Tensor CPU fallback. Small-tensor correctness alone cannot enable
the100-update run. Pending gate files are intentional and never claim success.

Real tiny-model integration probe
--------------------------------
run-probe-megatron-tiny.sh / probe_megatron_tiny.py defaults to --inspect-only.
The real NPU branch uses one random MoE layer with hidden1024/Qhead8/KVhead1/
head128, calls real GPT SBHD and two-document THD packed forward/backward, then
one actual Hybrid Adam step with FP32 master/moments and CPU offload0.35. It
requires finite gradients/states and changed QKV projection weights. This is
an adaptation test, not RL, and its update never counts toward the100 target.

Only the parent controlling resources may set HBY_C4V8_PROBE_NPU_RESERVED=1 and
ASCEND_RT_VISIBLE_DEVICES=1 and invoke --execute-npu. Every invocation requires
a new output directory; exceptions remain in report.json and are re-raised.

Independent21-point evaluator
----------------------------
prepare_evaluation.py derives dedicated CPU protocol/client/checking code from
the saved baseline sources. eval-c4v8-trajectory.py uses the original all30
AIME2024 questions, original tokenizer/nonthinking greedy4096 configuration and
unchanged math grader. The SG service uses the same python-c4v8 runtime/shim,
three explicit selectors, policy SHA and source audit as rollout. HTTP4 and
SG4/KV12288/prefill256 are an explicit execution profile, not the BF16 curve's
old30/32 profile. Each question retains its native response, token/logprobs,
score, cutoff flag and time. No scores are inferred for missing steps.

Output is declared_validation_parent/run_id, preventing collisions between
training attempts. Existing complete points are skipped only after validating
the exact manifest/source/policy identity, checkpoint path, all raw answers,
and full shard-byte SHA; an identical index JSON alone is never sufficient.
Partial/conflicting points fail closed and remain preserved. Full SHA reads
cost I/O. New points are hashed just before each evaluation rather than reading
all1.2TiB first. A resumed curve rehashes completed points before skipping them.

The evaluator refreshes an actual read-only all8-NPU/process/host-memory/port
guard immediately before starting SG. It owns a new process group and stops
only that verified PID/start-time/session; it never stops a training job or
uses global ray stop/pkill. Separate execution directories preserve prior logs.
All8 scheduler ranks must prove real backend creation, prefill and decode;
CPU-stub evidence is rejected. --validate-artifacts tests only policy, original
grader, all30 dataset identities and generated configuration; it never starts
an evaluator/model. Production completeness still requires the real21 points.

Backward precision audit
------------------------
The label is "forward C4V8 QAT/RL using QDQ and identity-STE". QDQ changes the
actual forward attention values; the optimized loss therefore sees these
errors. Identity STE uses a derivative surrogate through each rounding step,
so training can adjust BF16 parameters under that forward disturbance. It
does not make the true discontinuous quantizer differentiable and does not
reproduce quantized backward hardware. The replay computes QK/PV derivatives,
softmax/online-state derivatives and cross-query accumulation inFP32, returning
BF16 leaf gradients to the surrounding BF16 model. The optimizer keepsFP32
master parameters/moments. Calling this fully low-precision FA backward,
native NVFP4 acceleration, or a complete proprietary HiF4 implementation would
be incorrect. All performance claims require representative real measurements.

Explicit fused candidate, prepared-fast-v1
---------------------------------------
The separate fast candidate keeps prepared-v9 intact. It explicitly selects
fast_attention, query64/key128, and implementation ID
fast_attention_fused_forward_ste_v1. The six-field math policy SHA is separate
from an implementation SHA covering the nine numerical/helper sources,
installed toolchain versions and arithmetic details. Full training/evaluation
inventories also cover the serving adapter and training bridge. No module is
selected implicitly, and evaluator resume rejects changed implementation,
policy or source bytes rather than silently using the portable core.

HBY_C4V8_ALLOW_UNVALIDATED_FUSION=1 only permits preparing and probing this
candidate; it is not evidence that the candidate is ready for training. The
prepared launch gate remains pending and requires independently measured
representative throughput, no unexpected Tensor CPU fallback, actual roles,
and source-matched numerical/update validation. Its negative launcher test
rejects before Ray or a model is created. No training is launched by this work.

The fast source profile includes fast_attention.py, nvfp4_triton.py,
forward_api.py, fused_mxfp8.py, whole_decode.py, the portable three-file core,
and numerical_profile.py. Sources are loaded lazily; constructing the identity
does not import torch, Triton or the core. Source/toolchain hashes identify a
candidate and do not prove compiled kernel correctness. Real device tests
must additionally match the actual compiled implementation and current source.

The candidate retains100 real updates from the original model, every5 HF saves
and21 actual evaluations, FP32 Adam, unchanged authentic DAPO math reward and
the original nonthinking AIME protocol. Its final normalization explicitly
uses signed exact FP32 RNE division followed by MXFP8 QDQ. The current staged
candidate uses ordinary FP32 GEMM for both decode and prefill; whole_decode
remains inventoried but is not routed or accepted by numerical validation. These
details are recorded separately from the requested numerical format. Backward
remains high-precision identity STE and is not a fully quantized FA backward.

Role-probe preparation update
----------------------------
sg_smoke.py now has a stdlib-only --prepare-only mode and explicit core-module
selection. The actual branch collects HTTP health/server-info/generation,
requires actual parent/scheduler HF32=False and both prefill/decode evidence,
and rejects Tensor CPU-fallback warnings. Runtime identity is computed at
invocation and checked again before accepting the result. Owned-process cleanup
verifies PID/startticks/independent PGID/session and known runtime cmdline.
This preparation does not reserve or start an NPU. Exact commands and the
Q64/K128 policy are in role-probe-commands.txt and role-probe-policy-fast-v1.json.

The tiny probe's helper digest import is now named profile_digest so the
separate digest(path) function continues to hash actual file bytes. A pure CPU
regression checks both the bytes and the absence of a local alias collision.
Earlier evidence is not rewritten. The six CPU preparation checks additionally
exercise module/identity env, source drift rejection, process-identity guards
and actual-role audit requirements with clearly synthetic fixtures.

The later length-coverage correction retains the short SG request and adds a
separate request containing140 token IDs derived by the actual offline
Qwen3-0.6B tokenizer. Both requests demand two genuine generated tokens. The
long response's actual prompt_tokens must equal140 and lie in129..240; raw
requests/responses and separate durations are retained. Fixture/tokenizer
bytes are checked before and after execution. Tiny Megatron now defaults to
SBHD128 and THD97+31, exercising the128-key tree and real packed boundaries.
These are length-coverage checks and are not representative4096 throughput.
