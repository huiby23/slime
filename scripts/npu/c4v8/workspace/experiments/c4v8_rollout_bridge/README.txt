C4V8 rollout bridge — explicit experimental adapter, 2026-11-08
================================================================

Purpose and boundary
--------------------
This directory prepares request-isolated SGLang attention dispatch using the
same complete numerical attention core as training. QKV/O projections and
backbone weights stay BF16; persistent KV cache stays BF16. Quantization of
Q/K/V/P and Vector stages is inside the injected attention function.
This is not native FP4/FP8 FA, not a fast kernel, and not a completed 30B RL run.
The shared device core currently implements forward QDQ plus STE backward;
using this rollout adapter does not implement quantized backward in training.

Files
-----
logical_kv.py:
  Packed Q [query_tokens,Qheads,D] plus explicit RequestSpan records.
  Token-addressable BF16 K/V cache [physical_tokens,KVheads,D].
  req_to_token [request_pool_index,logical_position] resolves each sequence.
  Prefix/chunked prefill/decode all pass the complete logical KV sequence and
  query_offset=prefix to the same core. Key tiles start at logical zero, never
  at physical page or arbitrary prefill-chunk boundaries. Shared read-only
  prefixes are allowed; new writes cannot alias another current-batch prefix.
  Writes are validated before mutation. No low-bit values are stored in KV.

sglang_bridge.py:
  A backend factory with the audited pinned AttentionBackend interface.
  Actual NPU layout is continuous [page,page_size,KVhead,D], then a token view;
  pool.set_kv_buffer remains the write API. FIA/MLA/softcap/sliding windows,
  speculative/mixed/diffusion/split-prefill modes and graphs are rejected.
  Ordinary EXTEND (including normal chunked prefill), DECODE and base-class
  IDLE are the supported modes. Prefix lengths and total lengths are checked.
  req_pool_indices has no CPU copy in pinned ForwardBatch, so it is explicitly
  copied once at init_forward_metadata, not hidden per layer/quantization op.

Pinned production integration finding
------------------------------------
Read-only snapshots under pinned-source came from:
  /root/hby/deps/sglang/python/sglang/srt/
on the user's server. No production file was modified.

hardware_backend/npu/utils.py:set_default_server_args unconditionally sets
attention_backend, prefill_attention_backend and decode_attention_backend to
ascend. Registering a new backend and passing its name on the CLI alone is
therefore insufficient.

install_in_sglang(core=...,config=...) is an EXPLICIT per-process hook. It:
  1. Registers hby_c4v8_sim and its CLI choice.
  2. Wraps NPU default selection only when hby_c4v8_sim was explicitly requested.
  3. Runs the original NPU defaults (page/memory/all-reduce configuration), then
     restores both prefill/decode to hby_c4v8_sim for that opt-in request.
Ordinary ascend requests retain the original behavior. No install occurs on
module import. No production source patch or sitecustomize is installed.
The dedicated opt-in launcher described below is supplied in this directory.

Dedicated Slime parent/worker bootstrap
--------------------------------------
python-c4v8 is an executable shim for --sglang-python-executable. It intercepts
only the exact argv pair '-m slime.backends.sglang_utils.runtime' and executes
c4v8_runtime instead. Other Python invocations pass through unchanged. Default
real Python is /root/hby/envs/slime/bin/python; HBY_C4V8_REAL_PYTHON can explicitly
select a different executable for isolated CPU checks.

c4v8_runtime reads the existing Slime JSON stdin protocol without printing it
(API credentials could be present). It preserves prepare_server_environment,
stdin repair and owning-parent-death handling, installs in the HTTP parent
BEFORE ServerArgs construction, checks effective args, and calls the actual
HTTP launch_server with run_scheduler_process_func=c4v8_bootstrap.run_scheduler_process.

That function is top-level and pickleable. SGLang's original spawn, NUMA and
memory-saver configure_subprocess contexts still create every worker. Each
worker explicitly installs its own registry before invoking the unmodified
native run_scheduler_process. Registration is not assumed to survive spawn.
The original SG rollout pause/resume/reload methods are unchanged.

Driver setup before Slime parse_args:
  from c4v8_bootstrap import register_cli_choice
  register_cli_choice()
This only adds the parser choice; it does not import a numerical core or the
NPU cache-pool module, and does not install a backend in the trainer.

Required environment passed to the SG actor/process:
  HBY_C4V8_CORE_MODULE=device_attention
  HBY_C4V8_POLICY_FILE=/root/hby/.../shared-policy.json
  HBY_C4V8_ROLLOUT_AUDIT_DIR=/root/hby/.../backend-audit
  PYTHONPATH includes c4v8_device_core and c4v8_rollout_bridge directories
Keep existing SG PYTORCH_NPU_ALLOC_CONF=expandable_segments:False,
MAX_COMPILE_CORE_NUMBER=1, HCCL settings and standard offload lifecycle.
The bootstrap preserves and records these environment values; it does not
change the trainer's allocator or create a second memory-saver implementation.

shared-policy.json is exactly the flat six AttentionConfig kwargs used by the
training spec, including explicit query_block/key_block/vector/quantize_operands/
causal/reduction. The parent freezes a SHA identity over the policy, device core
and bootstrap sources in HBY_C4V8_EXPECTED_RUNTIME_SHA256. Every child rejects
any different identity. The variable is generated per launch; do not copy an
old value into a new run after editing code.

Required Slime/SG configuration:
  --sglang-python-executable /root/hby/experiments/c4v8_rollout_bridge/python-c4v8
  --sglang-attention-backend hby_c4v8_sim
  both prefill/decode selections must be hby_c4v8_sim or left unspecified
  explicit bfloat16 model dtype; BF16/auto KV cache; TP only on one node
  graphs, torch.compile, mixed chunks, speculative decoding, PD multiplexing
  and two-batch overlap disabled; no backbone quantization
The NPU selector wrapper preserves native defaults and restores both selected
backends after them. The numerical backend also validates actual BF16 KV data.

Actual-backend acceptance (still not executed on a live model)
------------------------------------------------------------
Per-PID JSON evidence records parent/worker registration, effective server args,
each rank's actual backend construction, and its first successful prefill/decode
through the numerical core. No credentials or tensor contents are recorded.
Use verify_backend_audit.py with --audit-dir, the verified HTTP-parent PID,
--tp-size, --server-info-json and a fresh --output path. It requires all ranks,
matching source/policy hashes, actual backend selectors, both forward modes and
BF16 cache evidence. It can rehash startup sources. CPU stubbed evidence is
rejected unless the explicit --allow-cpu-probe option is supplied.

It is incorrect to infer that the backend is active merely because the parent
registered a function. Actual SGLang/NPU model execution and HTTP/rank acceptance
remain OUTSTANDING; these scripts do not start a model until used by the parent.

CPU validation
--------------
Run with the root-provided CPU environment:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /path/to/python validate_cpu.py --output-dir evidence/UNIQUE_ID

Tests use the independent CPU numerical reference and the new shared device
core on CPU. For interface dispatch they AST-extract the actual pinned base
AttentionBackend/ForwardMode code and use a CPU mock of the native KV pool.
This executes their real dispatch logic without importing torch_npu or starting
SGLang. It is deliberately not described as a live engine integration test.

Covered: GQA, random physical addresses, prefix-cache reads, writes/chunked
prefill/decode versus full reference, changing query chunk boundaries, shared
prefix isolation, BF16 cache preservation, invalid locations before mutation,
head-sharding/replicated-KV analogue, unsupported semantics rejection, native
write API call sequencing, and explicit backend selection wrapper isolation.

Costs and remaining acceptance
------------------------------
This functional bridge gathers full logical K/V separately for every request
and re-quantizes the current context for every decode token. Python per-request
dispatch plus every-tile/every-tree-level Vector QDQ produces many kernel
launches. Validation checks (min/max/unique/equality) may synchronize a device.
Memory is lower than a full batch S² implementation but gather copies and the
numerical core's buffers still need measurement. An output-equivalent CPU
result provides no evidence of usable 30B RL throughput.

The next steps require separately authorized hardware testing: actual native
BF16 cache write/read, real prefill and incremental decode, long-prefix memory
and timing, each-rank TP4 training versus TP8 teacher-forced logprob comparison,
then short real RL. Do not omit P/softmax quantization to claim a fast C4V8 run,
do not switch to native BF16 attention silently, and do not use old BF16 policy
rollouts as if they came from the new C4V8 behavior policy.

Latest CPU evidence
-------------------
18/18 unit tests passed in evidence/cpu-20261108-v2. A separate real Python
spawn test passed with two child processes in evidence/spawn-20261108-v2. It
executes the real bootstrap and registry hooks, the AST-extracted pinned NPU
default-selection function, and CPU attention prefill/decode. SGLang/hardware
dependencies are explicitly stubbed; artifacts say cpu_stubbed_sglang and no
torch_npu was imported. This is not a native SG, NPU or distributed-model test.

The shell shim was syntax-checked and its two dispatch paths exercised with an
argument-printing executable. No production server was started by these checks.
