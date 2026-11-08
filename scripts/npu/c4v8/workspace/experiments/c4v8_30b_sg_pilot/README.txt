Prepared full Qwen3-30B-A3B SGLang performance pilot; no NPU/model run yet.

This is the next end-to-end measurement only if root grants all eight NPUs.
It does not generate training rollouts, compute math rewards, update weights,
or contribute any GRPO step. Responses stop at a fixed16 (optionally32) tokens,
so this is not an accuracy evaluation or a full RL throughput estimate.

Actual config
  Original /root/hby/models/Qwen3-30B-A3B, BF16 backbone/native BF16 KV.
  TP8, PP1, DP1, all physical NPUs0..7; context8192, KV capacity12288,
  memory fraction0.135, prefill chunk256, max running requests1.
  Same explicit full C4V8 core and caller-supplied six-field policy; numerical
  source/toolchain, policy and runtime identities are generated at each invocation.
  Query block64/256/512/1024 can be read from the final policy, key128 unchanged.
  Prepared identity is a review snapshot only; never reuse it as launch approval.

Prompt preparation
  prepare_prompts.py uses only the actual offline tokenizer, USE_TORCH=0,
  enable_thinking=False, add_generation_prompt=True and the first two existing
  DAPO train.jsonl messages. No truncation, invented question or answer substitution.
  Prepared rows have158 and101 prompt tokens. Source-row, dataset and tokenizer
  hashes plus exact token IDs are saved in dapo-first-two-nonthinking.json.
  CPU preparation observed torch_imported=False,npu_imported=False.

Request schedule and measurement
  For each of the two prompts: one unmeasured warm request, then two timed
  requests using identical input_ids/sampling options. Greedy, ignore_eos=True
  forces a fixed16/32-token performance workload. Every raw request and SSE
  event/response is retained. No outputs are eligible for training.
  return_logprob=True,logprob_start_len=0 forces full prompt logprob calculation
  and disables prefix cache reuse in the pinned implementation; actual cached_tokens
  must equal0. Thus warm repeats do not silently become cached-prefix-only tests.
  Stream interval1, metrics enabled. Save all actual meta_info including queue,
  prefill timing where available, decode throughput, prompt/completion counts.
  Client TTFT is recorded only when the first nonempty chunk contains one token;
  otherwise record first-chunk latency and its actual completion count. Client
  decode rate uses only tokens arriving after that chunk, excluding prefill.
  HTTP timings include transport/queue/detokenization, not pure kernel time.
  Output logprobs must be finite and input logprobs aligned to original IDs;
  first input token may have None. Internal C4V8 tensor checks still execute.
  These checks do not claim every model tensor has been scanned.

Reuse and safety
  Reuses sg_smoke.py helper/source checks, HTTP health, PID/startticks/PGID
  verification. Pilot-local cleanup allows TERM at most5s, then rechecks identity
  before group KILL and waits at most3s, inside outer controller10s grace. It
  records cleanup.json, signal evidence and any forced-stop reason. A repeated
  controller TERM cannot interrupt this bounded critical cleanup section.
  Uses existing verify_backend_audit.py
  to require all8 scheduler ranks, both actual forward modes, real NPUMHA/paged
  pools, same source identity and HF32False. Parent HF32 readback checked too.
  Initial order is health -> genuine first warm generate -> server_info, avoiding
  the proven pinned receive-loop issue. No production SG source is changed.
  Real execution requires existing run-owned-probe.py as its independent parent.
  SIGTERM raises into finally, which stops only the verified owned SG group.
  Outer controller's after-all8-idle and no-fallback fields must also pass before
  accepting evidence. Inner completion alone is insufficient.
  Actual pinned /server_info uses flat dataclasses.asdict(ServerArgs) fields,
  verified against sg-fast-staged-v2/server-info.json: model_path,tp_size,
  context_length,max_total_tokens,chunked_prefill_size,max_running_requests.
  Evidence HTTP wrapper has status/path/body; those fields are in body, not a
  nested server_args object. Legacy /get_server_info aliases the same handler.
  Default total1200s/startup480s/per-request240s are bounds, not promised duration.
  Use an outer timeout with cleanup margin. No server is started by --prepare-only.

Prepared review
  /root/hby/experiments/c4v8-integration-probes/evidence/30b-pilot-prepared-v1/manifest.json
  contains actual CPU-only configuration and source identities at preparation.
  Final core may change before real execution; choose its final policy file and
  a fresh output directory so identity/sources are frozen anew at startup.
  Synthetic long prefixes were deliberately not added to this minimal pilot.

Future command, only after root grants all eight NPUs:
  /root/hby/envs/slime/bin/python /root/hby/experiments/c4v8-integration-probes/run-owned-probe.py \
    --output /root/hby/experiments/c4v8-integration-probes/evidence/owned-full30b-performance-v1 \
    --timeout-seconds 1260 -- /bin/bash /root/hby/experiments/c4v8_30b_sg_pilot/run-owned-command.sh \
    --output /root/hby/experiments/c4v8-integration-probes/evidence/owned-full30b-performance-v1/pilot \
    --policy FINAL_SHARED_POLICY_JSON \
    --prompts /root/hby/experiments/c4v8_30b_sg_pilot/dapo-first-two-nonthinking.json \
    --max-new-tokens 16 --max-seconds 1200 --startup-seconds 480 --request-seconds 240 \
    --execute-after-resource-gate
