Prepared performance probe. No NPU benchmark has been run by this preparation.

Purpose
  Measure the frozen fast_attention callable and actual query-tile training
  recomputation bridge at B1, Qheads8, KVheads1, D128, BF16. This is a single
  attention operation. It excludes backbone projections, MoE, PP/TP/HCCL,
  Adam, full transformer recompute, SG cache gather and end-to-end scheduling.
  Do not multiply these numbers into a claimed 100-step training duration.

Default cases
  Inference: cached prefix128,1024,4096,8192, one new query; total KV129,1025,
  4097,8193 and absolute query_offset equal to prefix length.
  Training: real bridge.recomputed_attention at1024 and4096, separate forward
  and backward timings. Backward includes the bridge replay and core STE replay.
  Diagnostic direct full-graph autograd only at1024. Direct4096 is excluded;
  its graph memory is not the production path.
  Explicit --train-lengths can select up to8192; defaults stay1024/4096.
  --query-block selects64(default),256,512,1024 with key128 unchanged.
  Source inventory and import paths include the staged decode runtime files.
  Optional --checkpoint-lengths adds reentrant torch checkpoint around the real
  bridge, to expose additional FA forward replay. It is explicitly not a full
  Megatron layer benchmark and is off by default.

Timings and bounds
  Each case records first observed forward/backward, one warmup, then3 measured
  samples. Each timed call is synchronized before and after, using wall time.
  First observed includes possible compilation/cache misses and actual execution.
  Existing /root/hby/cache/triton-c4v8 is NEVER cleared; JSON cache metadata
  deltas are recorded for first/measured calls. They are not exact compiler CPU
  times. A warm sample with new compilation metadata is flagged, not hidden.
  Input generation/H2D and extra result finite checks are outside FA timings;
  the callable's own internal checks/dispatch are included.
  NPU1 only, max5% HBM, HF32False, jit_compileFalse, OMP/MAX_COMPILE1.
  Failure at this5% probe allocation cap does not establish production OOM;
  report the precise phase and probe cap separately before making that inference.
  Soft budget540s refuses the next phase; only run-owned-probe.py provides the
  hard timeout600s and verified process-group cleanup. Native code may run past
  the soft deadline. Save/inspect completed smaller cases even on timeout.
  A full default plan is NOT promised to fit600s. Start smaller subsets and use
  the measured cost to choose a later root-approved window. No launch permission
  is implied by these prepared files.

Preparation evidence
  evidence/preparation-checks.json: argument bounds, exact case plan, cache delta
  tracking and actual CPU decode/direct/recomputed/checkpoint plumbing checks.
  evidence/cpu-smoke-v1/report.json contains CPU16 functional evidence only,
  never NPU throughput or a new numerical-equivalence acceptance claim.

Root-approved future execution example (DO NOT execute without the NPU lease)
  /root/hby/envs/slime/bin/python \
    /root/hby/experiments/c4v8-integration-probes/run-owned-probe.py \
    --output /root/hby/experiments/c4v8-integration-probes/evidence/performance-small-v1 \
    --timeout-seconds 600 -- \
    /bin/bash /root/hby/experiments/c4v8_performance_probe/run-owned-command.sh \
    --output /root/hby/experiments/c4v8-integration-probes/evidence/performance-small-v1/measurements \
    --train-lengths 128 --direct-lengths --decode-prefixes 128 1024 \
    --warmup 1 --repeats 2 --max-seconds 540
  Additional exact argv commands are in prepared-commands.txt.

Outputs
  plan.json, progress.json atomically updated before every allocation/phase,
  report.partial.json after forward (before backward) and each iteration,
  final report.json/exit, and exact source snapshots. All ordinary outputs and
  Q/K/V gradients must be finite; random representative gradients must be nonzero.
  Fresh output directories prevent accidental evidence overwrite.
  run-owned-probe outer.log/result.json own timeout/resource/fallback evidence.
  If outer result contains npu_cpu_fallback warnings, timing is not accepted as
  a device-only measurement even when this inner script returns0.
