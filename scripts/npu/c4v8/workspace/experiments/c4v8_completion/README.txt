Independent one-shot completion pipeline for C4V8 run407f37

Prepared plan (not executed by preparation):
/root/hby/experiments/c4v8_completion/qwen30b-407f37-v2/plan.json
v1 is preserved as superseded and was never executed.

Root-reviewed execution command:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /root/hby/envs/slime/bin/python -u \
 /root/hby/experiments/c4v8_completion/run_pipeline.py \
 --execute /root/hby/experiments/c4v8_completion/qwen30b-407f37-v2/plan.json

It opens only this fixed controller's pidfd and blocks until exit. The pinned
Python lacks os.pidfd_open, so a checked Linux aarch64/x86_64 syscall434 fallback
is available; this kernel path passed an actual CPU-process exit test. There is
no cron/app automation, scheduled polling, training restart, or notification.

The immutable plan records controller PID/startticks/PGID/argv, canonical pointer
SHA, training run/config identity, accepted gate SHA, source hashes and full
source snapshots. A changed identity/source causes failure. A fresh execution/
directory is the atomic one-shot claim; repeated --execute refuses to overwrite.

After the controller exits, the worker requires child exit0, completed100-step
manifest,100 positive finite gradient files, complete HF100 successful-update
metadata and all20 every5 checkpoints. It performs a fresh all8-idle/private-Ray
released/port15115-free/host-memory gate before scans, and again before evaluation.
It never stops another task to make room. A failed training run ends the pipeline.

Stages: sampled real changes vs officialbase; complete finite/SHA scans of the
published model, checkpoint100 and base; exact published/checkpoint100 equality;
21 actual same-policy AIME30 points; verification of per-question evidence; plots.
Valid existing scan reports can be reused only for exactly the same hf_dir with
actual index/all-shard bytes matching. Invalid/conflicting reports fail closed
and are preserved. Equal shard hashes provide a fast export-equivalence proof;
different serialization falls back to every tensor's dtype/shape/full-byte SHA.
No sampled comparison substitutes for this full export-equivalence check.

Evaluation uses1800s startup,10800s/question,43200s/30-question point,604800s total.
The evaluator's own deadline/TERM handler cleans its owned SG session. The outer
worker allows180s extra and gives TERM120s grace. An evaluator that still cannot
exit is recorded as failed/cleanup_error; its ownership evidence is preserved
for explicit operator cleanup, and no subsequent stage starts. No broad kill is
used. CPU-stage timeouts can kill only their verified independent child group.

The CPU-only child environment cannot change SG's initialization: the frozen
run-c4v8-eval-server.sh unsets TORCH_DEVICE_BACKEND_AUTOLOAD/USE_TORCH/USE_TF before
the shim, and c4v8_bootstrap explicitly imports torch_npu in each real process.

Reports:
  plan.json                         status prepared, execution_started false
  execution/completion.json         waiting/accepting/evaluating/plotting/completed/failed
  execution/pipeline.exit           zero only for complete model+21 points+plots
  execution/STAGE/stage.json        exact command, owned identity, exit, timing
  execution/STAGE/stdout.log        full stage evidence; no environment dump
  execution/resources-*.json        actual resource gates
  execution/plots/                  PNG,SVG,PDF,CSV and plot-report.json

No BF16 comparison points are invented or copied from chat. The actual curve is
C4V8 forward simulation with a BF16 backbone and high-precision STE backward,
nonthinking, measured at every5 successful optimizer updates. The plot validates
21 points, one experiment/protocol, and all30 successful answers per point.

Each point additionally requires its30 raw boolean truncation flags to sum to
the exact reported summary count; an inconsistent aggregate is rejected.

CPU evidence: evidence/cpu-tests-v4.log,10/10 Linux tests. Rendering evidence in
evidence/cpu-plot-fixture-v1 is synthetic, prominently watermarked CPU FIXTURE—
NOT MODEL RESULTS, and reports actual_measured_points=0. It is never an experiment
result. Matplotlib/numpy live in the isolated plot-env; slime was not modified.
