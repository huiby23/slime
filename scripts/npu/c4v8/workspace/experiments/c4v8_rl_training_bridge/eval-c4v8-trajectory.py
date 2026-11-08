"""All 21 real C4V8 AIME points, immutable same-attempt resume, owned SG cleanup.

No NPU work occurs without a completed100-update manifest and fresh release
report. --validate-artifacts checks the prepared policy/protocol/client/parser
without constructing a model or starting an SG server.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from eval_support import configure_from_manifest, canonical, sha, verify_backend_audit, fresh_resource_gate
from eval_deadlines import EvaluationBudget, positive_seconds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--trajectory-dir', type=Path)
    parser.add_argument('--resources-released-report', type=Path)
    parser.add_argument('--base-sha-report', type=Path)
    parser.add_argument('--final-sha-report', type=Path)
    parser.add_argument('--server-start-timeout', type=positive_seconds, default=1800)
    parser.add_argument('--client-timeout', type=positive_seconds, default=14400,
                        help='Wall seconds allowed for each complete 30-question client process')
    parser.add_argument('--request-timeout', type=positive_seconds, default=3600,
                        help='HTTP timeout seconds for each question, forwarded to the CPU client')
    parser.add_argument('--total-timeout-seconds', type=positive_seconds,
                        help='Total wall budget including all 21 points and I/O; owned cleanup follows; default unlimited')
    parser.add_argument('--validate-artifacts', action='store_true')
    args = parser.parse_args(argv)
    if not args.validate_artifacts and any(value is None for value in (args.base_sha_report,args.final_sha_report)):
        parser.error('Actual evaluation requires base/final finite/SHA reports')
    return args


def evaluate(args, budget):
    manifest, experiment = configure_from_manifest(args.manifest, require_completed=not args.validate_artifacts)
    import c4v8_eval_protocol as protocol
    import c4v8_eval_inherited as inherited
    from c4v8_eval_runtime_support import server_payload
    actual, actual_sha = protocol.build_protocol()
    if actual != manifest['validation_protocol'] or actual_sha != manifest['validation_protocol_sha256']:
        raise ValueError('Evaluation numerical/data/grading/sampling policy differs from this training run')
    client = inherited.client_module()
    rows = client.load_data(inherited.DATA)
    payload = server_payload(os.getpid())
    inventory = manifest['attention_simulation']['source_sha256']
    if args.validate_artifacts:
        # Exact baseline grader cases + all30 dataset identities, no HTTP call.
        output = args.manifest.parent/'c4v8-evaluation-artifacts-validation.json'
        if output.exists(): raise FileExistsError('Preserve previous validation evidence')
        reward_output = output.with_name('c4v8-evaluation-reward-preflight.json')
        if reward_output.exists(): raise FileExistsError('Preserve previous grader evidence')
        client.self_test(reward_output)
        result = dict(passed=True, mode='CPU artifacts only; no model, SG or NPU',
            manifest=str(args.manifest.resolve()), experiment_identity_sha256=canonical(experiment),
            policy_sha256=os.environ['HBY_C4V8_POLICY_SHA256'], protocol_sha256=actual_sha,
            server_payload=payload, dataset_questions=len(rows), expected_steps=list(protocol.STEPS),
            actual_optimizer_steps=0, completed_validation_points=0,
            reward_preflight_sha256=sha(reward_output))
        protocol.atomic_json(output,result)
        print('C4V8_EVAL_ARTIFACTS_VALIDATED='+str(output),flush=True)
        return 0

    manifest, models = inherited.plan_models(args.manifest)
    # Fixed parent directory is allowed, but each training attempt owns its own
    # child. No other attempt's step number can silently join this trajectory.
    declared_root = Path(manifest['validation_trajectory_directory']).resolve()
    root = (args.trajectory_dir or declared_root/manifest['run_id']).resolve()
    if root.parent != declared_root or root.name != manifest['run_id']:
        raise ValueError('Trajectory must be the declared parent/run_id child for this exact attempt')
    root.mkdir(parents=True, exist_ok=True)
    marker = root/'training-experiment-identity.json'
    if marker.exists():
        if json.loads(marker.read_text()) != experiment:
            raise ValueError('Existing curve belongs to a different manifest, policy, source or checkpoint lineage')
    else:
        if any(root.iterdir()): raise ValueError('Refusing to adopt an unowned nonempty trajectory directory')
        protocol.atomic_json(marker,experiment)
    # Full bytes are hashed. Identical HF index JSON is not proof that two
    # optimizer-step weight tensors match. This CPU I/O may take significant time.
    identities = {}
    existing = []
    pending = []
    for step,model in models.items():
        point_dir = root/f'step_{step:06d}'
        if point_dir.exists():
            identities[step] = client.weight_identity(model,args.base_sha_report if step==0 else
                                args.final_sha_report if step==100 else None)
            point = inherited.validate_point(client,root,model,step,rows)
            summary = json.loads((point_dir/'summary.json').read_text())
            if summary['weight_identity'] != identities[step]:
                raise ValueError('Existing point weights/SHA differ from the current same-lineage checkpoint')
            existing.append(point)
        else: pending.append((step,model))
    if not pending:
        curve = protocol.rebuild_curve(root)
        if not curve['complete']: raise ValueError('Incomplete existing curve')
        print('C4V8_EVAL_ALL21_ALREADY_VALIDATED='+str(root),flush=True)
        return 0
    attempt = root/('execution_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    attempt.mkdir()
    # Refresh real resources after resume hashing. No timestamp speculation or
    # stopping another job to make room, and no imported accelerator runtime.
    gate = fresh_resource_gate(attempt/'resources-before-server.json',manifest,inherited.proc)
    if args.resources_released_report:
        gate['parent_report_path'] = str(args.resources_released_report.resolve())
        gate['parent_report_sha256'] = sha(args.resources_released_report)
    os.environ['HBY_C4V8_ROLLOUT_AUDIT_DIR'] = str(attempt/'backend-audit')
    protocol.atomic_json(attempt/'server-stdin.json',payload)
    evidence = dict(status='running', complete=False, experiment=experiment,
        experiment_identity_sha256=canonical(experiment), expected_steps=list(protocol.STEPS),
        resources_released=gate, started_utc=inherited.utc(), reused_complete_points=existing,
        points=list(existing), reloads=[], planned_weight_identities={str(k):v for k,v in identities.items()},
        time_budgets=dict(request_seconds=args.request_timeout, point_seconds=args.client_timeout,
                         startup_seconds=args.server_start_timeout, total_seconds=args.total_timeout_seconds,
                         total_scope='Entire evaluator invocation including hashing; bounded owned cleanup follows'),
        source_sha256={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    protocol.atomic_json(attempt/'trajectory-summary.json',evidence)
    process = identity = None
    started = time.monotonic()
    try:
        with budget.defer_termination():
            process,identity = inherited.launch_server(attempt)
        evidence['server_initial_model_info'] = inherited.wait_ready(client,process,identity,args.server_start_timeout)
        verify_backend_audit(attempt/'backend-audit',inventory,require_forward=False)
        for step,model in pending:
            inherited.verify_server_identity(process,identity)
            identities[step] = client.weight_identity(model,args.base_sha_report if step==0 else
                                args.final_sha_report if step==100 else None)
            if step: evidence['reloads'].append(inherited.reload_model(client,model,step))
            point_started = time.monotonic()
            inherited.run_point_client(root,model,step,args.base_sha_report if step==0 else
                                       args.final_sha_report if step==100 else None,args.client_timeout,
                                       request_timeout=args.request_timeout,budget=budget)
            point = inherited.validate_point(client,root,model,step,rows)
            summary = json.loads((root/f'step_{step:06d}'/'summary.json').read_text())
            if summary['weight_identity'] != identities[step]: raise ValueError('Checkpoint changed during evaluation')
            point.update(truncated=summary['truncated'], truncation_rate=summary['truncated']/30,
                         elapsed_seconds=time.monotonic()-point_started)
            point['backend_audit'] = verify_backend_audit(attempt/'backend-audit',inventory)
            evidence['points'].append(point)
            protocol.atomic_json(attempt/'trajectory-summary.json',evidence)
        curve = protocol.rebuild_curve(root)
        if not curve['complete'] or len(evidence['points'])!=21: raise RuntimeError('Need all21 measured points')
        evidence.update(status='completed',complete=True)
    except BaseException as error:
        evidence.update(status='failed',complete=False,error=dict(type=type(error).__name__,message=str(error)))
        print('C4V8_EVAL_FAILED='+type(error).__name__+': '+str(error),flush=True)
    finally:
        if process is not None:
            with budget.owned_cleanup():
                release = inherited.stop_server(attempt,process,identity)
            evidence['owned_server_release'] = release
            if not release['passed']: evidence.update(status='failed',complete=False)
        evidence.update(finished_utc=inherited.utc(),elapsed_seconds=time.monotonic()-started,
                        total_invocation_elapsed_seconds=time.monotonic()-budget.started)
        protocol.atomic_json(attempt/'trajectory-summary.json',evidence)
    code = 0 if evidence['status']=='completed' else 1
    (attempt/'trajectory.exit').write_text(str(code)+'\n')
    print(json.dumps(dict(status=evidence['status'],points=len(evidence['points']),root=str(root))),flush=True)
    return code


def main(argv=None):
    args = parse_args(argv)
    with EvaluationBudget(args.total_timeout_seconds) as budget:
        return evaluate(args,budget)


if __name__ == '__main__': raise SystemExit(main())
