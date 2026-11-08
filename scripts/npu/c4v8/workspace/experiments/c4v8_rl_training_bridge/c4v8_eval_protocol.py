"""Independent every-five-update trajectory; same fixed mathematical protocol.

The frozen builder remains authoritative. Only the allowed step grid and
trajectory writer are new. No accelerator/framework imports or environment edits.
"""
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import os
import importlib.util
_spec = importlib.util.spec_from_file_location('c4v8_eval_frozen', os.environ['HBY_C4V8_EVAL_PROTOCOL_MODULE'])
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)

FROZEN_SHA256 = os.environ['HBY_C4V8_EVAL_PROTOCOL_SOURCE_SHA256']
PROTOCOL_SHA256 = os.environ['HBY_C4V8_EVAL_PROTOCOL_SHA256']
STEPS = tuple(range(0, 101, 5))
canonical_sha = frozen.canonical_sha
sha = frozen.sha
atomic_json = frozen.atomic_json
normalize_sampling = frozen.normalize_sampling


def build_protocol(*args, **kwargs):
    if sha(Path(frozen.__file__)) != FROZEN_SHA256:
        raise ValueError('Frozen protocol source changed; review before new evaluation')
    protocol, identity = frozen.build_protocol(*args, **kwargs)
    if identity != PROTOCOL_SHA256:
        raise ValueError('Every5 evaluation must preserve the fixed mathematical protocol')
    return protocol, identity


def validate_summary(summary):
    frozen.validate_summary(summary)
    if type(summary['optimizer_step']) is not int or summary['optimizer_step'] not in STEPS:
        raise ValueError('Every5 trajectory requires integer steps 0,5,...,100')
    if summary['protocol_sha256'] != PROTOCOL_SHA256:
        raise ValueError('Unexpected fixed AIME protocol')
    if str(summary.get('weight_version')) != str(summary['optimizer_step']):
        raise ValueError('Every5 serving version must equal the explicit optimizer step')
    validate_execution(summary.get('execution_settings', {}))
    if summary.get('experiment_identity_sha256') != os.environ['HBY_C4V8_EXPERIMENT_SHA256']:
        raise ValueError('Evaluation point belongs to a different training attempt')


def validate_execution(execution):
    if execution.get('http_max_concurrency') != 4:
        raise ValueError('All every5 points require exactly 4 concurrent HTTP requests')
    settings = execution.get('server_settings', {})
    required = {'max_running_requests': 4, 'max_total_tokens': 12288,
                'mem_fraction_static': 0.4, 'context_length': 8192,
                'chunked_prefill_size': 256}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError('Every5 server execution profile changed')


def _ensure_root(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'trajectory-format-every5.json'
    value = {'format': 'aime2024-c4v8-every5-independent-v1', 'expected_optimizer_steps': list(STEPS),
             'protocol_sha256': PROTOCOL_SHA256, 'all_points_measured_after_training': True,
             'old_results_reused': False, 'experiment_identity_sha256': os.environ['HBY_C4V8_EXPERIMENT_SHA256']}
    if marker.exists():
        if json.loads(marker.read_text()) != value:
            raise ValueError('Trajectory directory belongs to a different experiment')
    else:
        if list(root.glob('step_*/summary.json')):
            raise ValueError('Refusing to adopt or merge existing old trajectory points')
        atomic_json(marker, value)
    return root


def valid_point_exists(root, step):
    if step not in STEPS:
        raise ValueError('Optimizer step is outside the every5 grid')
    path = Path(root) / f'step_{step:06d}' / 'summary.json'
    return path.exists() and bool(json.loads(path.read_text()).get('valid_curve_point'))


def write_summary(root, summary):
    validate_summary(summary)
    root = _ensure_root(root)
    path = root / f"step_{summary['optimizer_step']:06d}" / 'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Every5 points are never overwritten or retried: ' + str(path))
    for previous in root.glob('step_*/summary.json'):
        old = json.loads(previous.read_text()); validate_summary(old)
        if old['execution_settings'] != summary['execution_settings']:
            raise ValueError('All 21 points must share the exact same execution profile')
    atomic_json(path, summary)
    return rebuild_curve(root)


def rebuild_curve(root):
    root = _ensure_root(root)
    points = [json.loads(path.read_text()) for path in root.glob('step_*/summary.json')]
    for point in points:
        validate_summary(point)
    points.sort(key=lambda item: item['optimizer_step'])
    if len({point['optimizer_step'] for point in points}) != len(points):
        raise ValueError('Duplicate optimizer-step summaries')
    if points and any(point['execution_settings'] != points[0]['execution_settings'] for point in points):
        raise ValueError('Different execution profiles cannot share this curve')
    valid = [point for point in points if point.get('valid_curve_point')]
    curve = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'dataset': 'AIME2024 complete 30', 'denominator': 30,
             'expected_optimizer_steps': list(STEPS), 'expected_points': 21,
             'complete': [point['optimizer_step'] for point in valid] == list(STEPS),
             'valid_points': [{key: point[key] for key in ['optimizer_step', 'correct', 'accuracy',
                 'accuracy_percent', 'truncated', 'model_path']} | {
                 'summary_path': str(root / f"step_{point['optimizer_step']:06d}" / 'summary.json')}
                 for point in valid],
             'all_point_summaries': points,
             'scope': '21 fresh post-training measurements; all 30 questions per point; same execution profile; no interpolation or old-point reuse.'}
    atomic_json(root / 'accuracy-curve.json', curve)
    stream = io.StringIO(); writer = csv.writer(stream)
    writer.writerow(['optimizer_step', 'valid_curve_point', 'correct', 'questions', 'accuracy',
                     'accuracy_percent', 'successful_requests', 'errors', 'truncated', 'model_path',
                     'weight_index_sha256', 'full_weight_sha_report'])
    for point in points:
        writer.writerow([point[key] for key in ['optimizer_step', 'valid_curve_point', 'correct',
            'questions', 'accuracy', 'accuracy_percent', 'successful_requests', 'errors', 'truncated',
            'model_path']] + [point['weight_identity']['index_sha256'],
                             point['weight_identity'].get('full_weight_report', '')])
    temporary = root / 'accuracy-curve.csv.partial'
    temporary.write_text(stream.getvalue()); temporary.replace(root / 'accuracy-curve.csv')
    return curve
