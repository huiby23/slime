"""Shared, stdlib-only AIME trajectory identity and atomic summary writer.

Safe to import inside native training: no environment changes or Torch imports.
Execution memory limits, batching, transport and paths are point metadata;
the protocol identity covers the fixed questions/tokenizer/grader/math settings.
"""
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path

BASE = Path('/root/hby/models/Qwen3-30B-A3B')
DATA = Path('/root/hby/data/aime-2024/eval.jsonl')
REPO = Path('/root/hby/slime')
DATA_SHA = '76a1ecaab458f041f89f43fd98f882ec69c4d16b60bb14d38c003daf22e0d36a'
SAMPLING = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'max_new_tokens': 4096,
            'stop': None, 'stop_token_ids': None, 'skip_special_tokens': False,
            'no_stop_trim': True, 'spaces_between_special_tokens': False}
BACKEND = {'device': 'npu', 'tp_size': 8, 'attention_backend': 'hby_c4v8_sim',
           'sampling_backend': 'pytorch', 'dtype': 'bfloat16', 'disable_cuda_graph': True}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalize_sampling(sampling_params=None):
    result = dict(SAMPLING)
    supplied = dict(sampling_params or {})
    for key, value in supplied.items():
        if key not in SAMPLING:
            if value is not None:
                raise ValueError('Extra sampling setting would change the shared protocol: ' + key)
            continue
        result[key] = value
    for key in ['temperature', 'top_p']:
        result[key] = float(result[key])
    for key in ['top_k', 'max_new_tokens']:
        result[key] = int(result[key])
    if result != SAMPLING:
        raise ValueError('Sampling differs from fixed AIME greedy/non-thinking protocol')
    return result


def build_protocol(tokenizer_path=BASE, eval_data_path=DATA, sampling_params=None,
                   enable_thinking=False, backend=None):
    """Return (protocol_dict, sha256); verify actual tokenizer equals original.

    Paths deliberately do not enter the identity. Native parent20 tokenizer
    files must really hash identically to BASE, rather than merely declaring it.
    """
    tokenizer_path, eval_data_path = Path(tokenizer_path).resolve(), Path(eval_data_path).resolve()
    if sha(eval_data_path) != DATA_SHA:
        raise ValueError('Shared trajectory requires the exact complete AIME2024-30 file')
    with eval_data_path.open() as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if len(rows) != 30 or {row['metadata']['source_index'] for row in rows} != set(range(30)):
        raise ValueError('Shared trajectory requires all 30 unfiltered original questions')
    if enable_thinking is not False:
        raise ValueError('Shared trajectory requires enable_thinking=False')
    actual = {name: sha(tokenizer_path / name) for name in ['tokenizer.json', 'tokenizer_config.json']}
    original = {name: sha(BASE / name) for name in actual}
    if actual != original:
        raise ValueError('Actual tokenizer files differ from the original tokenizer')
    settings = dict(BACKEND)
    settings.update(backend or {})
    if settings != BACKEND:
        raise ValueError('Backend/precision differs from this C4V8 forward-QDQ TP8 protocol')
    math_path = REPO / 'slime/rollout/rm_hub/math_utils.py'
    dispatcher = REPO / 'slime/rollout/rm_hub/__init__.py'
    if 'return 1 if grade_answer_verl(response, label) else 0' not in dispatcher.read_text():
        raise ValueError('Native math RM dispatcher changed')
    protocol = {'protocol_version': 'aime2024-30-greedy-c4v8-forward-qdq-ste-tp8-v1',
                'dataset_sha256': DATA_SHA, 'questions': 30, 'filtered_questions': 0,
                'samples_per_question': 1, 'sampling_params': normalize_sampling(sampling_params),
                'enable_thinking': False, 'add_generation_prompt': True, 'tokenizer_add_special_tokens': False,
                'tokenizer_sha256': actual, 'reward': {'rm_type': 'math',
                    'math_utils_sha256': sha(math_path), 'dispatcher_sha256': sha(dispatcher),
                    'function': 'grade_answer_verl(response, label)', 'custom_rm_path': None},
                'backend': settings}
    policy_path = Path(os.environ['HBY_C4V8_POLICY_FILE'])
    policy_value = json.loads(policy_path.read_text())
    policy_sha = canonical_sha(policy_value)
    if policy_sha != os.environ.get('HBY_C4V8_POLICY_SHA256'):
        raise ValueError('The evaluation policy file differs from the declared training policy')
    from numerical_profile import implementation_profile, digest
    implementation = implementation_profile(BASE.parent.parent, os.environ['HBY_C4V8_CORE_MODULE'])
    if (implementation['implementation_id'] != os.environ.get('HBY_C4V8_IMPLEMENTATION_ID')
            or digest(implementation) != os.environ.get('HBY_C4V8_IMPLEMENTATION_SHA256')):
        raise ValueError('Numerical implementation differs from the declared training implementation')
    protocol['attention_simulation'] = {
        'label': 'NVFP4/MXFP8 forward QDQ with high-precision STE backward',
        'policy': policy_value, 'policy_sha256': policy_sha,
        'implementation': implementation, 'implementation_sha256': digest(implementation),
        'forward_source_sha256': implementation['source_sha256'],
        'backbone_dtype': 'bfloat16', 'quantized_backward': False, 'fp32_matmul_allow_hf32': False}
    return protocol, canonical_sha(protocol)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def valid_point_exists(root, step):
    path = Path(root) / f'step_{step:06d}' / 'summary.json'
    return path.exists() and bool(json.loads(path.read_text()).get('valid_curve_point'))


def validate_summary(summary):
    if (not isinstance(summary.get('optimizer_step'), int) or summary['optimizer_step'] < 0
            or summary.get('questions') != 30 or summary.get('protocol_sha256') != canonical_sha(summary['protocol'])):
        raise ValueError('Invalid optimizer step, denominator or protocol identity')
    if not isinstance(summary.get('correct'), int) or not 0 <= summary['correct'] <= 30:
        raise ValueError('Invalid correct-answer count')
    if not summary.get('model_path') or len(summary['weight_identity']['index_sha256']) != 64:
        raise ValueError('Missing explicit model/weight identity')
    if summary.get('valid_curve_point'):
        if summary.get('successful_requests') != 30 or summary.get('errors') != 0:
            raise ValueError('A valid point requires 30 successful responses and zero errors')
        if (not math.isclose(summary['accuracy'], summary['correct'] / 30, abs_tol=1e-12)
                or not math.isclose(summary['accuracy_percent'], summary['correct'] / 30 * 100, abs_tol=1e-12)):
            raise ValueError('Accuracy must equal actual correct/30')
    elif summary.get('accuracy') is not None or summary.get('accuracy_percent') is not None:
        raise ValueError('Incomplete/error points must have null accuracy')


def write_summary(root, summary):
    root = Path(root)
    validate_summary(summary)
    if summary.get('valid_curve_point'):
        for other_path in root.glob('step_*/summary.json'):
            other = json.loads(other_path.read_text())
            if other.get('valid_curve_point') and other['protocol_sha256'] != summary['protocol_sha256']:
                raise ValueError('Refusing to commit a point with a different shared evaluation protocol')
    directory = root / f"step_{summary['optimizer_step']:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'summary.json'
    if path.exists() and json.loads(path.read_text()).get('valid_curve_point'):
        raise FileExistsError('Refusing to overwrite an existing valid evaluation point: ' + str(path))
    atomic_json(path, summary)
    return rebuild_curve(root)


def rebuild_curve(root):
    root = Path(root)
    points = [json.loads(path.read_text()) for path in sorted(root.glob('step_*/summary.json'))]
    for point in points:
        validate_summary(point)
    points.sort(key=lambda point: point['optimizer_step'])
    valid = [point for point in points if point.get('valid_curve_point')]
    if len({point['protocol_sha256'] for point in valid}) > 1:
        raise ValueError('Refusing to merge different question/tokenizer/grader/sampling/backend protocols')
    curve = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'dataset': 'AIME2024 complete 30', 'denominator': 30,
             'valid_points': [{'optimizer_step': point['optimizer_step'], 'correct': point['correct'],
                               'accuracy': point['accuracy'], 'accuracy_percent': point['accuracy_percent'],
                               'truncated': point['truncated'], 'model_path': point['model_path'],
                               'summary_path': str(root / f"step_{point['optimizer_step']:06d}" / 'summary.json')}
                              for point in valid],
             'all_point_summaries': points,
             'scope': 'Only actual completed all-30 evaluations; execution memory/batching/transport recorded separately; no interpolation or fabricated optimizer steps.'}
    atomic_json(root / 'accuracy-curve.json', curve)
    stream = io.StringIO(); writer = csv.writer(stream)
    writer.writerow(['optimizer_step', 'valid_curve_point', 'correct', 'questions', 'accuracy', 'accuracy_percent',
                     'successful_requests', 'errors', 'truncated', 'model_path', 'weight_index_sha256', 'full_weight_sha_report'])
    for point in points:
        writer.writerow([point['optimizer_step'], point['valid_curve_point'], point['correct'], 30,
                         point['accuracy'], point['accuracy_percent'], point['successful_requests'], point['errors'],
                         point['truncated'], point['model_path'], point['weight_identity']['index_sha256'],
                         point['weight_identity'].get('full_weight_report', '')])
    temporary = root / 'accuracy-curve.csv.partial'
    temporary.write_text(stream.getvalue()); temporary.replace(root / 'accuracy-curve.csv')
    return curve
