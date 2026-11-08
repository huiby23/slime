#!/root/hby/envs/slime/bin/python
"""Independent every5 CPU client: 30 fresh AIME answers at each explicit step."""
import os
os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from c4v8_eval_protocol import build_protocol, rebuild_curve, write_summary, valid_point_exists, STEPS, validate_execution

REPO = Path('/root/hby/slime')
DEFAULT_DATA = Path('/root/hby/data/aime-2024/eval.jsonl')
DEFAULT_TOKENIZER = Path('/root/hby/models/Qwen3-30B-A3B')
DATA_SHA = '76a1ecaab458f041f89f43fd98f882ec69c4d16b60bb14d38c003daf22e0d36a'
SAMPLING = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'max_new_tokens': 4096,
            'stop': None, 'stop_token_ids': None, 'skip_special_tokens': False,
            'no_stop_trim': True, 'spaces_between_special_tokens': False}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_reward():
    path = REPO / 'slime/rollout/rm_hub/math_utils.py'
    spec = importlib.util.spec_from_file_location('aime_actual_math_rm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dispatcher = REPO / 'slime/rollout/rm_hub/__init__.py'
    if 'return 1 if grade_answer_verl(response, label) else 0' not in dispatcher.read_text():
        raise ValueError('Current native math RM dispatcher differs; review before evaluation')
    evidence = {'rm_type': 'math', 'math_utils_path': str(path), 'math_utils_sha256': sha(path),
                'dispatcher_path': str(dispatcher), 'dispatcher_sha256': sha(dispatcher),
                'function': 'grade_answer_verl(response, label)', 'custom_rm_path': None}
    return module, evidence


def load_data(path):
    if sha(path) != DATA_SHA:
        raise ValueError('Evaluation file differs from the fixed complete AIME2024 dataset')
    with path.open() as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if len(rows) != 30 or {row['metadata']['source_index'] for row in rows} != set(range(30)):
        raise ValueError('All 30 original AIME2024 questions must be evaluated, without selection')
    for row in rows:
        if (row['metadata']['dataset'] != 'aime-2024' or row['metadata']['rm_type'] != 'math'
                or not isinstance(row['label'], str) or not isinstance(row['messages'], list)):
            raise ValueError('Unexpected evaluation schema or reward metadata')
    return rows


def http_json(url, payload=None, timeout=1800):
    request = urllib.request.Request(url, method='GET' if payload is None else 'POST',
              data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
              headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read().decode('utf-8')
    try:
        parsed = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON literal: ' + value)))
    except Exception as error:
        error.response_raw_text = raw
        raise
    return parsed, raw


def check_server(info, model_path, tokenizer_path, version):
    if not isinstance(info, dict) or Path(info.get('model_path', '')).resolve() != model_path:
        raise ValueError('Server /model_info model_path does not match this curve point')
    if Path(info.get('tokenizer_path') or '').resolve() != tokenizer_path:
        raise ValueError('Server tokenizer_path must be the unchanged original tokenizer')
    if str(info.get('weight_version')) != version:
        raise ValueError('Server weight_version does not match the explicit evaluation version')


def weight_identity(model_path, report_path):
    index_path = model_path / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(not (model_path / filename).is_file() for filename in shards):
        raise ValueError('Model is missing indexed weight shards')
    identity = {'model_path': str(model_path), 'index_path': str(index_path), 'index_sha256': sha(index_path),
                'indexed_shards': len(shards), 'full_weight_sha256_verified_by_report': False,
                'verification_scope': 'index SHA256 and existing indexed shard files; full shard hashes not supplied'}
    if report_path is not None:
        report = json.loads(report_path.read_text())
        if (not report.get('passed') or not report.get('complete_scan')
                or Path(report.get('hf_dir', '')).resolve() != model_path
                or report.get('index_sha256') != identity['index_sha256']
                or report.get('nonfinite_count') != 0
                or {row['file'] for row in report.get('shards', [])} != set(shards)):
            raise ValueError('Full-weight finite/SHA report does not match this completed model')
        for row in report['shards']:
            if (len(row['sha256']) != 64 or (model_path / row['file']).stat().st_size != row['file_bytes']):
                raise ValueError('Weight shard size/SHA reference mismatch')
        identity.update(full_weight_sha256_verified_by_report=True,
                        verification_scope='full finite/SHA scanner report; files treated as immutable after that scan',
                        full_weight_report=str(report_path), full_weight_report_sha256=sha(report_path),
                        shard_sha256_references=[{key: row[key] for key in ['file', 'file_bytes', 'sha256']}
                                                 for row in report['shards']])
    from eval_support import full_weight_identity
    return full_weight_identity(identity, model_path)


def parse_completion(output, expected_version):
    if not isinstance(output, dict) or not isinstance(output.get('text'), str):
        raise ValueError('Native /generate response must contain text and meta_info')
    meta = output.get('meta_info')
    if not isinstance(meta, dict) or str(meta.get('weight_version')) != expected_version:
        raise ValueError('Completion weight_version missing or changed during evaluation')
    finish = meta.get('finish_reason')
    if not isinstance(finish, dict) or finish.get('type') not in ('stop', 'length'):
        raise ValueError('Generation aborted/failed or has an unknown finish_reason')
    values = meta.get('output_token_logprobs')
    if not isinstance(values, list) or not values:
        raise ValueError('Native output token/logprob evidence is missing')
    token_ids = []
    for value in values:
        if (not isinstance(value, list) or len(value) < 2 or not isinstance(value[1], int)
                or isinstance(value[1], bool) or value[1] < 0
                or not isinstance(value[0], (int, float)) or not math.isfinite(value[0])):
            raise ValueError('Invalid/nonfinite native output token/logprob evidence')
        token_ids.append(value[1])
    if meta.get('completion_tokens') != len(token_ids) or len(token_ids) > SAMPLING['max_new_tokens']:
        raise ValueError('Native completion_tokens differs from recorded output token count')
    return meta, token_ids, finish.get('type') == 'length'




def evaluate(args):
    if args.step not in STEPS or args.concurrency != 4 or args.resume:
        raise ValueError('Every5 requires steps0..100 stride5, concurrency4, and no retries/resume')
    args.server_url = args.server_url.rstrip('/')
    model_path, tokenizer_path = args.model_path.resolve(), args.tokenizer.resolve()
    rows = load_data(args.eval_data.resolve())
    reward, reward_evidence = load_reward()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if any(name in sys.modules for name in ['torch', 'torch_npu', 'ray', 'sglang']):
        raise RuntimeError('HTTP evaluation client must remain tokenizer/CPU only')
    tokenizer_hashes = {name: sha(tokenizer_path / name) for name in ['tokenizer.json', 'tokenizer_config.json']}
    server_settings, _ = http_json(args.server_url + '/server_info', timeout=min(args.timeout, 60))
    if not isinstance(server_settings, dict) or server_settings.get('tp_size') != 8:
        raise ValueError('This protocol requires the standalone TP8 SGLang server')
    settings = {key: server_settings.get(key) for key in [
        'device', 'tp_size', 'attention_backend', 'sampling_backend', 'disable_cuda_graph',
        'context_length', 'max_running_requests', 'max_total_tokens', 'chunked_prefill_size', 'mem_fraction_static',
        'prefill_attention_backend', 'decode_attention_backend', 'dtype', 'kv_cache_dtype']}
    protocol, protocol_sha = build_protocol(
        tokenizer_path=tokenizer_path, eval_data_path=args.eval_data.resolve(), sampling_params=SAMPLING,
        enable_thinking=False, backend={key: server_settings.get(key) for key in [
            'device', 'tp_size', 'attention_backend', 'sampling_backend', 'dtype', 'disable_cuda_graph']})
    execution_settings = {'http_max_concurrency': args.concurrency, 'server_settings': settings,
                          'transport': 'SGLang native HTTP /generate input_ids + return_logprob:true',
                          'tokenizer_path': str(tokenizer_path), 'dataset_path': str(args.eval_data.resolve()),
                          'reward_source_paths': reward_evidence}
    validate_execution(execution_settings)
    if any(server_settings.get(k) != 'hby_c4v8_sim' for k in
           ('attention_backend','prefill_attention_backend','decode_attention_backend')):
        raise ValueError('Every actual server selector must use the training C4V8 policy')
    version = args.expected_weight_version if args.expected_weight_version is not None else str(args.step)
    identity = weight_identity(model_path, args.weight_sha_report.resolve() if args.weight_sha_report else None)
    info_before, _ = http_json(args.server_url + '/model_info', timeout=min(args.timeout, 60))
    check_server(info_before, model_path, tokenizer_path, version)
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if valid_point_exists(root, args.step):
        raise FileExistsError('An existing valid point is immutable; use the curve subcommand to rebuild aggregates')
    point_dir = root / f'step_{args.step:06d}'; questions_dir = point_dir / 'questions'; questions_dir.mkdir(parents=True, exist_ok=True)
    point = {'optimizer_step': args.step, 'model_path': str(model_path), 'weight_version': version,
             'weight_identity': identity, 'protocol_sha256': protocol_sha, 'protocol': protocol,
             'execution_settings': execution_settings,
             'experiment_identity_sha256': os.environ['HBY_C4V8_EXPERIMENT_SHA256']}
    point_sha = canonical_sha(point)
    metadata_path = point_dir / 'protocol.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if not args.resume or previous['point_sha256'] != point_sha:
            raise ValueError('Point directory already exists; use --resume only with exactly identical model/protocol/weight references')
    atomic_json(metadata_path, {**point, 'point_sha256': point_sha, 'server_before': info_before, 'started_utc': utc()})
    atomic_json(point_dir / 'server-info.json', server_settings)

    def request_question(index):
        row = rows[index]; path = questions_dir / f'{index:02d}.json'
        if args.resume and path.exists():
            cached = json.loads(path.read_text())
            if cached.get('point_sha256') == point_sha and cached.get('status') == 'success':
                return cached
        text = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        payload = {'input_ids': prompt_ids, 'sampling_params': dict(SAMPLING), 'return_logprob': True}
        result = {'question_index': index, 'sample_id': row['metadata']['sample_id'], 'dataset_metadata': row['metadata'],
                  'label': row['label'], 'messages': row['messages'], 'rendered_prompt': text,
                  'request': payload, 'point_sha256': point_sha, 'optimizer_step': args.step,
                  'model_path': str(model_path), 'expected_weight_version': version, 'started_utc': utc(),
                  'experiment_identity_sha256': os.environ['HBY_C4V8_EXPERIMENT_SHA256'],
                  'status': 'error', 'math_score': None}
        started = time.monotonic()
        try:
            output, raw = http_json(args.server_url + '/generate', payload, args.timeout)
            result.update(response_raw_text=raw, response_json=output)
            meta, token_ids, truncated = parse_completion(output, version)
            score = 1 if reward.grade_answer_verl(output['text'], row['label']) else 0
            result.update(status='success', response_text=output['text'], extracted_answer=reward.extract_answer(output['text']),
                          output_token_ids=token_ids, completion_tokens=len(token_ids), finish_reason=meta['finish_reason'],
                          actual_weight_version=str(meta['weight_version']), truncated=truncated, math_score=score)
        except urllib.error.HTTPError as error:
            result.update(error=type(error).__name__ + ': ' + str(error), http_status=error.code,
                          response_raw_text=error.read().decode('utf-8', errors='replace'))
        except Exception as error:
            result['error'] = type(error).__name__ + ': ' + str(error)
            if hasattr(error, 'response_raw_text'):
                result['response_raw_text'] = error.response_raw_text
        result.update(completed_utc=utc(), elapsed_seconds=round(time.monotonic() - started, 3))
        atomic_json(path, result)
        print(json.dumps({key: result.get(key) for key in ['question_index', 'status', 'math_score', 'completion_tokens',
                                                          'truncated', 'elapsed_seconds', 'error']}), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(request_question, range(30)))
    identity_error = None; info_after = None
    try:
        info_after, _ = http_json(args.server_url + '/model_info', timeout=min(args.timeout, 60))
        check_server(info_after, model_path, tokenizer_path, version)
    except Exception as error:
        identity_error = type(error).__name__ + ': ' + str(error)
    successful = [result for result in results if result['status'] == 'success']
    valid = len(successful) == 30 and identity_error is None
    correct = sum(result['math_score'] for result in successful)
    summary = {**point, 'point_sha256': point_sha, 'completed_utc': utc(), 'valid_curve_point': valid,
               'correct': correct, 'questions': 30, 'successful_requests': len(successful),
               'errors': 30 - len(successful), 'truncated': sum(result.get('truncated', False) for result in successful),
               'accuracy': correct / 30 if valid else None, 'accuracy_percent': correct / 30 * 100 if valid else None,
               'server_before': info_before, 'server_after': info_after, 'model_identity_error': identity_error,
               'cpu_only_client': True, 'curve_scope': 'All 30 successful native responses required; truncation is reported and scored by unchanged math RM.'}
    temporary = point_dir / 'results.jsonl.partial'
    with temporary.open('w') as file:
        for result in results: file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(point_dir / 'results.jsonl')
    write_summary(root, summary)
    print(json.dumps({key: summary[key] for key in ['optimizer_step', 'valid_curve_point', 'correct', 'questions',
                                                  'accuracy_percent', 'errors', 'truncated']}), flush=True)
    return 0 if valid else 1


def self_test(output):
    reward, evidence = load_reward()
    cases = [('integer_correct', '73', '\\boxed{73}', 1), ('integer_wrong', '73', '\\boxed{74}', 0),
             ('unboxed_wrong', '73', '73', 0), ('fraction_equivalent', '\\frac{1}{2}', '\\boxed{0.5}', 1),
             ('fraction_wrong', '\\frac{1}{2}', '\\boxed{\\frac{2}{3}}', 0),
             ('radical_correct', '\\sqrt{2}', '\\boxed{\\sqrt{2}}', 1),
             ('radical_wrong', '\\sqrt{2}', '\\boxed{\\sqrt{3}}', 0), ('negative_correct', '-7', '\\boxed{-7}', 1)]
    results = []
    for name, label, response, expected in cases:
        score = 1 if reward.grade_answer_verl(response, label) else 0
        if score != expected: raise AssertionError((name, score, expected))
        results.append({'case': name, 'label': label, 'response': response, 'score': score, 'passed': True})
    report = {'passed': True, 'case_count': len(cases), 'cases': results, 'reward': evidence,
              'full_fixed_eval_dataset_rows': len(load_data(DEFAULT_DATA)), 'actual_http_or_generation_performed': False,
              'torch_ray_sglang_imported': any(name in sys.modules for name in ['torch', 'torch_npu', 'ray', 'sglang'])}
    atomic_json(output, report); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    test = commands.add_parser('self-test'); test.add_argument('--output', type=Path, required=True)
    curve = commands.add_parser('curve'); curve.add_argument('--output-root', type=Path, required=True)
    run = commands.add_parser('eval')
    run.add_argument('--server-url', required=True); run.add_argument('--model-path', type=Path, required=True)
    run.add_argument('--step', type=int, required=True); run.add_argument('--output-root', type=Path, required=True)
    run.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER); run.add_argument('--eval-data', type=Path, default=DEFAULT_DATA)
    run.add_argument('--weight-sha-report', type=Path); run.add_argument('--expected-weight-version')
    run.add_argument('--concurrency', type=int, choices=[4], default=4)
    run.add_argument('--timeout', type=float, default=3600); run.set_defaults(resume=False)
    args = parser.parse_args()
    if args.command == 'self-test': return self_test(args.output)
    if args.command == 'curve': rebuild_curve(args.output_root.resolve()); return 0
    if args.step not in STEPS or args.timeout <= 0: parser.error('step must be0..100 stride5 and timeout positive')
    return evaluate(args)


if __name__ == '__main__':
    raise SystemExit(main())
