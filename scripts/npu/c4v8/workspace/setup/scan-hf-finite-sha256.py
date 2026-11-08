#!/usr/bin/env python3
"""Scan completed indexed HF safetensors: full finite check + shard SHA256.

CPU/NumPy only; reads at most --block-mib (default 8, maximum 32) MiB at once.
Supported weight dtypes: BF16, F16, F32. No torch/Ray/NPU/model imports.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np

MIB = 1024 * 1024
FORMATS = {'BF16': (2, '<u2', 0x7F80), 'F16': (2, '<u2', 0x7C00),
           'F32': (4, '<u4', 0x7F800000)}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'Duplicate JSON key: {key!r}')
        value[key] = item
    return value


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'Nonstandard JSON constant {x}')))


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def read_index(root):
    path = root / 'model.safetensors.index.json'
    if not path.is_file() or path.stat().st_size > 16 * MIB:
        raise ValueError(f'Missing/oversized HF safetensors index: {path}')
    raw = path.read_bytes()
    index = parse_json(raw)
    if not isinstance(index, dict):
        raise ValueError('Index must be a JSON object')
    weights = index.get('weight_map')
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Index weight_map must be a nonempty object')
    shards = collections.defaultdict(set)
    for key, filename in weights.items():
        if not isinstance(key, str) or not key or key == '__metadata__':
            raise ValueError(f'Invalid index tensor key: {key!r}')
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith('.safetensors')):
            raise ValueError(f'Invalid shard filename: {filename!r}')
        path = root / filename
        if not path.is_file():
            raise ValueError(f'Missing indexed shard: {filename}')
        shards[filename].add(key)
    metadata = index.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('Index metadata must be an object')
    declared_bytes = metadata.get('total_size')
    if declared_bytes is not None and (not integer(declared_bytes) or declared_bytes < 0):
        raise ValueError('Index metadata.total_size must be a nonnegative integer')
    return shards, declared_bytes, hashlib.sha256(raw).hexdigest(), len(weights)


def scan_shard(path, expected_keys, block_bytes):
    initial = path.stat()
    with path.open('rb') as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f'{path.name}: truncated 8-byte header length')
        header_length = struct.unpack('<Q', prefix)[0]
        if header_length < 2 or header_length > 16 * MIB or 8 + header_length > initial.st_size:
            raise ValueError(f'{path.name}: invalid/truncated header length {header_length}')
        raw_header = file.read(header_length)
        if len(raw_header) != header_length or raw_header[:1] != b'{':
            raise ValueError(f'{path.name}: invalid/truncated JSON header')
        header = parse_json(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f'{path.name}: header must be a JSON object')
        metadata = header.pop('__metadata__', {})
        if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                 for k, v in metadata.items()):
            raise ValueError(f'{path.name}: invalid __metadata__')
        actual_keys = set(header)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)[:8]
            extra = sorted(actual_keys - expected_keys)[:8]
            raise ValueError(f'{path.name}: index/header key mismatch; missing={missing}, extra={extra}')
        payload_bytes = initial.st_size - 8 - header_length
        tensors = []
        total_elements = 0
        dtype_counts = collections.Counter()
        for key, spec in header.items():
            if not isinstance(spec, dict) or set(spec) != {'dtype', 'shape', 'data_offsets'}:
                raise ValueError(f'{path.name}:{key}: invalid tensor spec fields')
            dtype, shape, offsets = spec['dtype'], spec['shape'], spec['data_offsets']
            if not isinstance(dtype, str) or dtype not in FORMATS:
                raise ValueError(f'{path.name}:{key}: unsupported weight dtype {dtype!r}')
            if not isinstance(shape, list) or any(not integer(x) or x < 0 for x in shape):
                raise ValueError(f'{path.name}:{key}: invalid shape')
            if not isinstance(offsets, list) or len(offsets) != 2 or any(not integer(x) for x in offsets):
                raise ValueError(f'{path.name}:{key}: invalid data_offsets')
            begin, end = offsets
            width = FORMATS[dtype][0]
            elements = math.prod(shape)
            if begin < 0 or end < begin or end > payload_bytes or end - begin != elements * width:
                raise ValueError(f'{path.name}:{key}: offset/shape/size mismatch or truncated data')
            tensors.append((begin, end, key, dtype, elements))
            total_elements += elements
            dtype_counts[dtype] += 1
        tensors.sort()
        cursor = 0
        for begin, end, key, dtype, elements in tensors:
            if end != begin:
                if begin != cursor:
                    raise ValueError(f'{path.name}:{key}: overlapping tensor data or unindexed data gap')
                cursor = end
        if cursor != payload_bytes:
            raise ValueError(f'{path.name}: unindexed/trailing or truncated tensor bytes')
        # Hash exactly the original prefix, padded header and complete payload in order.
        digest = hashlib.sha256()
        digest.update(prefix)
        digest.update(raw_header)
        del raw_header, header, actual_keys
        bad_tensors = []
        nonfinite_count = 0
        for begin, end, key, dtype, elements in tensors:
            remaining = end - begin
            width, unsigned_dtype, exponent_mask = FORMATS[dtype]
            aligned_block_bytes = block_bytes - block_bytes % width
            bad_count = 0
            while remaining:
                size = min(aligned_block_bytes, remaining)
                block = file.read(size)
                if len(block) != size:
                    raise ValueError(f'{path.name}:{key}: data truncated during scan')
                digest.update(block)
                values = np.frombuffer(block, dtype=unsigned_dtype)
                # All exponent bits set means either +/-Inf or NaN for these IEEE formats.
                bad_count += int(np.count_nonzero((values & exponent_mask) == exponent_mask))
                remaining -= size
                del values, block
            if bad_count:
                bad_tensors.append({'key': key, 'dtype': dtype, 'elements': elements,
                                    'nonfinite_count': bad_count})
                nonfinite_count += bad_count
        if file.read(1):
            raise ValueError(f'{path.name}: unexpected trailing bytes')
    final = path.stat()
    if (initial.st_size, initial.st_mtime_ns, initial.st_ino) != (final.st_size, final.st_mtime_ns, final.st_ino):
        raise ValueError(f'{path.name}: file changed during scan')
    return {'file': path.name, 'file_bytes': initial.st_size, 'sha256': digest.hexdigest(),
            'tensor_count': len(tensors), 'elements': total_elements, 'tensor_bytes': payload_bytes,
            'dtype_tensor_counts': dict(dtype_counts), 'nonfinite_count': nonfinite_count,
            'nonfinite_tensors': bad_tensors}


def scan(root, block_bytes):
    shards, declared_bytes, index_sha, key_count = read_index(root)
    result = {'index_sha256': index_sha, 'index_keys': key_count, 'shard_count': len(shards),
              'declared_tensor_bytes': declared_bytes, 'shards': [], 'total_keys': 0,
              'total_elements': 0, 'total_tensor_bytes': 0, 'nonfinite_count': 0}
    for filename in sorted(shards):
        row = scan_shard(root / filename, shards[filename], block_bytes)
        result['shards'].append(row)
        result['total_keys'] += row['tensor_count']
        result['total_elements'] += row['elements']
        result['total_tensor_bytes'] += row['tensor_bytes']
        result['nonfinite_count'] += row['nonfinite_count']
        print(json.dumps({'shard_complete': filename, 'nonfinite_count': row['nonfinite_count']}), file=sys.stderr, flush=True)
    if result['total_keys'] != key_count:
        raise ValueError('Total tensor count differs from index key count')
    if declared_bytes is not None and declared_bytes != result['total_tensor_bytes']:
        raise ValueError('Index metadata.total_size differs from actual tensor bytes')
    result['complete_scan'] = True
    result['passed'] = result['nonfinite_count'] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hf_dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--block-mib', type=int, default=8, help='Read block MiB, 1..32; default 8')
    args = parser.parse_args()
    if not 1 <= args.block_mib <= 32:
        parser.error('--block-mib must be between 1 and 32')
    start = time.monotonic()
    report = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'hf_dir': str(args.hf_dir.resolve()), 'block_bytes': args.block_mib * MIB,
              'method': 'Strict index/header validation, sequential full shard SHA256 and all-element exponent-mask finite checks.',
              'supported_dtypes': list(FORMATS), 'cpu_only': True, 'complete_scan': False,
              'passed': False}
    try:
        report.update(scan(args.hf_dir.resolve(), args.block_mib * MIB))
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    report['elapsed_seconds'] = round(time.monotonic() - start, 3)
    report['peak_rss_bytes'] = peak_rss_bytes()
    report['peak_rss_below_200_mib'] = report['peak_rss_bytes'] < 200 * MIB
    report['forbidden_modules_imported'] = [name for name in ['torch', 'torch_npu', 'ray'] if name in sys.modules]
    if not report['peak_rss_below_200_mib'] or report['forbidden_modules_imported']:
        report['passed'] = False
        report['resource_or_import_contract_violation'] = True
    temporary = args.output.with_name(args.output.name + '.partial')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({key: report.get(key) for key in ['passed', 'complete_scan', 'total_keys', 'total_elements',
                                                      'total_tensor_bytes', 'nonfinite_count', 'peak_rss_bytes', 'error']},
                     ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
