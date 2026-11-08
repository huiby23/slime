#!/usr/bin/env python3
"""Bounded single-attention timing; no model, optimizer, server, or job estimate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback


BASE = Path(__file__).resolve().parent.parent
SOURCE_NAMES = (
    'c4v8_performance_probe/benchmark.py',
    'c4v8_performance_probe/run-owned-command.sh',
    'c4v8_fast_attention/fast_attention.py',
    'c4v8_device_core/device_attention.py', 'c4v8_device_core/device_quant.py',
    'c4v8_device_core/exact_fp32.py', 'c4v8_fused_nvfp4/nvfp4_triton.py',
    'c4v8_fused_qdq/forward_api.py', 'c4v8_fused_qdq/fused_mxfp8.py',
    'c4v8_fused_qdq/whole_decode.py', 'c4v8_rl_training_bridge/bridge.py',
    'c4v8_staged_decode/staged_decode.py', 'c4v8_staged_decode/staged_kernels.py',
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-lengths', nargs='*', type=int, default=[1024, 4096],
                   help='Production query-tile recomputed_attention forward/backward')
    p.add_argument('--direct-lengths', nargs='*', type=int, default=[1024],
                   help='Diagnostic full-graph autograd; default excludes4096')
    p.add_argument('--checkpoint-lengths', nargs='*', type=int, default=[],
                   help='Optional reentrant checkpoint around real bridge; FA part only')
    p.add_argument('--decode-prefixes', nargs='*', type=int, default=[128, 1024, 4096, 8192],
                   help='Already cached tokens; total KV length = prefix +1')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--query-block', type=int, choices=[64,256,512,1024], default=64,
                   help='Explicit query tiling policy; key tile stays128; a timing result is not numerical acceptance')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-seconds', type=float, default=540)
    p.add_argument('--memory-fraction', type=float, default=.05)
    p.add_argument('--case-order', choices=['decode-first', 'train-first'], default='decode-first')
    p.add_argument('--seed', type=int, default=731019)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--cpu-smoke', action='store_true', help='Small functional plumbing test; never NPU timings')
    return p


def cases_for(a):
    if not (0 <= a.warmup <= 5 and 1 <= a.repeats <= 10 and 1 <= a.max_seconds <= 3300):
        raise ValueError('warmup0..5, repeats1..10, and max-seconds1..3300 required')
    if not 0 < a.memory_fraction <= .05:
        raise ValueError('This probe permits at most5% of physical NPU1 HBM')
    train = []
    for mode, values in [('recomputed', a.train_lengths), ('direct', a.direct_lengths),
                         ('checkpoint-recomputed', a.checkpoint_lengths)]:
        if len(values) != len(set(values)) or any(n < 1 or n > 8192 for n in values):
            raise ValueError('Unique training lengths1..8192 required')
        train.extend(dict(kind='training', mode=mode, query_length=n, kv_length=n,
                          query_offset=0, id=f'{mode}-n{n}') for n in values)
    if len(a.decode_prefixes) != len(set(a.decode_prefixes)) or any(n < 0 or n > 8192 for n in a.decode_prefixes):
        raise ValueError('Unique decode prefixes0..8192 required')
    decode = [dict(kind='decode', mode='inference', query_length=1, kv_length=n+1,
                   prefix_length=n, query_offset=n, id=f'decode-prefix{n}') for n in a.decode_prefixes]
    cases = decode+train if a.case_order == 'decode-first' else train+decode
    if not cases:
        raise ValueError('At least one case is required')
    if a.cpu_smoke and any(c['query_length'] > 33 or c['kv_length'] > 129 for c in cases):
        raise ValueError('CPU smoke is limited to query<=33 and KV<=129; select small cases explicitly')
    return cases


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_inventory():
    return {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def require_owned_supervisor():
    parent = os.getppid()
    argv = [x.decode() for x in (Path('/proc')/str(parent)/'cmdline').read_bytes().split(b'\0') if x]
    expected = '/root/hby/experiments/c4v8-integration-probes/run-owned-probe.py'
    if expected not in argv or '--owned-child' not in argv or os.getpgid(parent) != parent:
        raise RuntimeError('Real NPU probe requires the existing independent run-owned-probe supervisor')


def cache_snapshot(path):
    # Metadata files identify compilation artifacts; never purge a shared cache.
    if not path.is_dir():
        return {}
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in path.rglob('*.json') if p.is_file()}


def cache_delta(before, after):
    return {'new_metadata_files': sorted(set(after)-set(before)),
            'changed_metadata_files': sorted(k for k in set(after)&set(before) if after[k] != before[k])}


def summary(values):
    return {'samples': len(values), 'min_seconds': min(values), 'median_seconds': statistics.median(values),
            'max_seconds': max(values)} if values else {'samples': 0}


class BudgetExceeded(RuntimeError):
    pass


class Probe:
    def __init__(self, args, report, torch):
        self.args, self.report, self.torch = args, report, torch
        self.start = time.monotonic()
        self.device = 'cpu' if args.cpu_smoke else 'npu:1'
        self.cache = Path(os.environ.get('TRITON_CACHE_DIR', '/root/hby/cache/triton-c4v8'))
        self.phase = None

    def save(self):
        self.report['elapsed_seconds'] = time.monotonic()-self.start
        write_json(self.args.output/'report.partial.json', self.report)

    def mark(self, case, phase, iteration):
        self.phase = {'case': case['id'], 'phase': phase, 'iteration': iteration,
                      'elapsed_seconds': time.monotonic()-self.start,
                      'utc': datetime.now(timezone.utc).isoformat()}
        self.report['active_phase'] = self.phase
        write_json(self.args.output/'progress.json', self.phase)
        self.save()
        print('PHASE '+json.dumps(self.phase), flush=True)

    def check_budget(self):
        if time.monotonic()-self.start >= self.args.max_seconds:
            raise BudgetExceeded('Soft budget reached before starting next phase; external owned wrapper enforces hard timeout')

    def sync(self):
        if not self.args.cpu_smoke:
            self.torch.npu.synchronize(1)

    def memory(self):
        if self.args.cpu_smoke:
            return {'available': False, 'reason': 'CPU smoke has no NPU allocation evidence'}
        npu = self.torch.npu
        return {'allocated_bytes': int(npu.memory_allocated(1)), 'reserved_bytes': int(npu.memory_reserved(1)),
                'peak_allocated_bytes': int(npu.max_memory_allocated(1)),
                'peak_reserved_bytes': int(npu.max_memory_reserved(1))}

    def timed(self, case, phase, iteration, function, observe_cache):
        self.check_budget()
        self.mark(case, phase, iteration)
        before_cache = cache_snapshot(self.cache) if observe_cache else None
        self.sync()
        baseline = self.memory()
        if not self.args.cpu_smoke:
            self.torch.npu.reset_peak_memory_stats(1)
        started = time.perf_counter()
        result = function()
        self.sync()
        seconds = time.perf_counter()-started
        evidence = {'seconds': seconds, 'memory_before': baseline, 'memory_after': self.memory()}
        if observe_cache:
            evidence['cache_artifacts'] = cache_delta(before_cache, cache_snapshot(self.cache))
        return result, evidence

    def finite(self, value):
        values = value if isinstance(value, (tuple, list)) else [value]
        checks = [bool(self.torch.isfinite(v).all()) for v in values]
        if not all(checks):
            raise FloatingPointError('Nonfinite output or STE gradient; timing cannot be accepted')
        return checks

    def run_case(self, case, kernel, config, recomputed):
        torch = self.torch
        self.check_budget()
        self.mark(case, 'allocate_inputs', -1)
        g = torch.Generator(device='cpu').manual_seed(self.args.seed+case['kv_length'])
        shapes = [(1,8,case['query_length'],128), (1,1,case['kv_length'],128), (1,1,case['kv_length'],128)]
        # CPU generation and H2D copy are explicit setup, outside measured FA calls.
        base = [torch.randn(shape, generator=g).bfloat16().to(self.device) for shape in shapes]
        dy = torch.randn(shapes[0], generator=g).bfloat16().to(self.device) if case['kind'] == 'training' else None
        row = dict(case, shapes=[list(s) for s in shapes], dtype='bfloat16', iterations=[], completed=False)
        self.report['cases'].append(row)
        self.save()
        for index in range(1+self.args.warmup+self.args.repeats):
            label = 'first_observed' if index == 0 else 'warmup' if index <= self.args.warmup else 'measured'
            entry = {'index': index, 'label': label}
            row['iterations'].append(entry)
            observe_cache = index == 0 or label == 'measured'
            if case['kind'] == 'decode':
                def forward():
                    with torch.inference_mode():
                        return kernel(*base, config, query_offset=case['query_offset'])
                output, entry['forward'] = self.timed(case, 'inference_forward', index, forward, observe_cache)
                entry['output_finite'] = self.finite(output)
                del output
            else:
                leaves = [x.detach().requires_grad_() for x in base]
                if case['mode'] == 'direct':
                    forward = lambda: kernel(*leaves, config)
                elif case['mode'] == 'recomputed':
                    forward = lambda: recomputed(*leaves, kernel=kernel, config=config)
                else:
                    from torch.utils.checkpoint import checkpoint
                    forward = lambda: checkpoint(lambda q,k,v: recomputed(q,k,v,kernel=kernel,config=config),
                                                 *leaves, use_reentrant=True, preserve_rng_state=False)
                output, entry['forward'] = self.timed(case, 'training_forward', index, forward, observe_cache)
                entry['output_finite'] = self.finite(output)
                self.save()  # Preserve first-forward evidence if backward exceeds the lease.
                if case['mode'] == 'checkpoint-recomputed':
                    def backward():
                        output.backward(dy)
                        return tuple(x.grad for x in leaves)
                else:
                    backward = lambda: torch.autograd.grad(output, leaves, dy)
                gradients, entry['backward'] = self.timed(case, 'training_backward_including_replay', index, backward, observe_cache)
                entry['gradient_finite'] = self.finite(gradients)
                entry['gradient_nonzero'] = [bool((x != 0).any()) for x in gradients]
                if not all(entry['gradient_nonzero']):
                    raise AssertionError('Expected nonzero Q/K/V gradients for representative random case')
                entry['forward_plus_backward_seconds'] = entry['forward']['seconds']+entry['backward']['seconds']
                del gradients, output, leaves, backward, forward
            self.save()
        measured = [x for x in row['iterations'] if x['label'] == 'measured']
        row['warm_forward'] = summary([x['forward']['seconds'] for x in measured])
        if case['kind'] == 'training':
            row['warm_backward_including_replay'] = summary([x['backward']['seconds'] for x in measured])
            row['warm_forward_plus_backward'] = summary([x['forward_plus_backward_seconds'] for x in measured])
        artifacts = [x[phase].get('cache_artifacts',{}) for x in measured for phase in ('forward','backward') if phase in x]
        row['warm_samples_had_observed_cache_activity'] = any(a.get('new_metadata_files') or a.get('changed_metadata_files') for a in artifacts)
        row['completed'] = True
        del base, dy
        gc.collect()
        self.sync()
        row['after_cleanup_memory'] = self.memory()
        self.mark(case, 'case_complete', index)


def main(argv=None):
    args = parser().parse_args(argv)
    cases = cases_for(args)
    args.output = args.output.resolve()
    if not args.cpu_smoke and not args.plan_only and not args.output.is_relative_to(Path('/root/hby/experiments')):
        raise ValueError('Real NPU evidence must stay under /root/hby/experiments')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_inventory()
    report = {'status': 'prepared', 'measurement_device': 'cpu-functional-only' if args.cpu_smoke else 'physical_npu1',
              'plan': cases, 'cases': [], 'source_sha256': hashes, 'all_requested_cases_completed': False,
              'actual_grpo_optimizer_updates': 0, 'native_low_precision_acceleration_claimed': False,
              'policy': {'query_block':args.query_block,'key_block':128,'vector':'mxfp8','quantize_operands':True,'causal':True,'reduction':'mxfp8_tree_reduction'},
              'timing_scope': 'Synchronized single-attention host wall time, including kernel CPU dispatch and internal finite checks; external result checks/input allocation/H2D excluded',
              'compile_caveat': 'First observed call includes execution and possible cache miss/compilation. Shared canonical cache is not cleared; first-minus-warm is not pure compilation time. Metadata deltas are observations, not compiler CPU timings.',
              'training_caveat': 'recomputed uses actual production query-tile bridge. direct is a diagnostic, not production memory footprint. Optional checkpoint-recomputed adds reentrant FA checkpoint only; no full Transformer/Megatron/TP/PP/Adam is benchmarked.',
              'feasibility_caveat': 'These layer measurements cannot establish end-to-end tokens/sec, optimizer-step time, or a100-step completion estimate.',
              'warmup': args.warmup, 'repeats': args.repeats, 'soft_budget_seconds': args.max_seconds,
              'memory_fraction_limit': args.memory_fraction,
              'memory_limit_caveat': 'An OOM at the5% single-probe limit is not proof of production training OOM; report the active phase and limit without conflating them.'}
    write_json(args.output/'plan.json', report)
    if args.plan_only:
        print(json.dumps({'status':'prepared_only','output':str(args.output),'cases':len(cases)})); return 0
    for name in hashes:
        target = args.output/'sources'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((BASE/name).read_bytes())
    probe = None
    code = 1
    try:
        for folder in ['c4v8_device_core','c4v8_fused_nvfp4','c4v8_fused_qdq','c4v8_fast_attention','c4v8_rl_training_bridge','c4v8_staged_decode']:
            sys.path.insert(0,str(BASE/folder))
        os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
        if not args.cpu_smoke:
            require_owned_supervisor()
        import torch
        # Register NPU before the first autograd execution, including CPU engines.
        if not args.cpu_smoke:
            import torch_npu  # noqa: F401
            torch.npu.set_device(1)
            torch.npu.set_per_process_memory_fraction(args.memory_fraction,1)
            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.matmul.allow_hf32=False
            if bool(torch.npu.matmul.allow_hf32):raise RuntimeError('HF32 remained enabled')
            report['actual_hf32']=False
            report['jit_compile']=False
        torch.set_num_threads(1)
        os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
        from fast_attention import attention, AttentionConfig
        from bridge import recomputed_attention
        probe=Probe(args,report,torch)
        def interrupted(*_):raise BudgetExceeded('Owned wrapper sent TERM; saving partial progress')
        signal.signal(signal.SIGTERM,interrupted)
        report['status']='running'
        report['cache_directory']=str(probe.cache)
        config=AttentionConfig(**report['policy'])
        for case in cases:probe.run_case(case,attention,config,recomputed_attention)
        report['all_requested_cases_completed']=True
        report['status']='cpu_smoke_completed' if args.cpu_smoke else 'measurement_completed'
        code=0
    except BudgetExceeded as error:
        report.update(status='budget_exhausted',error=str(error));code=124
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        (args.output/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        if probe is not None:report['elapsed_seconds']=time.monotonic()-probe.start
        report['sources_unchanged']=source_inventory()==hashes
        if not report['sources_unchanged']:report['status']='source_changed';code=1
        write_json(args.output/'report.json',report)
        (args.output/'exit').write_text(str(code)+'\n')
        print(json.dumps({'status':report['status'],'completed_cases':sum(c['completed'] for c in report['cases']),
                          'requested_cases':len(cases),'error':report.get('error')}),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
