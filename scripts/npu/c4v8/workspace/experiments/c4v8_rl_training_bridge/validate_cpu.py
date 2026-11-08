"""CPU-only evidence runner; never initializes any accelerator."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
from datetime import datetime, timezone
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if 'torch_npu' in sys.modules:
        raise RuntimeError('Unexpected torch_npu import in CPU-only test runner')
    torch.set_num_threads(1)
    here=Path(__file__).resolve().parent
    log=io.StringIO()
    started=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(here),pattern='test_bridge.py')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (args.output_dir/'unittest.log').write_text(log.getvalue())
    source_dirs=[here,here.parent/'nvfp4_mxfp8_attention_sim',here.parent/'c4v8_device_core']
    sources={str(f.relative_to(here.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
             for d in source_dirs for f in d.glob('*.py')}
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),device='cpu',
        torch_version=torch.__version__,python=sys.version,threads=torch.get_num_threads(),
        torch_npu_imported='torch_npu' in sys.modules,tests_run=result.testsRun,
        passed=result.wasSuccessful(),failures=[str(x[0]) for x in result.failures+result.errors],
        elapsed_seconds=time.monotonic()-started,source_sha256=sources,
        scope='Query-tile recompute STE gradients, SBHD/THD GQA adapters and spec substitution; CPU only.',
        limitations=['No quantized backward GEMMs or reductions',
                     'Pinned interface read, not actual Megatron distributed model execution',
                     'No NPU test, model training or rollout integration performed by this runner',
                     'Forward stores Q/K/V only; replay graph is still O(query_tile*key_length)'])
    (args.output_dir/'cpu-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','tests_run','device','elapsed_seconds')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
