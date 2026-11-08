"""Record CPU-only logical-KV/interface evidence; no SG/NPU imports or startup."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose a fresh evidence directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if 'torch_npu' in sys.modules:
        raise RuntimeError('CPU validation must not import torch_npu')
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parent
    sources=list(root.glob('*.py'))+list((root/'pinned-source').glob('*.py'))
    sources += list((root.parent/'c4v8_device_core').glob('device_*.py'))
    sources += [root.parent/'c4v8_device_core'/'exact_fp32.py']
    sources += [root.parent/'nvfp4_mxfp8_attention_sim'/name for name in ('attention.py','nvfp4.py','mxfp8.py')]
    hashes={str(path.relative_to(root.parent)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    stream=io.StringIO()
    start=time.monotonic()
    tests=unittest.defaultTestLoader.discover(str(root),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    log=stream.getvalue()
    (args.output_dir/'unittest.log').write_text(log)
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),device='cpu',torch=torch.__version__,
                python=sys.version,platform=platform.platform(),threads=torch.get_num_threads(),
                torch_npu_imported='torch_npu' in sys.modules,tests_run=result.testsRun,
                passed=result.wasSuccessful(),elapsed_seconds=time.monotonic()-start,
                source_sha256=hashes,
                validation_scope=['logical physical-token KV mapping, prefix/chunk/decode, BF16 cache preservation',
                                  'CPU GQA and head-sharding analogues',
                                  'AST-extracted pinned base dispatch and forward-mode enum with mocked native pool',
                                  'scoped selector wrapper with ordinary production defaults preserved'],
                not_validated=['real SGLang module initialization or registration in child workers',
                               'NPU native cache read/write execution','distributed TP4/TP8 generation or logprobs',
                               '30B model, RL, throughput, native bit parity'])
    (args.output_dir/'cpu-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(log)
    print(json.dumps({key:report[key] for key in ('device','tests_run','passed','elapsed_seconds')}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
