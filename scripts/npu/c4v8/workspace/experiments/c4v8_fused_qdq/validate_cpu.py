"""CPU-only arithmetic/source audit. Does not import or execute Triton kernels."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['OMP_NUM_THREADS']='1'
import argparse,ast,hashlib,json,sys,time,unittest
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Choose a new evidence file')
    root=Path(__file__).resolve().parent
    for f in root.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
    torch.set_num_threads(1);start=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(root),pattern='test_*cpu*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    files=list(root.glob('*.py'))+list((root/'pinned-source').glob('*.py'))
    files+=list((root.parent/'c4v8_device_core').glob('*.py'))
    report={'passed':result.wasSuccessful(),'tests':result.testsRun,
            'elapsed_seconds':time.monotonic()-start,'device':'cpu','torch_version':torch.__version__,
            'torch_npu_imported':'torch_npu' in sys.modules,'triton_imported':'triton' in sys.modules,
            'numerical_validation':'CPU independent MXFP8 encoding and prescribed vector operation graph',
            'npu_kernel_compiled':False,'npu_kernel_executed':False,'production_activated':False,
            'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
