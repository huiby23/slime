"""Run CPU-only numerical evidence; no accelerator availability/probe calls."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if 'torch_npu' in sys.modules:
    raise RuntimeError('CPU-only validation must not autoload torch_npu')

from attention import AttentionConfig,attention
from test_attention import dense


def errors(actual,reference):
    actual=actual.detach().double();reference=reference.detach().double()
    delta=actual-reference
    return dict(relative_l2=(delta.norm()/reference.norm().clamp_min(1e-30)).item(),
                rmse=delta.square().mean().sqrt().item(),max_abs=delta.abs().max().item())


def compare():
    arms={
        'c4v8_forward_tree_baseline':AttentionConfig(),
        'c4v8_mixed_fp32_reduction_ablation':AttentionConfig(reduction='mixed_fp32_reduction'),
        'c4v16_mixed_fp32_reduction_control':AttentionConfig(vector='bf16',reduction='mixed_fp32_reduction'),
        'c4v32_control':AttentionConfig(vector='fp32',reduction='mixed_fp32_reduction'),
    }
    records=[]
    for seed,n,d,dtype in ((1708,33,32,torch.float32),(1808,65,35,torch.float32),(1908,33,32,torch.bfloat16)):
        generator=torch.Generator().manual_seed(seed)
        original=[torch.randn((1,h,n,d),generator=generator).to(dtype) for h in (4,2,2)]
        coefficient=torch.randn((1,4,n,d),generator=generator)/n
        outputs={};gradients={}
        for label,config in [('unquantized_fp32_reference',None),*arms.items()]:
            values=[x.detach().clone().float().requires_grad_() if config is None else x.detach().clone().requires_grad_() for x in original]
            output=dense(*values) if config is None else attention(*values,config)
            loss=(output.float()*coefficient).sum()
            loss.backward()
            if not bool(torch.isfinite(output).all()) or any(not bool(torch.isfinite(x.grad).all()) for x in values):
                raise FloatingPointError('Nonfinite comparison output/gradient')
            outputs[label]=output.detach().float()
            gradients[label]=torch.cat([x.grad.detach().float().flatten() for x in values])
        record=dict(seed=seed,shape_q=list(original[0].shape),shape_kv=list(original[1].shape),input_dtype=str(dtype),
            loss='sum(output * fixed private-generator random coefficient / sequence_length)',arms={})
        for label in arms:
            record['arms'][label]=dict(
                output_vs_unquantized=errors(outputs[label],outputs['unquantized_fp32_reference']),
                gradient_vs_unquantized=errors(gradients[label],gradients['unquantized_fp32_reference']),
                output_vs_c4v32=errors(outputs[label],outputs['c4v32_control']),
                gradient_vs_c4v32=errors(gradients[label],gradients['c4v32_control']),
                gradient_l2=gradients[label].double().norm().item(),finite=True)
        records.append(record)
    return dict(configurations={key:value.metadata() for key,value in arms.items()},cases=records,
        interpretation='Small random CPU tensor experiments only. Not 30B, RL, accuracy, throughput, native-format parity, or quantized backward evidence.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();source=Path(__file__).resolve().parent
    modules=['test_nvfp4','test_mxfp8','test_attention']
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
    with (args.output_dir/'unittest.log').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,torch=torch.__version__,
        platform=platform.platform(),device='cpu',threads=torch.get_num_threads(),
        backend_autoload=os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'],torch_npu_imported='torch_npu' in sys.modules,
        test_count=result.testsRun,tests_passed=result.wasSuccessful(),
        failures=[dict(test=str(test),traceback=trace) for test,trace in result.failures+result.errors],
        skipped=[str(item) for item in result.skipped],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))},
        limitations=['CPU numerical reference only','Forward QDQ with STE backward surrogate',
                    'Ordinary autograd retains tiles; no memory-efficient attention backward',
                    'No NPU/CUDA execution, real-model RL, native-bit-exact or speed claim'])
    if result.wasSuccessful():
        try:
            comparison=compare()
            (args.output_dir/'numerical-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            report['comparison_passed']=True
        except Exception:
            import traceback
            report['comparison_passed']=False;report['comparison_error']=traceback.format_exc()
    report['elapsed_seconds']=time.monotonic()-start
    report['passed']=report['tests_passed'] and report.get('comparison_passed',False)
    (args.output_dir/'cpu-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),passed=report['passed'],test_count=result.testsRun,
        elapsed_seconds=report['elapsed_seconds'],failed_tests=[str(x) for x,_ in result.failures+result.errors])))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
