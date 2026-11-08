"""Versioned CPU-oracle audit; no NPU kernel or production module activation."""
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT.parent
files=[ROOT/'fast_attention.py',ROOT/'tests/test_fast_custom_cpu.py',Path(__file__),
    EXP/'c4v8_device_core/device_attention.py',EXP/'c4v8_device_core/device_quant.py',
    EXP/'c4v8_device_core/exact_fp32.py',EXP/'c4v8_rl_training_bridge/bridge.py',
    EXP/'c4v8_rl_training_bridge/layouts.py']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
before={str(p):sha(p) for p in files}
import test_fast_custom_cpu as tests
import torch
from unittest.mock import patch


def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: run_cpu_review.py NEW_OUTPUT_DIRECTORY')
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    (output/'tests.log').write_text(stream.getvalue())
    api=tests.OracleAPI();g=torch.Generator().manual_seed(771)
    numerator=torch.randn(2,3,5,17,generator=g);denominator=torch.rand(2,3,5,1,generator=g)+0.125
    weight=torch.randn(numerator.shape,generator=g)
    a=tests.clone_inputs((numerator,denominator),(True,True));b=tests.clone_inputs((numerator,denominator),(True,True))
    with patch.object(tests.fast,'_normalize_api',lambda:api.normalize):
        actual,_=tests.fast._Normalize.apply(*a)
        expected=tests.vr(b[0]/b[1])
        actual.backward(weight);expected.backward(weight)
    metrics={}
    for label,x,y in [('output',actual,expected),('numerator_gradient',a[0].grad,b[0].grad),('denominator_gradient',a[1].grad,b[1].grad)]:
        delta=(x-y).float()
        metrics[label]=dict(bit_equal=torch.equal(x,y),different_values=int((x!=y).sum()),
            max_abs_error=float(delta.abs().max()),relative_l2=float(delta.norm()/y.float().norm().clamp_min(1e-30)))
    after={str(p):sha(p) for p in files}
    report=dict(passed=result.wasSuccessful() and before==after,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),source_unchanged=before==after,
        source_sha256=before,checked_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        device='CPU only',npu_validation=False,production_activation=False,
        actual_classes_exercised=['_MX','_Pre','_Post','_Normalize'],
        API='Injected CPU oracle implementing quantizer values/online-softmax; actual custom Functions forward/backward',
        full_attention_entry='Original AST; only CPU early-return/device allowlist/HF32 hardware guard adapted; unmodified imported Functions',
        bypassed_hardware_guards=['CPU fallback return','NPU-only device allowlist','NPU HF32 accessor'],
        forward_comparison='Exact values versus existing CPU reference',
        gradient_comparison='STE numerical closeness, not bitwise backward equality',
        normalize_metrics=metrics,
        memory_scope='Forward outer-recompute saves Q/K/V only; invalid flags one scalar per tile; no real peak NPU memory measurement',
        remaining=['actual fused kernel NPU values and gradients','actual NPU replay memory/throughput','whole-decode/prefill consistency'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','tests_run','source_unchanged','normalize_metrics']},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
