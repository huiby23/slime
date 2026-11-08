"""Measure one attention layer and compare optimized decode with causal replay."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import check_hif4 as checks
import torch
import torch_npu

from hif4_attention import AttentionConfig, P_METHOD, attention
from hif4_quant import GROUP, quantize
from c8v16_attention import _attention
from bridge import recomputed_attention


def elapsed(func, n):
    samples = []
    for _ in range(n):
        torch.npu.synchronize()
        t = time.perf_counter()
        func()
        torch.npu.synchronize()
        samples.append(time.perf_counter() - t)
    return median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    cfg = AttentionConfig(p_method=P_METHOD)
    gen = torch.Generator().manual_seed(4421)
    def tensor(shape):
        return torch.randn(shape, generator=gen).bfloat16().to("npu")
    results = dict(decode=[], training=[], scope="single attention layer, synthetic BF16 Q/K/V")
    def save():
        opts.output.write_text(json.dumps(results, indent=2) + "\n")
    with torch.no_grad():
        for b, length in [(1,1),(3,64),(3,65),(2,128),(3,129),(3,511),(3,1025),(3,2048),(3,4097),(1,8192)]:
            q, k, v = tensor((b,4,1,128)), tensor((b,1,length,128)), tensor((b,1,length,128))
            old = lambda: _attention(q,k,v,cfg,length-1,qdq=quantize,group_size=GROUP)
            fast = lambda: attention(q,k,v,cfg,query_offset=length-1)
            expected = old()
            actual = fast()
            diff = int((actual != expected).sum())
            item = dict(batch=b, prefix=length, different_output_values=diff,
                        max_abs=float((actual.float()-expected.float()).abs().max()))
            results["decode"].append(item)
            save()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if length >= 511:
                item.update(reference_s=elapsed(old,3), optimized_s=elapsed(fast,5))
                save()
    for length in (1024,2048,4096):
        leaves = [tensor((1,h,length,128)).requires_grad_() for h in (8,1,1)]
        cotangent = tensor(leaves[0].shape)
        def training():
            y = recomputed_attention(*leaves, kernel=attention,config=cfg)
            grads = torch.autograd.grad(y, leaves, cotangent)
            if not all(bool(torch.isfinite(g).all()) for g in grads):
                raise ArithmeticError("Nonfinite gradient")
        warmup_s = elapsed(training,1)
        results["training"].append(dict(length=length, fwd_and_bwd_s=elapsed(training,1),warmup_s=warmup_s))
        save()
    results["passed"] = True
    save()
    print(json.dumps(results))

if __name__ == "__main__":
    main()
