"""Test the HiF4 converter against pinned public values and the attention routes.

Optional --upstream takes HiF4_NVFP4_v14f16.py from GCC-HiFloat at
6d937b6fcf34f63b8fc563bd72e3aea0f44a46b4. The expected digest covers all
decoded FP32 values with either sign of zero canonicalized to +0. Input signs
of zero are tested separately. --device npu uses one explicitly visible card.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.extend(str(p) for p in (
    HERE.parent / "c8v16/workspace/experiments/c8v16_attention",
    HERE.parent / "c4v8/workspace/experiments/c4v8_device_core",
    HERE.parent / "c4v8/workspace/experiments/c4v8_rl_training_bridge",
))
from hif4_quant import quantize
from hif4_attention import AttentionConfig, P_METHOD, attention
from bridge import recomputed_attention

UPSTREAM_SHA = "37b376bae331ddd90c72749653185eac00378b69cfc0e09c7c82dcbd80782928"
EXPECTED_SHA = "ca250f7e082c670d9ec78a572ea7b56d776c5a4ba7306211352ac60c7f8b7674"


def vectors():
    rng = np.random.Generator(np.random.PCG64(311))
    groups = []
    for exponent in range(-53, 21):
        # Dyadic integers vary the local 8- and 4-value maxima independently.
        values = rng.integers(-383, 384, size=(8, 64)).astype(np.float32)
        values = np.ldexp(values, exponent - 8)
        groups.append(values)
    ties = np.zeros((3, 64), np.float32)
    ties[:, -1] = 7
    ties[0, :8] = [.125, -.125, .375, -.375, .625, -.625, 1.625, -1.625]
    ties[1, :8] = [4, .25, -.25, 0, 2, .75, -.75, .5]
    ties[2, :8] = [2, .375, .625, .875, .125, .375, .625, .875]
    return torch.from_numpy(np.concatenate([*groups, ties])).bfloat16().float()


def digest(values):
    array = np.asarray(values, dtype="<f4").copy()
    array[array == 0] = 0.
    return hashlib.sha256(array.tobytes()).hexdigest()


def check(device, upstream):
    x = vectors()
    expected = EXPECTED_SHA
    if upstream:
        if hashlib.sha256(upstream.read_bytes()).hexdigest() != UPSTREAM_SHA:
            raise ValueError("Unexpected GCC-HiFloat reference file")
        to_hif4 = runpy.run_path(str(upstream))["To_HiFX"]
        with np.errstate(all="ignore"):
            expected = digest(to_hif4(x.numpy().T).T)
        assert EXPECTED_SHA == expected
    out = quantize(x.to(device)).cpu().numpy()
    assert digest(out) == expected
    assert not bool(quantize(torch.zeros(3, 64, device=device)).any())
    neg_zero = torch.tensor([-0., 0.], device=device)
    assert torch.equal(torch.signbit(quantize(neg_zero).cpu()), torch.tensor([True, False]))
    toy = quantize(torch.tensor([[1., .3]], device=device)).tolist()[0]
    assert toy == [.9375, .3125]
    for tail in (1, 17, 63, 65, 129):
        view = x.reshape(-1)[:tail * 4].reshape(4, tail)
        padded = torch.nn.functional.pad(view, (0, -tail % 64))
        torch.testing.assert_close(quantize(view.to(device)).cpu(), quantize(padded)[:, :tail], rtol=0, atol=0)
    try:
        quantize(torch.tensor([1., float("nan")], device=device))
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Nonfinite ordinary training operands must be rejected")
    leaf = torch.tensor([1.01, .31, -.125], device=device, requires_grad=True)
    weights = torch.tensor([.5, 2., -3.], device=device)
    grad, = torch.autograd.grad((quantize(leaf) * weights).sum(), leaf)
    torch.testing.assert_close(grad, weights, atol=0, rtol=0)
    # Sweep maximum values through E6M2 midpoints. Adjacent BF16 peaks
    # can round to the same SF midpoint before the E6M2 tie is resolved.
    mantissas = 1. + torch.arange(128, dtype=torch.float32) / 128
    exponents = torch.tensor([-50, -40, -11, -1, 0, 1, 7, 15, 19])
    peaks = torch.ldexp(mantissas[None, :], exponents[:, None]).reshape(-1, 1)
    fractions = torch.linspace(-1., .875, 64)
    sweep = (peaks * fractions).bfloat16().float()
    torch.testing.assert_close(quantize(sweep.to(device)).cpu(), quantize(sweep), rtol=0, atol=0)
    g = torch.Generator().manual_seed(717)
    inputs = [torch.randn(1, heads, 133, 128, generator=g).bfloat16() for heads in (4, 1, 1)]
    inputs[-1][:, :, 63, ::3] *= 256
    inputs[-1][:, :, 130, ::3] *= 64
    for operand in (*inputs[:2], inputs[-1].transpose(-1, -2)):
        torch.testing.assert_close(quantize(operand.to(device)).cpu(), quantize(operand), rtol=0, atol=0)
    cotangent = torch.randn(inputs[0].shape, generator=g).bfloat16()
    cfg = AttentionConfig(p_method=P_METHOD)
    def forward_backward(on):
        q, k, v = [t.to(on).requires_grad_() for t in inputs]
        output = recomputed_attention(q, k, v, kernel=attention, config=cfg)
        grads = torch.autograd.grad(output, (q, k, v), cotangent.to(on))
        assert output.dtype == torch.bfloat16
        for t in (output, *grads):
            assert bool(torch.isfinite(t).all()) and float(t.detach().float().norm()) > 0
        return [t.detach().float().cpu() for t in (output, *grads)]
    compared = forward_backward(device)
    relative_l2 = None
    if device != "cpu":
        reference = forward_backward("cpu")
        relative_l2 = [float((a-b).norm()/b.norm()) for a,b in zip(compared,reference)]
        assert max(relative_l2) < .025, relative_l2
    with torch.no_grad():
        q, k, v = [t.to(device) for t in inputs]
        for index in (30, 62, 63, 64, 127, 128):
            incremental = attention(q[:, :, index:index+1], k[:, :, :index+1], v[:, :, :index+1],
                                    cfg, query_offset=index)
            torch.testing.assert_close(incremental.float().cpu(), compared[0][:,:,index:index+1],
                                       rtol=0, atol=0)
    return dict(device=device, decoded_sha256=expected, count=x.numel(),
                groups64=x.numel()//64, exact_against_public_reference=True,
                e6m2_scale_sweep_groups=peaks.numel(), qkv_qdq_exact_against_cpu=True,
                reference_sha256=UPSTREAM_SHA, signed_and_all_zero_format_checks=True,
                max_abs_prefix_difference=0., forward_dq_dk_dv_relative_l2=relative_l2,
                probability_example=toy, p_candidate=P_METHOD)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    p.add_argument("--upstream", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.device == "npu":
        if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdigit():
            p.error("Expose one idle physical NPU for the device test")
        import torch_npu
        torch.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
    result = check(args.device, args.upstream)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n")
