"""Run a small C8V16 attention forward and backward without model weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "workspace/experiments/c8v16_attention"
REFERENCE = ROOT.parent / "c4v8/workspace/experiments/c4v8_device_core"
FUSED = REFERENCE.parent / "c4v8_fused_qdq"
TRITON = REFERENCE.parent / "c4v8_fused_nvfp4"
sys.path[:0] = list(map(str, (CORE, REFERENCE, FUSED, TRITON)))


def run(device):
    from c8v16_attention import AttentionConfig, attention
    from device_quant import mxfp8_values

    if device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
    config = AttentionConfig()
    torch.manual_seed(211)
    count = 133
    # At head_dim=128 the operands use four independent MX scale groups.
    shapes = [(1, 4, count, 128), (1, 1, count, 128), (1, 1, count, 128)]
    inputs = [torch.randn(shape).to(torch.bfloat16) for shape in shapes]
    # A future outlier in the *same* V scale block tests for causal leakage.
    inputs[2][:, :, 130, ::3] *= 512
    cotangent = torch.randn(shapes[0]).to(torch.bfloat16)
    def execute(on):
        q, k, v = [t.to(on).detach().requires_grad_() for t in inputs]
        y = attention(q, k, v, config)
        assert y.dtype == torch.bfloat16 and y.device == q.device
        grads = torch.autograd.grad(y, (q, k, v), cotangent.to(on))
        for t in grads:
            assert torch.isfinite(t).all() and t.float().norm() > 0
        return [v.detach().float().cpu() for v in (y, *grads)]

    start = time.monotonic()
    values = execute(device)
    reference = execute("cpu") if device != "cpu" else values
    errors = {}
    for name, got, expected in zip(("output", "dQ", "dK", "dV"), values, reference):
        relative_l2 = ((got - expected).norm() / expected.norm()).item()
        errors[name] = dict(relative_l2=relative_l2,
                            max_abs=(got - expected).abs().max().item())
        if not relative_l2 < 0.025:
            raise AssertionError(f"{name} relative L2 {relative_l2} exceeds 0.025")

    with torch.no_grad():
        q, k, v = [x.to(device) for x in inputs]
        # For one visible token softmax yields unit probability. Each V channel
        # owns a distinct length-one block with tail padding on the key axis.
        y_one = attention(q[:, :, :1], k[:, :, :1], v[:, :, :1], config)
        dq_v = mxfp8_values(inputs[2][:, :, :1].transpose(-1, -2)).values.transpose(-1, -2)
        expected_one = dq_v.expand_as(y_one.cpu()).to(torch.bfloat16)
        torch.testing.assert_close(y_one.cpu(), expected_one, rtol=0, atol=0)
        prefix_errors = {}
        for index in (30, 32, 128):
            decoded = attention(q[:, :, index:index + 1], k[:, :, :index + 1],
                                v[:, :, :index + 1], config, query_offset=index)
            full = values[0][:, :, index:index + 1]
            diff = (decoded.float().cpu() - full).abs()
            prefix_errors[str(index + 1)] = diff.max().item()
            torch.testing.assert_close(decoded.float().cpu(), full, rtol=0, atol=0)
    return dict(device=device, policy=vars(config), shape_q=shapes[0], shape_kv=shapes[1],
                output_dtype="bfloat16", one_token_pv_oracle_exact=True,
                errors_vs_cpu=errors, prefix_vs_full_max_abs=prefix_errors,
                gradients_finite_and_nonzero=True, wall_seconds=time.monotonic()-start,
                optimized_model_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "npu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "npu" and len(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")) != 1:
        parser.error("Expose a single device for this check")
    result = run(args.device)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
