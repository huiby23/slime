"""Numeric and gradient checks for the custom W4A8 recipe."""

import pytest
import torch
import torch.nn.functional as F

from slime_plugins.nvfp4_sim.recipe import quantize
from slime_plugins.w4a8_sim.recipe import W4A8Recipe, fp8_quantize, grouped_linear, linear, routed_experts

NUM_GPUS = 0


@pytest.mark.parametrize("format,dtype,maximum", [("e4m3", torch.float8_e4m3fn, 448), ("e5m2", torch.float8_e5m2, 57344)])
def test_fp8_payload_matches_native_conversion(format, dtype, maximum):
    torch.manual_seed(42)
    x = torch.randn(7, 197) * 3
    q = fp8_quantize(x, format=format)
    padded = F.pad(x, (0, (-x.shape[1]) % 64)).reshape(7, -1, 64)
    scaled = (padded / q.scales[..., None]).clamp(-maximum, maximum)
    reference = scaled.to(dtype).float()
    torch.testing.assert_close(q.values.float().reshape_as(reference), reference, atol=0, rtol=0)
    assert torch.isfinite(q.dequantize()).all()
    assert (q.dequantize() - x).norm() / x.norm() < (0.04 if format == "e4m3" else 0.07)


@pytest.mark.parametrize("shape", [(0, 32), (3, 128), (2, 129)])
def test_zero_and_empty_fp8(shape):
    x = torch.zeros(shape)
    torch.testing.assert_close(fp8_quantize(x).dequantize(), x, rtol=0, atol=0)


def test_nonfinite_inputs_are_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        fp8_quantize(torch.tensor([[float("nan")]]))


def test_forward_and_both_gradients_use_declared_quantized_operands():
    torch.manual_seed(43)
    x = torch.randn(13, 64).bfloat16().requires_grad_()
    w = torch.randn(48, 64).bfloat16().requires_grad_()
    bias = torch.randn(48).bfloat16().requires_grad_()
    g = torch.randn(13, 48).bfloat16()
    recipe = W4A8Recipe()
    output = linear(x, w, bias, recipe=recipe)
    qx = fp8_quantize(x).dequantize()
    qw = quantize(w, weight=True).dequantize()
    expected = (qx @ qw.T + bias.float()).bfloat16()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.backward(g)
    qg = fp8_quantize(g, format="e5m2").dequantize()
    qwt = quantize(w.T.contiguous(), weight=True).dequantize()
    torch.testing.assert_close(x.grad, (qg @ qwt.T).bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(w.grad, (qg.T @ qx).bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(bias.grad, g.float().sum(0).bfloat16(), atol=0, rtol=0)
    assert recipe.counters["fprop"] == recipe.counters["dgrad"] == recipe.counters["wgrad"] == 1
    assert not torch.equal(x.grad, (g.float() @ w.float()).bfloat16())


def test_grouped_experts_accumulate_fp32_gradients_with_empty_expert():
    torch.manual_seed(44)
    x = torch.randn(9, 32).bfloat16().requires_grad_()
    weights = [torch.nn.Parameter(torch.randn(48, 32).bfloat16()) for _ in range(3)]
    for w in weights:
        w.main_grad = torch.zeros_like(w, dtype=torch.float32)
        w.grad_added_to_main_grad = False
    recipe = W4A8Recipe()
    grouped_linear(x, weights, [4, 0, 5], recipe=recipe).float().square().mean().backward()
    assert x.grad.isfinite().all() and x.grad.norm() > 0
    assert weights[0].main_grad.norm() > 0 and weights[2].main_grad.norm() > 0
    assert weights[1].main_grad.count_nonzero() == 0
    assert all(w.grad_added_to_main_grad for w in weights)
    assert all(w.main_grad.dtype == torch.float32 for w in weights)
    assert recipe.counters["empty_expert"] == 1


def test_routing_and_weighted_combination_preserve_selected_experts():
    torch.manual_seed(45)
    x = torch.randn(5, 32).bfloat16()
    w13 = torch.randn(4, 64, 32).bfloat16()
    w2 = torch.randn(4, 32, 32).bfloat16()
    ids = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 2], [1, 2]])
    probabilities = torch.tensor([[0.2, 0.8]]).expand(5, 2)
    result = routed_experts(x, w13, w2, ids, probabilities, recipe=W4A8Recipe())
    reference = torch.zeros_like(x, dtype=torch.float32)
    for token in range(5):
        for slot in range(2):
            expert = ids[token, slot]
            a, b = linear(x[token:token+1], w13[expert], recipe=W4A8Recipe()).chunk(2, -1)
            y = linear(F.silu(a) * b, w2[expert], recipe=W4A8Recipe())
            reference[token] += y[0].float() * probabilities[token, slot]
    torch.testing.assert_close(result, reference.bfloat16(), atol=0, rtol=0)


def test_logical_weight_amax_preserves_quantization_across_tp_shards():
    torch.manual_seed(46)
    w = torch.randn(64, 768).bfloat16()
    w[:, :192] *= 20
    whole = quantize(w, weight=True).dequantize()
    pieces = [quantize(part, weight=True, tensor_amax=w.abs().float().amax()).dequantize()
              for part in w.split(192, dim=1)]
    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=0, atol=0)
    x = torch.randn(9, 768).bfloat16()
    torch.testing.assert_close(torch.cat([fp8_quantize(part).dequantize() for part in x.split(192, dim=1)], dim=1),
                               fp8_quantize(x).dequantize(), rtol=0, atol=0)


def test_weight_amax_cache_requires_explicit_loader_invalidation():
    calls = []
    recipe = W4A8Recipe(weight_amax_reducer=lambda a: calls.append(a.item()) or a)
    w = torch.randn(32, 32).bfloat16()
    recipe.weight(w)
    recipe.weight(w)
    assert len(calls) == 1
    recipe.invalidate_weight_cache()
    recipe.weight(w)
    assert len(calls) == 2


def test_reject_bad_expert_splits():
    with pytest.raises(ValueError, match="splits"):
        grouped_linear(torch.zeros(3, 32).bfloat16(), [torch.zeros(32, 32).bfloat16()], [2], recipe=W4A8Recipe())
def test_inference_weight_cache_preserves_values_and_invalidates_hot_updates():
    import torch
    from slime_plugins.w4a8_sim.recipe import W4A8Recipe
    cached = W4A8Recipe(cache_inference_weights=True)
    reference = W4A8Recipe()
    w = torch.randn(48, 64).bfloat16()
    with torch.no_grad():
        first = cached.weight(w)
        torch.testing.assert_close(first, reference.weight(w), rtol=0, atol=0)
        assert cached.weight(w).data_ptr() == first.data_ptr()
        w.data.copy_(torch.randn_like(w))
        cached.invalidate_weight_cache()
        second = cached.weight(w)
        torch.testing.assert_close(second, reference.weight(w), rtol=0, atol=0)
        assert not torch.equal(first, second)
        assert cached.counters['weight_cache_hits'] == 1
