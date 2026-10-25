"""Numerical contracts and backward-path checks for the NVFP4 recipe simulator."""

import importlib
import inspect

import pytest
import torch

from slime_plugins.nvfp4_sim.recipe import (
    FP4_VALUES,
    NVFP4Recipe,
    _constants,
    gemm,
    nvfp4_linear,
    quantize,
    rht_transpose,
    round_e2m1,
    round_e4m3,
)

NUM_GPUS = 0


def test_diagnostic_sr_generator_is_independent_of_default_rng():
    x = torch.randn(19, 32).bfloat16().requires_grad_()
    weight = torch.randn(48, 32).bfloat16().requires_grad_()
    gradients = []
    for seed in [17, 913]:
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(42)
        recipe = NVFP4Recipe(stochastic_generator=generator)
        output = nvfp4_linear(x, weight, recipe=recipe)
        gradients.append(torch.autograd.grad(output.float().square().mean(), (x, weight)))
    for first, second in zip(*gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_grouped_qkv_trace_conversion_preserves_projection():
    from slime_plugins.nvfp4_sim.diagnostic_trace import canonical_qkv

    x = torch.randn(3, 1024)
    weight = torch.randn(4096, 1024)
    actual = canonical_qkv(x @ weight.T)
    expected = x @ canonical_qkv(weight, weight=True).T
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_real_activation_scale_tie_uses_reference_division():
    x = torch.zeros(32, 32, dtype=torch.bfloat16)
    x[0, 0] = 10.0
    x[1, :16] = .9375
    corrected = quantize(x)
    legacy = quantize(x, reference_arithmetic=False)
    assert corrected.scales[1, 0] == 40
    assert legacy.scales[1, 0] == 44


@pytest.mark.parametrize("sign", [1, -1])
def test_e2m1_ties_even_and_saturation(sign):
    x = torch.tensor([0, 0.25, 0.5, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, 100.0]) * sign
    expected = torch.tensor([0, 0, 0.5, 1, 1, 2, 2, 4, 4, 6, 6.0]) * sign
    torch.testing.assert_close(round_e2m1(x), expected, rtol=0, atol=0)


def test_e4m3_matches_pytorch_cast_including_subnormals():
    x = torch.cat((torch.linspace(0, 448, 20000), torch.linspace(0, 2**-5, 20000)))
    torch.testing.assert_close(round_e4m3(x), x.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)


def test_stochastic_rounding_is_unbiased_and_reproducible():
    x = torch.full((100000,), 2.25)
    a = round_e2m1(x, True, torch.Generator().manual_seed(42))
    b = round_e2m1(x, True, torch.Generator().manual_seed(42))
    assert torch.equal(a, b)
    assert set(a.unique().tolist()) == {2.0, 3.0}
    assert abs(a.mean().item() - 2.25) < 0.006
    assert round_e2m1(x).mean() == 2


@pytest.mark.parametrize("weight", [False, True])
@pytest.mark.parametrize("shape", [(17, 31), (16, 16), (0, 32), (32, 0)])
def test_zero_and_padded_tensors(shape, weight):
    q = quantize(torch.zeros(shape).bfloat16(), weight=weight)
    output = q.dequantize()
    assert output.shape == shape
    assert torch.isfinite(output).all()
    assert not output.count_nonzero()


def test_2d_weight_scaling_is_transpose_equivalent():
    torch.manual_seed(7)
    w = torch.randn(37, 29).bfloat16()
    w[0, 0] = 100
    row = quantize(w, weight=True)
    col = quantize(w.T.contiguous(), weight=True)
    torch.testing.assert_close(row.dequantize().T, col.dequantize(), rtol=0, atol=0)
    torch.testing.assert_close(row.transpose_weight().dequantize(), col.dequantize(), rtol=0, atol=0)
    assert torch.equal(row.scales[:16], row.scales[0].expand(16, -1))
    # 1D scaling must be re-quantized from the original transpose.
    assert not torch.equal(quantize(w).dequantize().T, quantize(w.T.contiguous()).dequantize())


@pytest.mark.parametrize("weight", [False, True])
def test_matches_nvidia_reference_quantization(weight):
    # Runtime image pins TE 2.18. Also support the renamed reference in newer TE.
    try:
        ref = importlib.import_module("transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4")
    except ImportError:
        ref = pytest.importorskip("transformer_engine.pytorch.custom_recipes.reference_nvfp4")
    torch.manual_seed(17)
    x = torch.randn(32, 64).bfloat16()
    x[0, :16] = 0
    q = quantize(x, weight=weight)
    cls = ref.NVFP4QuantizerRef
    method = cls._quantize_blockwise_reference
    args = (x, x.abs().max().float().view(1), 16, 16 if weight else 1)
    if next(iter(inspect.signature(method).parameters)) == "cls":
        args = (cls, *args)
    payload, scale = method(*args, pow_2_scales=False, eps=0.0)
    expected_payload = ref.cast_from_fp4x2(payload, torch.float32)
    torch.testing.assert_close(q.values.float(), expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(q.scales, scale.float(), rtol=0, atol=0)


def test_hadamard_orientation_and_shared_signs():
    _, h = _constants(torch.device("cpu"))
    torch.testing.assert_close(h @ h.T, torch.eye(16), rtol=0, atol=0)
    x = torch.arange(19 * 32).reshape(19, 32).bfloat16()
    output = rht_transpose(x)
    padded = torch.nn.functional.pad(x.T, (0, 13))
    expected = (padded.reshape(-1, 16) @ h.bfloat16()).reshape_as(padded)
    assert output.shape == (32, 32)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="BF16"):
        rht_transpose(x.float())


def test_hadamard_matches_nvidia_reference():
    try:
        ref = importlib.import_module("transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4")
        utils = ref.utils
    except ImportError:
        ref = pytest.importorskip("transformer_engine.pytorch.custom_recipes.reference_nvfp4")
        utils = ref.reference_utils
    quantizer = ref.NVFP4QuantizerRef(dtype=utils.Fp4Formats.E2M1, with_rht=True)
    x = torch.randn(32, 48).bfloat16()
    torch.testing.assert_close(rht_transpose(x), quantizer._apply_rht(x.T.contiguous()), rtol=0, atol=0)


def test_block_gemm_matches_fp64_dequantized_oracle():
    torch.manual_seed(8)
    a = quantize(torch.randn(19, 41).bfloat16())
    b = quantize(torch.randn(23, 41).bfloat16(), weight=True)
    expected = (a.dequantize().double() @ b.dequantize().double().T).float()
    torch.testing.assert_close(gemm(a, b), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(gemm(a, b, chunk_blocks=1), expected, rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match="reduction"):
        gemm(a, quantize(torch.zeros(3, 32)))


def test_custom_backward_matches_three_independent_gemms():
    torch.manual_seed(42)
    x = torch.randn(2, 19, 32).bfloat16().requires_grad_()
    w = torch.randn(48, 32).bfloat16().requires_grad_()
    bias = torch.randn(48).bfloat16().requires_grad_()
    g = torch.randn(2, 19, 48).bfloat16()
    recipe = NVFP4Recipe(stochastic_rounding=False)
    output = nvfp4_linear(x, w, bias, recipe=recipe)
    output.backward(g)
    xm, gm = x.detach().reshape(-1, 32), g.reshape(-1, 48)
    expected_y = (gemm(quantize(xm), quantize(w.detach(), weight=True)) + bias.detach().float()).bfloat16()
    expected_dx = gemm(quantize(gm), quantize(w.detach().T.contiguous(), weight=True)).bfloat16()
    expected_dw = gemm(quantize(rht_transpose(gm)), quantize(rht_transpose(xm))).bfloat16()
    torch.testing.assert_close(output.reshape_as(expected_y), expected_y, rtol=0, atol=0)
    torch.testing.assert_close(x.grad.reshape_as(expected_dx), expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(w.grad, expected_dw, rtol=0, atol=0)
    torch.testing.assert_close(bias.grad, gm.float().sum(0).bfloat16(), rtol=0, atol=0)
    assert recipe.counters == {"fprop": 1, "dgrad": 1, "wgrad": 1, "rht_x": 1, "rht_g": 1}


def test_default_recipe_uses_sr_in_both_backward_gemms_and_changes_gradients():
    torch.manual_seed(21)
    x = torch.randn(31, 32).bfloat16().requires_grad_()
    w = torch.randn(48, 32).bfloat16().requires_grad_()
    g = torch.randn(31, 48).bfloat16()
    before = w.detach().clone()
    recipe = NVFP4Recipe()
    nvfp4_linear(x, w, recipe=recipe).backward(g)
    assert recipe.counters["sr_dgrad"] == recipe.counters["sr_wgrad"] == 1
    assert recipe.counters["rht_x"] == recipe.counters["rht_g"] == 1
    assert torch.equal(before, w)
    # Simple forward-only STE/QAT uses G.T @ QDQ(X), without gradient FP4/RHT.
    qat_dw = (g.float().T @ quantize(x.detach()).dequantize()).bfloat16()
    assert not torch.equal(w.grad, qat_dw)
    assert torch.isfinite(w.grad).all() and w.grad.norm() > 0


def test_fp32_main_grad_accumulates_without_double_adding():
    torch.manual_seed(19)
    w = torch.randn(32, 16).bfloat16().requires_grad_()
    w.main_grad = torch.zeros_like(w, dtype=torch.float32)
    w.grad_added_to_main_grad = False
    x = torch.randn(19, 16).bfloat16()
    g = torch.randn(19, 32).bfloat16()
    recipe = NVFP4Recipe(stochastic_rounding=False)
    expected = gemm(quantize(rht_transpose(g)), quantize(rht_transpose(x)))
    for step in (1, 2):
        nvfp4_linear(x, w, recipe=recipe).backward(g)
        assert w.grad_added_to_main_grad
        assert not w.grad.count_nonzero()
        torch.testing.assert_close(w.main_grad, expected * step, rtol=0, atol=0)


def test_recipe_rejects_non_bf16_training():
    with pytest.raises(ValueError, match="BF16"):
        nvfp4_linear(torch.ones(16, 16), torch.ones(16, 16), recipe=NVFP4Recipe())


def test_payload_values_are_e2m1():
    q = quantize(torch.randn(19, 33).bfloat16())
    assert set(q.values.abs().unique().tolist()) <= set(FP4_VALUES)


def test_recipe_is_independent_of_default_dtype():
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        # Explicitly check the RNG dtype, even if constants were already cached.
        x = torch.full((100000,), 2.25, dtype=torch.float32)
        result = round_e2m1(x, True, torch.Generator().manual_seed(42))
        assert abs(result.mean().item() - 2.25) < 0.006
        assert result.dtype == torch.float32
    finally:
        torch.set_default_dtype(old_dtype)
