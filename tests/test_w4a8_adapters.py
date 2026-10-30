"""CPU checks of the actual Megatron/TE and SGLang Python interfaces."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from slime_plugins.w4a8_sim.recipe import W4A8Recipe, linear

NUM_GPUS = 0


@pytest.fixture
def modules(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TEColumnParallelGroupedLinear,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        ffn_hidden_size=64,
        num_moe_experts=3,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        normalization="RMSNorm",
    )

    def make(grouped=False, normalized=False):
        with patch("torch.cuda.is_available", return_value=True):
            if grouped:
                module = TEColumnParallelGroupedLinear(
                    3,
                    32,
                    32,
                    config=config,
                    init_method=torch.nn.init.normal_,
                    bias=False,
                    skip_bias_add=False,
                    is_expert=True,
                )
            else:
                cls = TELayerNormColumnParallelLinear if normalized else TEColumnParallelLinear
                module = cls(
                    32,
                    32,
                    config=config,
                    init_method=torch.nn.init.normal_,
                    gather_output=False,
                    bias=False,
                    skip_bias_add=False,
                    is_expert=False,
                )
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if parameter.ndim == 2:
                    parameter.normal_(std=0.1)
        return module

    try:
        yield make
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_real_grouped_experts_and_dense_projection_coverage(modules):
    from slime_plugins.w4a8_sim.megatron import install

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attention = torch.nn.Module()
    layer.self_attention.linear_qkv = modules(normalized=True)
    layer.self_attention.linear_proj = modules()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.linear_fc1 = modules(grouped=True)
    layer.mlp.experts.linear_fc2 = modules(grouped=True)
    model.decoder.layers = torch.nn.ModuleList([layer])
    model.output_layer = torch.nn.Linear(32, 32, dtype=torch.bfloat16)
    keys = set(model.state_dict())
    install(model)
    assert set(model.state_dict()) == keys
    assert len(model._w4a8_modules) == 4 and len(model._w4a8_grouped_modules) == 2
    assert not hasattr(model.output_layer, "_w4a8_recipe")
    x = torch.randn(9, 32).bfloat16().requires_grad_()
    y, _ = layer.self_attention.linear_qkv(x)
    y, _ = layer.self_attention.linear_proj(y)
    y, _ = layer.mlp.experts.linear_fc1(y, torch.tensor([4, 0, 5]))
    y, _ = layer.mlp.experts.linear_fc2(y, torch.tensor([4, 0, 5]))
    y.float().square().mean().backward()
    assert x.grad.isfinite().all() and x.grad.norm() > 0
    assert layer.self_attention.linear_qkv.layer_norm_weight.grad.norm() > 0
    assert layer.mlp.experts.linear_fc1.weight0.grad.norm() > 0
    assert layer.mlp.experts.linear_fc1.weight1.grad.count_nonzero() == 0
    with pytest.raises(RuntimeError, match="already"):
        install(model)


def test_sglang_linear_adapter_matches_training_forward_and_hot_updates():
    from slime_plugins.w4a8_sim.sglang_linear import W4A8LinearMethod

    recipe = W4A8Recipe()
    method = W4A8LinearMethod(recipe)
    layer = SimpleNamespace(weight=torch.randn(48, 32).bfloat16())
    x = torch.randn(9, 32).bfloat16()
    before = method.apply(layer, x)
    torch.testing.assert_close(before, linear(x, layer.weight, recipe=W4A8Recipe()), rtol=0, atol=0)
    layer.weight.data.copy_(torch.randn_like(layer.weight))
    after = method.apply(layer, x)
    assert not torch.equal(before, after)
    torch.testing.assert_close(after, linear(x, layer.weight, recipe=W4A8Recipe()), rtol=0, atol=0)


def test_dense_adapter_import_does_not_require_the_moe_adapter(monkeypatch):
    import importlib
    import sys

    from slime_plugins.w4a8_sim.sglang_linear import W4A8LinearMethod

    dense_name = "slime_plugins.w4a8_sim.sglang_models.qwen3"
    moe_name = "slime_plugins.w4a8_sim.sglang_models.qwen3_moe"
    with monkeypatch.context() as context:
        context.setitem(sys.modules, moe_name, None)
        context.delitem(sys.modules, dense_name, raising=False)
        dense = importlib.import_module(dense_name)
        assert dense.W4A8LinearMethod is W4A8LinearMethod

    # Preserve the old import path used by existing launch and validation code.
    assert importlib.import_module(moe_name).W4A8LinearMethod is W4A8LinearMethod


def test_router_computes_in_fp32_without_operand_quantization():
    from slime_plugins.w4a8_sim.sglang_models.qwen3_moe import FP32RouterMethod

    x = torch.randn(5, 32).bfloat16()
    layer = SimpleNamespace(weight=torch.randn(8, 32).bfloat16())
    result = FP32RouterMethod().apply(layer, x)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, x.float() @ layer.weight.float().T, rtol=0, atol=0)
