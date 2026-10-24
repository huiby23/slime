"""CPU checks of real backend Python interfaces (native CUDA is checked separately)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from slime_plugins.nvfp4_sim.recipe import NVFP4Recipe, nvfp4_linear

NUM_GPUS = 0


@pytest.fixture
def te_layers(tmp_path):
    pytest.importorskip("megatron.core")
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELayerNormColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def make(normalized=False, bias=False, skip_bias=False):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=2,
            ffn_hidden_size=64,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            normalization="RMSNorm",
        )
        cls = TELayerNormColumnParallelLinear if normalized else TEColumnParallelLinear
        # TE's base constructor rejects a CPU-only machine, even with explicit
        # CPU parameter initialization. Bypass only that availability guard;
        # restore it BEFORE invoking any simulator forward/backward operation.
        # These tests do not exercise or claim to validate native CUDA kernels.
        with patch("torch.cuda.is_available", return_value=True):
            layer = cls(
                32,
                32,
                config=config,
                init_method=torch.nn.init.normal_,
                gather_output=False,
                bias=bias,
                skip_bias_add=skip_bias,
                is_expert=False,
            )
        with torch.no_grad():
            layer.weight.normal_(std=0.1)
            if bias:
                layer.bias.zero_()
        return layer

    try:
        yield make
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("bias,skip_bias", [(False, False), (True, False), (True, True)])
def test_real_te_layer_interface(te_layers, normalized, bias, skip_bias):
    from types import MethodType

    from slime_plugins.nvfp4_sim.megatron import _forward

    layer = te_layers(normalized, bias, skip_bias)
    layer._nvfp4_recipe = NVFP4Recipe()
    layer.forward = MethodType(_forward, layer)
    x = torch.randn(19, 1, 32).bfloat16().requires_grad_()
    output, returned_bias = layer(x)
    assert (returned_bias is not None) == (bias and skip_bias)
    if returned_bias is not None:
        output = output + returned_bias
    output.float().square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.norm() > 0
    if normalized:
        assert layer.layer_norm_weight.grad.norm() > 0
    assert layer._nvfp4_recipe.counters["dgrad"] == layer._nvfp4_recipe.counters["wgrad"] == 1


def test_coverage_fp32_master_updates_and_checkpoint(te_layers, tmp_path):
    from slime_plugins.nvfp4_sim.megatron import install

    model = torch.nn.Module()
    model.post_process = True
    model.decoder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attention = torch.nn.Module()
    layer.self_attention.linear_qkv = te_layers(True)
    layer.self_attention.linear_proj = te_layers()
    layer.mlp = torch.nn.Module()
    layer.mlp.linear_fc1 = te_layers(True)
    layer.mlp.linear_fc2 = te_layers()
    model.decoder.layers = torch.nn.ModuleList([layer])
    model.output_layer = te_layers()
    before_keys = set(model.state_dict())
    install(model)
    assert len(model._nvfp4_modules) == 5
    assert before_keys == set(model.state_dict())
    parameters = list(model.parameters())
    masters = [torch.nn.Parameter(p.detach().float().clone()) for p in parameters]
    optimizer = torch.optim.AdamW(masters, lr=0.01)
    for p in parameters:
        if p.ndim == 2:
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
            p.grad_added_to_main_grad = False
    initial = layer.mlp.linear_fc1.weight.detach().clone()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        for p in parameters:
            if hasattr(p, "main_grad"):
                p.main_grad.zero_()
                p.grad_added_to_main_grad = False
        x = torch.randn(19, 1, 32).bfloat16().requires_grad_()
        for module in (
            layer.self_attention.linear_qkv,
            layer.self_attention.linear_proj,
            layer.mlp.linear_fc1,
            layer.mlp.linear_fc2,
            model.output_layer,
        ):
            x, _ = module(x)
        x.float().square().mean().backward()
        for p, master in zip(parameters, masters, strict=True):
            grad = p.main_grad if hasattr(p, "main_grad") else p.grad.float()
            assert torch.isfinite(grad).all()
            master.grad = grad.clone()
        optimizer.step()
        with torch.no_grad():
            for p, master in zip(parameters, masters, strict=True):
                p.copy_(master)
    assert not torch.equal(initial, layer.mlp.linear_fc1.weight)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    for key in ("fprop", "dgrad", "wgrad", "rht_x", "rht_g", "sr_dgrad", "sr_wgrad"):
        assert model._nvfp4_recipe.counters[key] == 10
    with pytest.raises(RuntimeError, match="already installed"):
        install(model)


def test_sglang_method_requantizes_in_place_weight_updates():
    pytest.importorskip("sglang")
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    from slime_plugins.nvfp4_sim.sglang_models.qwen3 import NVFP4SimLinearMethod

    recipe = NVFP4Recipe()
    method = NVFP4SimLinearMethod(recipe)
    layer = SimpleNamespace(weight=torch.randn(48, 32).bfloat16())
    x = torch.randn(19, 32).bfloat16()
    bias = torch.randn(48).bfloat16()
    expected = nvfp4_linear(x, layer.weight, bias, recipe=NVFP4Recipe())
    initial = method.apply(layer, x, bias)
    torch.testing.assert_close(initial, expected, rtol=0, atol=0)
    assert should_apply_lm_head_quant_method(layer, method)
    layer.weight.copy_(torch.randn_like(layer.weight))
    updated = method.apply(layer, x, bias)
    assert not torch.equal(initial, updated)
    torch.testing.assert_close(updated, nvfp4_linear(x, layer.weight, bias, recipe=NVFP4Recipe()), rtol=0, atol=0)


def test_inference_cache_preserves_values_and_reuses_only_weight_casts():
    pytest.importorskip("sglang")
    from slime_plugins.nvfp4_sim.sglang_models import qwen3

    method = qwen3.NVFP4SimLinearMethod(NVFP4Recipe(), cache_weights=True)
    layer = SimpleNamespace(weight=torch.randn(48, 32).bfloat16())
    x = torch.randn(19, 32).bfloat16()
    with patch.object(qwen3, "quantize", wraps=qwen3.quantize) as quantizer:
        first = method.apply(layer, x)
        second = method.apply(layer, x)
        assert quantizer.call_count == 3  # Two activation casts, one weight cast.
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    layer.weight.copy_(torch.randn_like(layer.weight))
    updated = method.apply(layer, x)
    torch.testing.assert_close(updated, nvfp4_linear(x, layer.weight, recipe=NVFP4Recipe()), rtol=0, atol=0)


def test_model_loader_invalidates_cache_even_for_data_writes():
    pytest.importorskip("sglang")
    from slime_plugins.nvfp4_sim.sglang_models import qwen3

    model = qwen3.NVFP4SimQwen3ForCausalLM.__new__(qwen3.NVFP4SimQwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.projection = torch.nn.Linear(32, 48, bias=False, dtype=torch.bfloat16)
    method = qwen3.NVFP4SimLinearMethod(NVFP4Recipe(), cache_weights=True)
    model.projection.quant_method = method
    x = torch.randn(19, 32).bfloat16()
    initial = method.apply(model.projection, x)

    def load(instance, weights):
        instance.projection.weight.data.copy_(weights)
        return {"projection.weight"}

    with patch.object(qwen3.Qwen3ForCausalLM, "load_weights", load):
        model.load_weights(torch.randn_like(model.projection.weight))
    assert method._weight_cache is None
    updated = method.apply(model.projection, x)
    assert not torch.equal(initial, updated)
    torch.testing.assert_close(updated, nvfp4_linear(x, model.projection.weight, recipe=NVFP4Recipe()), rtol=0, atol=0)


def test_native_megatron_head_keeps_tied_embedding_gradient(te_layers):
    from types import MethodType

    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    from slime_plugins.nvfp4_sim.megatron import _output_forward
    from slime_plugins.nvfp4_sim.recipe import gemm, quantize, rht_transpose

    config = te_layers().config
    head = ColumnParallelLinear(
        32, 32, config=config, init_method=torch.nn.init.normal_, bias=False, skip_weight_param_allocation=True
    )
    head._nvfp4_recipe = NVFP4Recipe(stochastic_rounding=False)
    head.forward = MethodType(_output_forward, head)
    shared = torch.nn.Parameter(torch.randn(32, 32).bfloat16())
    shared.main_grad = torch.zeros_like(shared, dtype=torch.float32)
    shared.grad_added_to_main_grad = False
    # GPTModel.setup_embeddings_and_output_layer sets this for tied weights.
    shared.zero_out_wgrad = True
    x = torch.randn(19, 1, 32).bfloat16().requires_grad_()
    g = torch.randn(19, 1, 32).bfloat16()
    tokens = torch.tensor([0, 1, 1, 4])
    logits, _ = head(x, weight=shared, runtime_gather_output=False)
    (logits.float().mul(g.float()).sum() + torch.nn.functional.embedding(tokens, shared).float().sum()).backward()
    head_grad = gemm(quantize(rht_transpose(g.reshape(-1, 32))), quantize(rht_transpose(x.detach().reshape(-1, 32))))
    embedding_grad = torch.bincount(tokens, minlength=32).float()[:, None].expand(32, 32)
    torch.testing.assert_close(shared.main_grad, head_grad, rtol=0, atol=0)
    torch.testing.assert_close(shared.grad.float(), embedding_grad, rtol=0, atol=0)
    # The real DDP hook adds param.grad when zero_out_wgrad=True, even when
    # grad_added_to_main_grad=True. The dummy gradient must therefore be zero.
    assert shared.grad_added_to_main_grad and shared.zero_out_wgrad
    shared.main_grad.add_(shared.grad)
    torch.testing.assert_close(shared.main_grad, head_grad + embedding_grad, rtol=0, atol=0)


def test_sglang_tied_head_keeps_embedding_lookup():
    pytest.importorskip("sglang")
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    from slime_plugins.nvfp4_sim.sglang_models.qwen3 import NVFP4SimEmbeddingLinearMethod

    layer = SimpleNamespace(weight=torch.randn(48, 32).bfloat16())
    method = NVFP4SimEmbeddingLinearMethod(NVFP4Recipe())
    indices = torch.tensor([0, 7, 7, 12])
    torch.testing.assert_close(method.embedding(layer, indices), layer.weight[indices], rtol=0, atol=0)
    assert should_apply_lm_head_quant_method(layer, method)
    x = torch.randn(19, 32).bfloat16()
    torch.testing.assert_close(
        method.apply(layer, x), nvfp4_linear(x, layer.weight, recipe=NVFP4Recipe()), rtol=0, atol=0
    )


def test_sglang_constructor_covers_tied_lm_head_alias(monkeypatch):
    pytest.importorskip("sglang")
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod, UnquantizedLinearMethod

    from slime_plugins.nvfp4_sim.sglang_models import qwen3

    # Exercise the real adapter constructor over a CPU model skeleton, without
    # initializing SGLang's CUDA attention backend or a distributed scheduler.
    def base_init(self, config):
        torch.nn.Module.__init__(self)
        self.pp_group = SimpleNamespace(world_size=1)
        self.quant_config = None
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(48, 32, dtype=torch.bfloat16)
        self.model.embed_tokens.quant_method = UnquantizedEmbeddingMethod()
        layers = []
        for _ in range(config.num_hidden_layers):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            for owner, name in (
                (layer.self_attn, "qkv_proj"),
                (layer.self_attn, "o_proj"),
                (layer.mlp, "gate_up_proj"),
                (layer.mlp, "down_proj"),
            ):
                projection = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
                projection.quant_method = UnquantizedLinearMethod()
                setattr(owner, name, projection)
            layers.append(layer)
        self.model.layers = torch.nn.ModuleList(layers)
        self.lm_head = self.model.embed_tokens

    monkeypatch.setattr(qwen3.Qwen3ForCausalLM, "__init__", base_init)
    monkeypatch.setattr(qwen3, "get_tensor_model_parallel_world_size", lambda: 1)
    model = qwen3.NVFP4SimQwen3ForCausalLM(SimpleNamespace(num_hidden_layers=2))
    assert len(model._nvfp4_modules) == 9 and "lm_head" in model._nvfp4_modules
    assert model.lm_head is model.model.embed_tokens
    assert isinstance(model.lm_head.quant_method, qwen3.NVFP4SimEmbeddingLinearMethod)
    indices = torch.tensor([0, 7, 12])
    torch.testing.assert_close(
        model.model.embed_tokens.quant_method.embedding(model.model.embed_tokens, indices),
        model.lm_head.weight[indices],
        rtol=0,
        atol=0,
    )
