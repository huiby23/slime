"""CPU contracts for the audited sampler adapter, using a tiny fake loader.

Actual quantization arithmetic and weight updates run in Torch. Only SGLang's
model-construction interface is replaced; these tests do not validate CUDA or
distributed transport.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slime_plugins.nvfp4_sim.recipe import NVFP4Recipe, nvfp4_linear

NUM_GPUS = 0


@pytest.fixture
def runtime(monkeypatch):
    class UnquantizedLinearMethod:
        pass

    class UnquantizedEmbeddingMethod:
        pass

    class Linear(torch.nn.Module):
        def __init__(self, embedding=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(16, dtype=torch.bfloat16))
            cls = UnquantizedEmbeddingMethod if embedding else UnquantizedLinearMethod
            self.quant_method = cls()

    class FakeQwen(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.pp_group = SimpleNamespace(world_size=1)
            self.quant_config = None
            self.model = torch.nn.Module()
            self.model.embed_tokens = Linear(embedding=True)
            layers = []
            for _ in range(config.num_hidden_layers):
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.qkv_proj = Linear()
                layer.self_attn.o_proj = Linear()
                layer.mlp = torch.nn.Module()
                layer.mlp.gate_up_proj = Linear()
                layer.mlp.down_proj = Linear()
                layers.append(layer)
            self.model.layers = torch.nn.ModuleList(layers)
            self.lm_head = self.model.embed_tokens

        def load_weights(self, weights):
            params = dict(self.named_parameters(remove_duplicate=False))
            loaded = set()
            for name, value in weights:
                if name == "fail_after_partial_load":
                    raise RuntimeError("stock loader failed")
                # Exercise the cache bug's exact trigger: native loaders can
                # write through .data without incrementing tensor._version.
                params[name].data.copy_(value)
                loaded.add(name)
            return loaded

        def forward(self, x):
            for layer in self.model.layers:
                for module in (
                    layer.self_attn.qkv_proj,
                    layer.self_attn.o_proj,
                    layer.mlp.gate_up_proj,
                    layer.mlp.down_proj,
                ):
                    x = module.quant_method.apply(module, x)
            return self.lm_head.quant_method.apply(self.lm_head, x)

    packages = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.models",
    )
    for name in packages:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    distributed = ModuleType("sglang.srt.distributed")
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    unquant = ModuleType("sglang.srt.layers.quantization.unquant")
    unquant.UnquantizedLinearMethod = UnquantizedLinearMethod
    unquant.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod
    qwen = ModuleType("sglang.srt.models.qwen3")
    qwen.Qwen3ForCausalLM = FakeQwen
    for module in (distributed, unquant, qwen):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    source = Path(__file__).resolve().parents[1] / "slime_plugins/tim/nvfp4_models/qwen3.py"
    spec = importlib.util.spec_from_file_location("_tim_nvfp4_contract_runtime", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_config(audit_path=None):
    config = SimpleNamespace(num_hidden_layers=1, nvfp4_sim_weight_cache=True)
    if audit_path is not None:
        config.nvfp4_sim_audit_path = str(audit_path)
    return config


def inputs():
    return (torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32).bfloat16()


def test_sync_invalidates_and_rebuilds_caches_without_mutating_payload(runtime, tmp_path):
    rng = torch.get_rng_state().clone()
    model = runtime.NVFP4SimQwen3ForCausalLM(model_config(tmp_path / "quantization.json"))
    original = model(inputs())
    methods = model._quantization_audit_methods()
    assert all(method._weight_cache is not None for method in methods)
    name = "model.layers.0.self_attn.qkv_proj.weight"
    parameter = dict(model.named_parameters())[name]
    version = parameter._version
    fresh = torch.eye(16, dtype=torch.bfloat16) * 2
    payload_copy = fresh.clone()
    assert model.load_weights([(name, fresh)]) == {name}
    assert parameter._version == version
    assert torch.equal(fresh, payload_copy)
    assert all(method._weight_cache is None and method._weight_key is None for method in methods)
    changed = model(inputs())
    assert not torch.equal(original, changed)
    audit = json.loads(model._nvfp4_audit_path.read_text())
    assert audit["installed_linears"] == audit["expected_linears"] == 5
    assert audit["load_weights_calls"] == audit["cache_invalidations_verified"] == 1
    assert audit["cache_rebuilds_verified"] == 2
    assert audit["cached_linears"] == audit["latest_rebuilt_linears"] == 5
    assert audit["last_rebuilt_weight_generation"] == audit["weight_generation"] == 1
    assert audit["load_weights_failures"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_partial_loader_failure_still_invalidates_every_cached_weight(runtime, tmp_path):
    model = runtime.NVFP4SimQwen3ForCausalLM(model_config(tmp_path / "quantization.json"))
    model(inputs())
    name = "model.layers.0.self_attn.qkv_proj.weight"
    with pytest.raises(RuntimeError, match="stock loader failed"):
        model.load_weights([(name, torch.eye(16, dtype=torch.bfloat16) * 2), ("fail_after_partial_load", None)])
    assert all(method._weight_cache is None for method in model._quantization_audit_methods())
    audit = json.loads(model._nvfp4_audit_path.read_text())
    assert audit["load_weights_failures"] == 1
    assert audit["weight_generation"] == audit["cache_invalidations_verified"] == 0
    assert audit["latest_invalidated_linears"] == 5


def test_four_replicas_publish_separate_audits(runtime, monkeypatch, tmp_path):
    paths = []
    for pid in range(100, 104):
        # Replace this module's os binding, leaving process-wide os untouched.
        monkeypatch.setattr(runtime, "os", SimpleNamespace(getpid=lambda value=pid: value, environ={}))
        model = runtime.NVFP4SimQwen3ForCausalLM(model_config(tmp_path / "quantization.json"))
        model(inputs())
        paths.append(model._nvfp4_audit_path)
    assert len(set(paths)) == 4
    assert all(path.parent == tmp_path / "quantization.workers" for path in paths)
    assert {json.loads(path.read_text())["worker_pid"] for path in paths} == set(range(100, 104))
    assert all(path.with_suffix(".jsonl").is_file() for path in paths)
    assert not (tmp_path / "quantization.json").exists()


def test_audit_is_optional_and_cache_remains_correct(runtime, tmp_path):
    model = runtime.NVFP4SimQwen3ForCausalLM(model_config())
    initial = model(inputs())
    model.load_weights([("model.layers.0.self_attn.qkv_proj.weight", torch.eye(16, dtype=torch.bfloat16) * 2)])
    assert not torch.equal(initial, model(inputs()))
    assert model._nvfp4_audit_path is None
    assert not list(tmp_path.iterdir())


def test_linear_arithmetic_matches_shared_recipe(runtime):
    layer = SimpleNamespace(weight=(torch.arange(256).reshape(16, 16) / 256).bfloat16())
    method = runtime.NVFP4SimLinearMethod(NVFP4Recipe(), cache_weights=True)
    expected = nvfp4_linear(inputs(), layer.weight, recipe=NVFP4Recipe())
    actual = method.apply(layer, inputs())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
