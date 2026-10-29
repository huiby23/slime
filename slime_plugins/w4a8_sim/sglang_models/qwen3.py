"""W4A8 operands for stock dense Qwen3, preserving loaders and TP collectives."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from sglang.srt.distributed import get_tp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from ..recipe import W4A8Recipe
from .qwen3_moe import W4A8LinearMethod


class W4A8SimQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        torch.backends.cuda.matmul.allow_tf32 = False
        if self.pp_group.world_size != 1 or getattr(self, "quant_config", None) is not None:
            raise ValueError("W4A8 loads original BF16 dense weights with rollout PP=1")
        recipe = W4A8Recipe(cache_inference_weights=True)
        if get_tensor_model_parallel_world_size() > 1:
            def reduce_amax(amax):
                dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=get_tp_group().device_group)
                return amax
            recipe.weight_amax_reducer = reduce_amax
        names = []
        for name, module in self.named_modules():
            if name.rsplit(".", 1)[-1] in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                if not isinstance(module.quant_method, UnquantizedLinearMethod):
                    raise TypeError(f"Expected original BF16 projection: {name}")
                module.quant_method = W4A8LinearMethod(recipe)
                names.append(name)
        if len(names) != 4 * config.num_hidden_layers:
            raise RuntimeError(f"Incomplete dense W4A8 coverage: {names}")
        self._w4a8_recipe, self._w4a8_modules = recipe, tuple(names)
        self._w4a8_audit_written = False

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        directory = os.environ.get("W4A8_ROLLOUT_AUDIT_DIR")
        if directory and not self._w4a8_audit_written:
            path = Path(directory)
            if os.environ.get("W4A8_AUDIT_PER_ENGINE") == "1":
                path = path / f"engine-{os.getpid()}"
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rollout-rank-{get_tp_group().rank_in_group}.json").write_text(json.dumps(dict(
                architecture=type(self).__name__, projections=self._w4a8_modules,
                counters=dict(self._w4a8_recipe.counters),
                tp_size=get_tensor_model_parallel_world_size()), indent=2) + "\n")
            self._w4a8_audit_written = True
        return result

    def load_weights(self, *args, **kwargs):
        try:
            return super().load_weights(*args, **kwargs)
        finally:
            if hasattr(self, "_w4a8_recipe"):
                self._w4a8_recipe.invalidate_weight_cache()
                self._w4a8_audit_written = False


EntryClass = W4A8SimQwen3ForCausalLM
