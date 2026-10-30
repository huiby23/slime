"""Preserve stock Qwen3 MoE loaders, router and tensor parallel collectives.

The rollout uses TP>=1, EP=PP=1, the Triton backend and standard top-k outputs.
Quantized inference is Python numerical simulation with CUDA graphs disabled.
"""

from types import MethodType
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from sglang.srt.distributed import get_tp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod, UnquantizedFusedMoEMethod
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM

from ..recipe import W4A8Recipe, routed_experts
from ..sglang_linear import W4A8LinearMethod


class FP32RouterMethod(UnquantizedLinearMethod):
    """Match Megatron --moe-router-dtype fp32 without quantizing the router."""

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x.float(), layer.weight.float(), None if bias is None else bias.float())


def _experts_forward(self, hidden_states, topk_output):
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise ValueError("W4A8 requires standard top-k outputs, no bypassed router")
    result = routed_experts(
        hidden_states,
        self.w13_weight,
        self.w2_weight,
        topk_output.topk_ids,
        topk_output.topk_weights,
        recipe=self._w4a8_recipe,
    )
    callback = getattr(self._w4a8_recipe, "audit_callback", None)
    if callback is not None:
        callback()
    return result


class W4A8SimQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        torch.backends.cuda.matmul.allow_tf32 = False
        if self.pp_group.world_size != 1 or getattr(self, "quant_config", None) is not None:
            raise ValueError("W4A8 loads original BF16 weights with rollout PP=1")
        if not get_moe_runner_backend().is_triton() or not get_moe_a2a_backend().is_none():
            raise ValueError("W4A8 simulation requires --moe-runner-backend triton, no MoE all-to-all backend")
        recipe = W4A8Recipe()
        if get_tensor_model_parallel_world_size() > 1:

            def reduce_amax(amax):
                dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=get_tp_group().device_group)
                return amax

            recipe.weight_amax_reducer = reduce_amax
        names, experts, routers = [], [], []
        for name, module in self.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("qkv_proj", "o_proj"):
                if not isinstance(module.quant_method, UnquantizedLinearMethod):
                    raise TypeError(f"Expected original linear at {name}")
                module.quant_method = W4A8LinearMethod(recipe)
                names.append(name)
            elif leaf == "experts":
                if not isinstance(module, FusedMoE) or not isinstance(module.quant_method, UnquantizedFusedMoEMethod):
                    raise TypeError(f"Expected unpacked original MoE at {name}")
                if module.moe_ep_size != 1 or module.activation != "silu":
                    raise ValueError("W4A8 rollout supports EP=1 SwiGLU experts")
                module._w4a8_recipe = recipe
                module.forward = MethodType(_experts_forward, module)
                experts.append(name)
            elif leaf == "gate" and name.endswith(".mlp.gate"):
                if not isinstance(module.quant_method, UnquantizedLinearMethod):
                    raise TypeError(f"Expected unquantized router at {name}")
                module.quant_method = FP32RouterMethod()
                routers.append(name)
        if (
            len(names) != config.num_hidden_layers * 2
            or len(experts) != config.num_hidden_layers
            or len(routers) != config.num_hidden_layers
        ):
            raise RuntimeError(f"Incomplete W4A8 coverage: {len(names)} projections, {len(experts)} expert banks")
        self._w4a8_recipe = recipe
        self._w4a8_modules = tuple(names + experts)
        self._w4a8_routers = tuple(routers)
        self._w4a8_audit_written = False
        audit_dir = os.environ.get("W4A8_ROLLOUT_AUDIT_DIR")
        if audit_dir:

            def audit():
                if self._w4a8_audit_written:
                    return
                destination = Path(audit_dir)
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / f"rollout-rank-{get_tp_group().rank_in_group}.json"
                path.write_text(
                    json.dumps(
                        dict(
                            architecture=type(self).__name__,
                            projections=names,
                            expert_banks=experts,
                            fp32_routers=routers,
                            counters=dict(recipe.counters),
                            tp_size=get_tensor_model_parallel_world_size(),
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                self._w4a8_audit_written = True

            recipe.audit_callback = audit

    def load_weights(self, *args, **kwargs):
        try:
            return super().load_weights(*args, **kwargs)
        finally:
            if hasattr(self, "_w4a8_recipe"):
                self._w4a8_recipe.invalidate_weight_cache()
                self._w4a8_audit_written = False


EntryClass = W4A8SimQwen3MoeForCausalLM
