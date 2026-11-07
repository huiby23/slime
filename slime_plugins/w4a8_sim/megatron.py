"""Dense projections and grouped experts for TP=PP=CP=1, EP>=1.

Expert-parallel token dispatch, combination, gradient synchronization and
optimizer offload remain native Megatron. No global monkey-patching is used.
"""

import copy
import os
from types import MethodType

import torch

from .recipe import W4A8Recipe, grouped_linear, linear


def _forward(self, x):
    if hasattr(self, "layer_norm_weight"):
        if self.normalization != "RMSNorm":
            raise ValueError("The Qwen3 adapter requires RMSNorm")
        gamma = self.layer_norm_weight.float()
        if self.zero_centered_gamma:
            gamma = gamma + 1
        x = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps) * gamma).to(x.dtype)
    bias = getattr(self, "bias", None) if self.use_bias else None
    output = linear(x, self.weight, None if self.te_return_bias else bias, recipe=self._w4a8_recipe)
    self.is_first_microbatch = False
    return output, bias if self.te_return_bias else None


def _grouped_forward(self, x, m_splits):
    weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
    result = grouped_linear(x, weights, m_splits, recipe=self._w4a8_recipe)
    self.is_first_microbatch = False
    return result, None


def install(model, recipe=None):
    from megatron.core.extensions.transformer_engine import TELinear, TELayerNormColumnParallelLinear, TEGroupedLinear

    recipe = recipe or W4A8Recipe()
    torch.backends.cuda.matmul.allow_tf32 = False
    names, grouped = [], []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"):
            continue
        if hasattr(module, "_w4a8_recipe"):
            raise RuntimeError(f"W4A8 already installed at {name}")
        if getattr(module, "tp_size", 1) != 1 or getattr(module, "sequence_parallel", False):
            raise ValueError("This adapter supports TP=1, no sequence parallelism; use EP to shard experts")
        if getattr(module, "delay_wgrad_compute", False):
            raise ValueError("Delayed WGRAD is not supported by the simulator")
        if isinstance(module, TEGroupedLinear):
            if getattr(module, "use_bias", False) or getattr(module, "single_grouped_weight", False):
                raise ValueError("Use separate, bias-free expert weights")
            for i in range(module.num_gemms):
                if not hasattr(module, f"weight{i}"):
                    raise ValueError("Packed grouped expert weights are not supported")
            method = _grouped_forward
            grouped.append(name)
        elif isinstance(module, (TELinear, TELayerNormColumnParallelLinear)):
            method = _forward
        else:
            raise TypeError(f"Unsupported projection {name}: {type(module).__name__}")
        module._w4a8_recipe = recipe
        module.forward = MethodType(method, module)
        names.append(name)
    expected = 4 * len(model.decoder.layers)
    if len(names) != expected:
        raise RuntimeError(f"W4A8 projection coverage mismatch: expected {expected}, found {len(names)}")
    model._w4a8_recipe = recipe
    model._w4a8_modules = tuple(names)
    model._w4a8_grouped_modules = tuple(grouped)
    return model


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    from megatron.training import get_args
    from slime.backends.megatron_utils.model_provider import _get_model_provider_func

    args = get_args()
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_tensor_parallel_size",
    ):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"W4A8 requires {name}=1; expert parallelism is supported")
    for name in (
        "fp4",
        "fp8",
        "sequence_parallel",
        "delay_wgrad_compute",
        "tp_comm_overlap",
        "moe_single_grouped_weight",
    ):
        if getattr(args, name, None):
            raise ValueError(f"Disable {name} for W4A8 numerical simulation")
    if getattr(args, "num_experts", None) and not args.moe_grouped_gemm:
        raise ValueError("MoE W4A8 requires --moe-grouped-gemm")
    base_args = copy.copy(args)
    base_args.custom_model_provider_path = None
    route = getattr(args, "w4a8_attention", None) or os.environ.get("W4A8_ATTENTION", "c16")
    spec_functions = {
        "c16": "get_c16_reference_spec",
        "naive-c8": "get_naive_c8_spec",
        "naive-c4": "get_naive_c4_spec",
        "attn-qat-c4": "get_attn_qat_c4_spec",
    }
    if route not in spec_functions:
        raise ValueError(f"Unsupported W4A8 attention route: {route}")
    base_args.spec = ("w4a8_sim.attention", spec_functions[route])
    provider = _get_model_provider_func(base_args)
    return install(provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage))
