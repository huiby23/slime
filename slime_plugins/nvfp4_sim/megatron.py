"""Slime custom model provider for the single-GPU dense Qwen3 baseline.

The original TE parameters and checkpoint keys are retained. Only the forward
methods of dense linear modules change; their custom autograd implements all
three NVFP4 GEMMs. No native TE FP4 context should be enabled on an A100.
"""

import copy
import logging
from types import MethodType

import torch
import torch.nn.functional as F

from .recipe import NVFP4Recipe, nvfp4_linear

logger = logging.getLogger(__name__)
PROJECTIONS = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")


def _forward(self, x):
    if hasattr(self, "layer_norm_weight"):
        gamma = self.layer_norm_weight
        if self.zero_centered_gamma:
            gamma = gamma.float() + 1
        if self.normalization == "RMSNorm":
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            x = (normalized * gamma.float()).to(x.dtype)
        elif self.normalization == "LayerNorm":
            x = F.layer_norm(x, (x.shape[-1],), gamma, self.layer_norm_bias, self.eps)
        else:
            raise ValueError(f"Unsupported fused normalization: {self.normalization}")
    bias = self.bias if self.use_bias else None
    output = nvfp4_linear(x, self.weight, None if self.te_return_bias else bias, recipe=self._nvfp4_recipe)
    self.is_first_microbatch = False
    return output, bias if self.te_return_bias else None


def _output_forward(self, input_, weight=None, runtime_gather_output=None):
    """Megatron's LM head uses ColumnParallelLinear and an optional tied weight."""
    weight = self.weight if weight is None else weight
    if weight is None:
        raise ValueError("The tied LM head requires its original embedding weight")
    output = nvfp4_linear(input_, weight, None if self.skip_bias_add else self.bias, recipe=self._nvfp4_recipe)
    return output, self.bias if self.skip_bias_add else None


def install(model, recipe=None, *, include_output_layer=True):
    """Patch every transformer projection, validate coverage, preserve parameters."""
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TELinear
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    recipe = recipe or NVFP4Recipe()
    names = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        selected = leaf in PROJECTIONS or (include_output_layer and leaf == "output_layer")
        if not selected:
            continue
        native_output = leaf == "output_layer" and isinstance(module, ColumnParallelLinear)
        if not native_output and not isinstance(module, (TELinear, TELayerNormColumnParallelLinear)):
            raise TypeError(f"Expected a dense TE linear at {name}, got {type(module).__name__}")
        if getattr(module, "tp_size", 1) != 1 or getattr(module, "sequence_parallel", False):
            raise ValueError("NVFP4 simulation currently requires TP=1 and no sequence parallelism")
        if hasattr(module, "_nvfp4_recipe"):
            raise RuntimeError(f"NVFP4 simulation already installed at {name}")
        module._nvfp4_recipe = recipe
        module.forward = MethodType(_output_forward if native_output else _forward, module)
        names.append(name)
    layers = list(model.decoder.layers)
    expected = len(layers) * len(PROJECTIONS) + int(include_output_layer and model.post_process)
    if len(names) != expected:
        raise RuntimeError(f"NVFP4 linear coverage mismatch: expected {expected}, installed {len(names)}: {names}")
    model._nvfp4_recipe = recipe
    model._nvfp4_modules = tuple(names)
    logger.warning("NVFP4 recipe simulation installed: %d linears; FPROP/DGRAD/WGRAD, RHT, SR, 2D weights", len(names))
    return model


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Use with --custom-model-provider-path nvfp4_sim.megatron.model_provider.

    For a pinned older Slime checkout, add this package's parent directory
    (slime_plugins) to PYTHONPATH ahead of that checkout. No checkout edits are
    needed, and Slime/Megatron stay pinned to the original experiment revisions.
    """
    from megatron.training import get_args

    from slime.backends.megatron_utils.model_provider import _get_model_provider_func

    args = get_args()
    for key in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
    ):
        if getattr(args, key, 1) != 1:
            raise ValueError(f"The baseline NVFP4 simulation requires {key}=1")
    if getattr(args, "num_experts", None):
        raise ValueError("This adapter targets dense Qwen3, not grouped MoE linears")
    for key in ("fp4", "fp8", "sequence_parallel", "delay_wgrad_compute", "tp_comm_overlap", "use_te_activation_func"):
        if getattr(args, key, None):
            raise ValueError(f"Disable {key} when using the software NVFP4 recipe")
    base_args = copy.copy(args)
    base_args.custom_model_provider_path = None
    provider = _get_model_provider_func(base_args)
    model = provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    recipe = NVFP4Recipe(chunk_blocks=getattr(args, "nvfp4_sim_chunk_blocks", 8))
    return install(model, recipe, include_output_layer=getattr(args, "nvfp4_sim_include_output_layer", True))
