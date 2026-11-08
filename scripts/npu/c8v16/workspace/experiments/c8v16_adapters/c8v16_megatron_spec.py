"""Slime --spec entry point for the BF16 Qwen MoE model."""

import copy
import os

import torch

from c8v16_profile import load_module, load_profile, record_event

# Reuse validated SBHD/THD validation, padding and query-tile recomputation.
for _name in ("bridge", "layouts"):
    load_module(_name)
_shared = load_module("megatron_spec")


class C8V16CoreAttention(_shared.C4V8CoreAttention):
    """The base module implements layout conversion and calls the injected kernel."""

    def forward(self, query, key, value, *args, **kwargs):
        if any(x.dtype != torch.bfloat16 for x in (query, key, value)):
            raise ValueError("W16A16 training attention requires BF16 Q/K/V")
        return super().forward(query, key, value, *args, **kwargs)


def replace_attention(block_spec, config, core, module_spec_type):
    spec = copy.deepcopy(block_spec)
    if not getattr(spec, "layer_specs", None):
        raise ValueError("Expected at least one local Megatron decoder layer")
    for layer in spec.layer_specs:
        submodules = layer.submodules.self_attention.submodules
        if not hasattr(submodules, "core_attention"):
            raise ValueError("Unsupported Megatron attention specification")
        submodules.core_attention = module_spec_type(
            module=C8V16CoreAttention, params={"kernel": core.attention, "precision_config": config})
    return spec


_trainer_identity = None


def get_spec(args, config, vp_stage=None):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.transformer.spec_utils import ModuleSpec

    if (not getattr(args, "num_experts", None) or not getattr(args, "bf16", False)
            or getattr(args, "fp4", None) or getattr(args, "fp8", None)):
        raise ValueError("W16A16C8V16 requires the BF16 MoE backbone")
    profile = load_profile()
    control = _shared.configure_npu_matmul()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    identity = (os.getpid(), profile.implementation_sha256, profile.policy_sha256, rank)
    global _trainer_identity
    if _trainer_identity is None:
        record_event(profile, "HBY_C8V16_TRAIN_AUDIT_DIR", "trainer-policy",
                     rank=rank, core_module=profile.core.__name__, fp32_matmul=control)
        _trainer_identity = identity
    elif _trainer_identity != identity:
        raise RuntimeError("Trainer precision or rank changed in process")
    options = {"use_transformer_engine": args.transformer_impl == "transformer_engine"}
    if vp_stage is not None:
        options["vp_stage"] = vp_stage
    return replace_attention(get_gpt_decoder_block_spec(config, **options), profile.config,
                             profile.core, ModuleSpec)
