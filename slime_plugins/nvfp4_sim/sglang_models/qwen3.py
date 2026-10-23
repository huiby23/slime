"""Preserve Qwen3's loader/update paths while simulating NVFP4 linear GEMMs."""

import logging

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod, UnquantizedLinearMethod
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from ..recipe import NVFP4Recipe, gemm, quantize


class NVFP4SimLinearMethod(UnquantizedLinearMethod):
    def __init__(self, recipe):
        self.recipe = recipe

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16 or layer.weight.dtype != torch.bfloat16:
            raise ValueError("NVFP4 rollout simulation requires BF16 weights and inputs")
        # Recompute from original weights so in-place distributed updates cannot
        # leave a stale quantization cache. Activations are quantized every call.
        with torch.no_grad():
            qx = quantize(x.reshape(-1, x.shape[-1]))
            qw = quantize(layer.weight, weight=True)
            output = gemm(qx, qw, chunk_blocks=self.recipe.chunk_blocks)
            if bias is not None:
                output.add_(bias.float())
            self.recipe.record("fprop")
            return output.to(x.dtype).reshape(*x.shape[:-1], layer.weight.shape[0])


class NVFP4SimEmbeddingLinearMethod(UnquantizedEmbeddingMethod):
    """Use BF16 embedding lookup and NVFP4 output projection on the same weight."""

    def __init__(self, recipe):
        self.recipe = recipe

    apply = NVFP4SimLinearMethod.apply


class NVFP4SimQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.pp_group.world_size != 1 or get_tensor_model_parallel_world_size() != 1:
            raise ValueError("The baseline NVFP4 rollout simulation requires TP=PP=1")
        if getattr(self, "quant_config", None) is not None:
            raise ValueError("NVFP4 simulation loads original BF16 weights; disable native quantization")
        recipe = NVFP4Recipe()
        names = []
        selected = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
        include_head = getattr(config, "nvfp4_sim_include_output_layer", True)
        # The tied lm_head aliases model.embed_tokens. Include that alias while
        # preserving the embedding() implementation on their shared module.
        for name, module in self.named_modules(remove_duplicate=False):
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in selected or (leaf == "lm_head" and not include_head):
                continue
            expected_method = (
                (UnquantizedLinearMethod, UnquantizedEmbeddingMethod) if leaf == "lm_head" else UnquantizedLinearMethod
            )
            if not isinstance(getattr(module, "quant_method", None), expected_method):
                raise TypeError(f"Expected an unquantized linear method at {name}")
            method_cls = (
                NVFP4SimEmbeddingLinearMethod
                if isinstance(module.quant_method, UnquantizedEmbeddingMethod)
                else NVFP4SimLinearMethod
            )
            module.quant_method = method_cls(recipe)
            names.append(name)
        expected = config.num_hidden_layers * 4 + int(include_head)
        if len(names) != expected:
            raise RuntimeError(f"NVFP4 rollout coverage mismatch: expected {expected}, got {names}")
        self._nvfp4_recipe = recipe
        self._nvfp4_modules = tuple(names)
        logging.getLogger(__name__).warning("NVFP4 rollout simulation installed on %d linears", len(names))


EntryClass = NVFP4SimQwen3ForCausalLM
