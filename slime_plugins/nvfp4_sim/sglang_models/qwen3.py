"""Preserve Qwen3's loader/update paths while simulating NVFP4 linear GEMMs."""

import logging

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod, UnquantizedLinearMethod
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from ..recipe import NVFP4Recipe, gemm, quantize


class NVFP4SimLinearMethod(UnquantizedLinearMethod):
    def __init__(self, recipe, cache_weights=False):
        self.recipe = recipe
        self.cache_weights = cache_weights
        self.invalidate_weight_cache()

    def invalidate_weight_cache(self):
        self._weight_key = self._weight_cache = None

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16 or layer.weight.dtype != torch.bfloat16:
            raise ValueError("NVFP4 rollout simulation requires BF16 weights and inputs")
        # Activations are quantized every call. Optional cached weight casts are
        # invalidated after stock loading and both baseline update APIs.
        with torch.no_grad():
            qx = quantize(x.reshape(-1, x.shape[-1]))
            try:
                version = layer.weight._version
            except RuntimeError:  # Inference tensors intentionally have no version counter.
                version = None
            key = (
                id(layer.weight),
                layer.weight.data_ptr(),
                version,
                layer.weight.device,
                layer.weight.dtype,
                tuple(layer.weight.shape),
            )
            if not self.cache_weights or key != self._weight_key:
                qw = quantize(layer.weight, weight=True)
                if self.cache_weights:
                    self._weight_key, self._weight_cache = key, qw
            else:
                qw = self._weight_cache
            output = gemm(qx, qw, chunk_blocks=self.recipe.chunk_blocks)
            if bias is not None:
                output.add_(bias.float())
            self.recipe.record("fprop")
            return output.to(x.dtype).reshape(*x.shape[:-1], layer.weight.shape[0])


class NVFP4SimEmbeddingLinearMethod(UnquantizedEmbeddingMethod):
    """Use BF16 embedding lookup and NVFP4 output projection on the same weight."""

    def __init__(self, recipe, cache_weights=False):
        NVFP4SimLinearMethod.__init__(self, recipe, cache_weights)

    apply = NVFP4SimLinearMethod.apply
    invalidate_weight_cache = NVFP4SimLinearMethod.invalidate_weight_cache


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
            module.quant_method = method_cls(recipe, cache_weights=getattr(config, "nvfp4_sim_weight_cache", False))
            names.append(name)
        expected = config.num_hidden_layers * 4 + int(include_head)
        if len(names) != expected:
            raise RuntimeError(f"NVFP4 rollout coverage mismatch: expected {expected}, got {names}")
        self._nvfp4_recipe = recipe
        self._nvfp4_modules = tuple(names)
        logging.getLogger(__name__).warning("NVFP4 rollout simulation installed on %d linears", len(names))

    def load_weights(self, *args, **kwargs):
        # Both the baseline's regular and flattened-bucket updates call this
        # method. Their loaders may write through .data without a version bump.
        try:
            return super().load_weights(*args, **kwargs)
        finally:
            for module in self.modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, (NVFP4SimLinearMethod, NVFP4SimEmbeddingLinearMethod)):
                    method.invalidate_weight_cache()


EntryClass = NVFP4SimQwen3ForCausalLM
