"""Shared SGLang linear method for dense and MoE W4A8 models."""

import torch
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

from .recipe import linear


class W4A8LinearMethod(UnquantizedLinearMethod):
    def __init__(self, recipe):
        self.recipe = recipe

    def apply(self, layer, x, bias=None):
        with torch.no_grad():
            return linear(x, layer.weight, bias, recipe=self.recipe)
