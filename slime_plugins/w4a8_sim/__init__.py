"""A100 numerical simulation of NVFP4 weights and FP8 activations.

This is a custom mixed-format W4A8 recipe, not NVIDIA's W4A4 training recipe.
"""

from .recipe import W4A8Recipe, fp8_quantize, linear

__all__ = ["W4A8Recipe", "fp8_quantize", "linear"]
