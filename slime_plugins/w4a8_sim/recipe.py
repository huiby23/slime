"""Explicit W4A8 forward and backward without native FP4/FP8 instructions.

Fprop: E4M3 activations, NVFP4 weights. Dgrad: E5M2 gradients, NVFP4 W.T.
Wgrad: E5M2 gradients and E4M3 activations. Every product accumulates in FP32.
FP8 uses current per-row blocks of 64 and FP32 scales. Weights reuse the
reference NVFP4 E2M1 / E4M3 / FP32 16x16 quantizer. These are numerical operands,
stored unpacked, not a memory-saving or native Tensor Core implementation.
"""

from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    from slime_plugins.nvfp4_sim.recipe import quantize as nvfp4_quantize
except ModuleNotFoundError:
    from nvfp4_sim.recipe import quantize as nvfp4_quantize


@dataclass
class FP8Tensor:
    values: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int]
    block_size: int
    format: str

    def dequantize(self):
        rows, columns = self.shape
        padded = self.values.shape[1]
        return (self.values.float().reshape(rows, padded // self.block_size, self.block_size)
                * self.scales[..., None]).reshape(rows, padded)[:, :columns]


@torch.no_grad()
def fp8_quantize(tensor, *, format="e4m3", block_size=64):
    if tensor.ndim != 2 or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("FP8 simulation expects a floating point matrix")
    if format not in ("e4m3", "e5m2") or block_size < 1:
        raise ValueError("Unsupported FP8 format or block size")
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite FP8 input")
    mantissa, emin, maximum = (3, -6, 448.0) if format == "e4m3" else (2, -14, 57344.0)
    rows, columns = tensor.shape
    x = F.pad(tensor.float(), (0, (-columns) % block_size))
    # reshape with explicit dimensions also handles zero-token experts.
    blocks = x.reshape(rows, x.shape[1] // block_size, block_size)
    scales = blocks.abs().amax(-1) / maximum
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scaled = blocks / scales[..., None]
    mag = scaled.abs().clamp_max(maximum)
    exponent = torch.floor(torch.log2(mag.clamp_min(2.0 ** emin))).clamp_min(emin)
    step = torch.exp2(exponent - mantissa)
    payload = torch.copysign((torch.round(mag / step) * step).clamp_max(maximum), scaled)
    return FP8Tensor(payload.reshape_as(x).bfloat16(), scales, (rows, columns), block_size, format)


@dataclass
class W4A8Recipe:
    # 64 divides the 30B expert width both before and after rollout TP=4:
    # 768/4=192. Blocks of 128 would change activation scales after sharding.
    activation_block_size: int = 64
    counters: Counter = field(default_factory=Counter)
    weight_amax_reducer: object = field(default=None, repr=False)
    _amax_cache: dict = field(default_factory=dict, repr=False)

    def activation(self, x):
        self.counters["activation_e4m3"] += 1
        return fp8_quantize(x, block_size=self.activation_block_size).dequantize()

    def gradient(self, g):
        self.counters["gradient_e5m2"] += 1
        return fp8_quantize(g, format="e5m2", block_size=self.activation_block_size).dequantize()

    def weight(self, w):
        self.counters["weight_nvfp4"] += 1
        amax = None
        if self.weight_amax_reducer is not None:
            try:
                version = w._version
            except RuntimeError:
                version = None
            key = (w.data_ptr(), tuple(w.shape), tuple(w.stride()), version)
            if key not in self._amax_cache:
                self._amax_cache[key] = self.weight_amax_reducer(w.detach().abs().float().amax())
                self.counters["weight_amax_reductions"] += 1
            amax = self._amax_cache[key]
        return nvfp4_quantize(w, weight=True, reference_arithmetic=True, tensor_amax=amax).dequantize()

    def invalidate_weight_cache(self):
        self._amax_cache.clear()


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, recipe):
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("W4A8 uses BF16 master-facing parameters and inputs")
        if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
            raise ValueError("Invalid W4A8 linear dimensions")
        matrix = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(matrix, weight)
        ctx.recipe, ctx.input_shape, ctx.parameter = recipe, x.shape, weight
        ctx.bias_dtype = None if bias is None else bias.dtype
        if matrix.shape[0] == 0:
            recipe.counters["empty_expert"] += 1
            return x.new_empty((*x.shape[:-1], weight.shape[0]))
        result = recipe.activation(matrix) @ recipe.weight(weight).T
        if bias is not None:
            result.add_(bias.float())
        recipe.counters["fprop"] += 1
        return result.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        g = grad_output.reshape(-1, grad_output.shape[-1]).bfloat16()
        recipe = ctx.recipe
        dx = dw = db = None
        qg = recipe.gradient(g) if g.shape[0] else g.float()
        if ctx.needs_input_grad[0]:
            if g.shape[0]:
                dx = (qg @ recipe.weight(weight.T.contiguous()).T).to(x.dtype)
                recipe.counters["dgrad"] += 1
            else:
                dx = torch.zeros_like(x)
            dx = dx.reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            dw = qg.T @ (recipe.activation(x) if x.shape[0] else x.float())
            if x.shape[0]:
                recipe.counters["wgrad"] += 1
            parameter = ctx.parameter
            if hasattr(parameter, "main_grad") and hasattr(parameter, "grad_added_to_main_grad"):
                parameter.main_grad.add_(dw)
                parameter.grad_added_to_main_grad = True
                dw = torch.zeros_like(parameter)
            else:
                dw = dw.to(weight.dtype)
        if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
            db = g.float().sum(0).to(ctx.bias_dtype)
        return dx, dw, db, None


def linear(x, weight, bias=None, *, recipe):
    return _Linear.apply(x, weight, bias, recipe)


def grouped_linear(x, weights, splits, *, recipe):
    """Expert-major inputs, preserving the original expert Parameters and grads."""
    if isinstance(splits, torch.Tensor):
        splits = splits.detach().cpu().tolist()
    splits = [int(n) for n in splits]
    if len(weights) != len(splits) or min(splits, default=0) < 0 or sum(splits) != x.shape[0]:
        raise ValueError("Expert splits do not match inputs and weights")
    recipe.counters["expert_nonempty_fprop"] += sum(n > 0 for n in splits)
    return torch.cat([linear(part, weight, recipe=recipe)
                      for part, weight in zip(x.split(splits), weights, strict=True)], dim=0)


@torch.no_grad()
def routed_experts(x, w13, w2, topk_ids, topk_weights, *, recipe):
    """Unpacked SwiGLU expert inference; router decisions remain external."""
    if x.ndim != 2 or topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != x.shape[0]:
        raise ValueError("Invalid routed expert inputs")
    if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("Expected unpacked [expert, output, input] matrices")
    if w13.shape[1] != 2 * w2.shape[2] or w13.shape[2] != x.shape[1] or w2.shape[1] != x.shape[1]:
        raise ValueError("Expert dimensions do not describe SwiGLU")
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in torch.unique(topk_ids).detach().cpu().tolist():
        if not 0 <= expert < w13.shape[0]:
            raise ValueError("Only rollout EP=1 with valid expert ids is supported")
        tokens, slots = torch.where(topk_ids == expert)
        recipe.counters["expert_nonempty_fprop"] += 1
        gate_up = linear(x[tokens], w13[expert], recipe=recipe)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        output = linear(activated, w2[expert], recipe=recipe)
        result.index_add_(0, tokens, output.float() * topk_weights[tokens, slots, None].float())
    return result.to(x.dtype)
