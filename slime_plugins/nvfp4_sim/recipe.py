"""NVFP4 numerical simulation, including FPROP, DGRAD and WGRAD.

No FP4/FP8 CUDA instructions are used. E2M1 payloads are held in BF16 (all
payload values are exactly representable); E4M3 scales are rounded in software.
GEMMs multiply 16-element payload blocks, apply block scales, accumulate in FP32,
and apply the tensor scales at the epilogue. This avoids an extra BF16 rounding
of dequantized operands before GEMM.

Reference: NVIDIA Transformer Engine 2.18 NVFP4BlockScaling, default settings.
Hardware packing, swizzles, RNG bit streams and accumulation order are not
emulated. This is a recipe simulation, not a bitwise hardware emulator.
"""

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

BLOCK = 16
# Fixed across layers, matching TE's get_wgrad_sign_vector().
RHT_SIGNS = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@lru_cache(maxsize=32)
def _constants(device):
    values = torch.tensor(FP4_VALUES, device=device, dtype=torch.float32)
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    for _ in range(4):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h = torch.tensor(RHT_SIGNS, device=device, dtype=torch.float32)[:, None] * h / 4
    return values, h


def round_e4m3(x):
    """Nonnegative finite E4M3FN scales, round-to-nearest, ties-to-even.

    Subnormal spacing is 2**-9; normal spacing is 2**(exponent-3).
    Keeping this in FP32 makes it usable on Ampere and CPU alike.
    """
    x = x.clamp(0, 448)
    exponent = torch.floor(torch.log2(x.clamp_min(2**-6))).clamp_min(-6)
    step = torch.exp2(exponent - 3)
    return (torch.round(x / step) * step).clamp_max(448)


def round_e2m1(x, stochastic=False, generator=None):
    """Saturating E2M1 conversion; deterministic ties use the even encoding."""
    values, _ = _constants(x.device)
    magnitude = x.abs().clamp_max(6)
    upper_index = torch.bucketize(magnitude.contiguous(), values).clamp(1, 7)
    lower_index = upper_index - 1
    lower, upper = values[lower_index], values[upper_index]
    fraction = (magnitude - lower) / (upper - lower)
    if stochastic:
        choose_upper = torch.rand(x.shape, device=x.device, dtype=torch.float32, generator=generator) < fraction
    else:
        choose_upper = (fraction > 0.5) | ((fraction == 0.5) & (upper_index.remainder(2) == 0))
    return torch.copysign(torch.where(choose_upper, upper, lower), x)


@dataclass
class NVFP4Tensor:
    values: torch.Tensor
    scales: torch.Tensor  # Expanded row scales, [padded_rows, padded_columns / 16].
    global_scale: torch.Tensor
    shape: tuple[int, int]
    weight: bool = False

    def dequantize(self):
        rows, columns = self.shape
        if self.values.numel() == 0:
            return self.values.float()[:rows, :columns]
        result = self.values.float().reshape(self.values.shape[0], -1, BLOCK)
        result = result * self.scales[..., None] * self.global_scale
        return result.reshape_as(self.values)[:rows, :columns]

    def transpose_weight(self):
        if not self.weight:
            raise ValueError("Only 16x16 weight tiles can be transposed without requantization")
        # Transpose 2D tiles, then expand the single scale to each of 16 rows.
        scales = self.scales[::BLOCK].T.repeat_interleave(BLOCK, 0).contiguous()
        return NVFP4Tensor(self.values.T.contiguous(), scales, self.global_scale, self.shape[::-1], True)


@torch.no_grad()
def quantize(tensor, *, weight=False, stochastic=False, generator=None, reference_arithmetic=True,
             tensor_amax=None):
    """Quantize a matrix with current tensor amax, 1x16 or 16x16 blocks."""
    if tensor.ndim != 2 or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("NVFP4 simulation expects a BF16/FP16/FP32 matrix")
    rows, columns = tensor.shape
    padded_rows = ((rows + 15) // 16) * 16 if weight else rows
    padded_columns = ((columns + 15) // 16) * 16
    x = F.pad(tensor.float(), (0, padded_columns - columns, 0, padded_rows - rows))
    if x.numel() == 0:
        scales = x.new_zeros((padded_rows, padded_columns // 16))
        return NVFP4Tensor(x.to(torch.bfloat16), scales, x.new_ones(()), (rows, columns), weight)
    blocks = x.reshape(padded_rows, -1, BLOCK)
    if weight:
        amax = blocks.reshape(padded_rows // BLOCK, BLOCK, -1, BLOCK).abs().amax((1, 3))
        amax = amax.repeat_interleave(BLOCK, 0)
    else:
        amax = blocks.abs().amax(-1)
    # Match TE's finite encode-scale clamping, including zero and tiny tensors.
    limit = torch.finfo(torch.float32).max
    global_amax = x.abs().amax() if tensor_amax is None else torch.as_tensor(tensor_amax, dtype=torch.float32, device=x.device)
    if reference_arithmetic:
        # Tensor.__rtruediv__ may lower a scalar/tensor division to reciprocal
        # followed by multiplication. TE's reference uses an explicit div;
        # the one-ULP difference can cross E4M3 rounding ties on real activations.
        global_encode = torch.div(x.new_tensor(2688.0), global_amax).clamp_max(limit)
    else:
        global_encode = (2688.0 / global_amax).clamp_max(limit)
    global_encode = torch.where(global_encode == 0, torch.ones_like(global_encode), global_encode)
    global_decode = torch.div(torch.ones_like(global_encode), global_encode) if reference_arithmetic else global_encode.reciprocal()
    reciprocal_six = torch.reciprocal(x.new_tensor(6.0)) if reference_arithmetic else 1.0 / 6.0
    scales = round_e4m3((amax * (global_encode * reciprocal_six)).clamp_max(limit))
    denominator = scales * global_decode
    encode = (torch.div(torch.ones_like(denominator), denominator) if reference_arithmetic else denominator.reciprocal()).clamp_max(limit)
    payload = round_e2m1(blocks * encode[..., None], stochastic, generator)
    return NVFP4Tensor(payload.reshape_as(x).to(torch.bfloat16), scales, global_decode, (rows, columns), weight)


@torch.no_grad()
def rht_transpose(tensor):
    """Transform the token reduction axis for WGRAD, from original BF16 data.

    Both G.T and X.T receive the SAME normalized orthogonal 16x16 transform.
    Zero-pad tokens before rotation, and retain padding through quantization.
    """
    if tensor.dtype != torch.bfloat16:
        raise ValueError("TE's RHT training recipe requires BF16 inputs and gradients")
    transposed = tensor.T.contiguous()
    padded = F.pad(transposed, (0, (-transposed.shape[-1]) % BLOCK))
    _, h = _constants(tensor.device)
    return (padded.reshape(-1, BLOCK) @ h.to(tensor.dtype)).reshape_as(padded)


@torch.no_grad()
def gemm(a, b, *, chunk_blocks=8):
    """A @ B.T with FP32 block-scaled accumulation and global-scale epilogue."""
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"GEMM reduction dimensions differ: {a.shape} and {b.shape}")
    if chunk_blocks < 1:
        raise ValueError("chunk_blocks must be positive")
    m, k = a.shape
    n = b.shape[0]
    result = torch.zeros((m, n), device=a.values.device, dtype=torch.float32)
    if m == 0 or n == 0 or k == 0:
        return result
    av = a.values[:m].reshape(m, -1, BLOCK).permute(1, 0, 2)
    bv = b.values[:n].reshape(n, -1, BLOCK).permute(1, 2, 0)
    # Bound the temporary output, especially for the vocabulary projection.
    chunk = min(chunk_blocks, max(1, (16 * 1024**2) // max(4 * m * n, 1)))
    for start in range(0, (k + 15) // BLOCK, chunk):
        end = start + chunk
        products = torch.bmm(av[start:end].float(), bv[start:end].float())
        products *= a.scales[:m, start:end].T[:, :, None] * b.scales[:n, start:end].T[:, None, :]
        result.add_(products.sum(0))
    return result * (a.global_scale * b.global_scale)


@dataclass
class NVFP4Recipe:
    """Default TE training recipe; toggles exist for controlled ablations/tests."""

    stochastic_rounding: bool = True
    hadamard: bool = True
    weight_2d: bool = True
    chunk_blocks: int = 8
    counters: Counter = field(default_factory=Counter)
    # Optional independent stream for frozen diagnostic replays. Normal training
    # keeps using PyTorch's default CUDA RNG, exactly as before.
    stochastic_generator: object = field(default=None, repr=False, compare=False)
    # False is an explicit compatibility arm for the already-completed baseline.
    reference_arithmetic: bool = True

    def record(self, path):
        self.counters[path] += 1

    def backward_operands(self, x, g):
        if self.hadamard:
            self.record("rht_x")
            self.record("rht_g")
            return rht_transpose(x), rht_transpose(g)
        return x.T.contiguous(), g.T.contiguous()


class _NVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, recipe):
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("The default NVFP4 training simulation requires BF16 parameters and inputs")
        matrix = x.reshape(-1, x.shape[-1])
        qw = quantize(weight, weight=recipe.weight_2d, reference_arithmetic=recipe.reference_arithmetic)
        qx = quantize(matrix, reference_arithmetic=recipe.reference_arithmetic)
        output = gemm(qx, qw, chunk_blocks=recipe.chunk_blocks)
        if bias is not None:
            output.add_(bias.float())
        recipe.record("fprop")
        ctx.recipe, ctx.input_shape = recipe, x.shape
        ctx.weight_parameter = weight
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.save_for_backward(matrix, weight)
        return output.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        recipe = ctx.recipe
        g = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.bfloat16)
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            qg = quantize(g, stochastic=recipe.stochastic_rounding, generator=recipe.stochastic_generator,
                          reference_arithmetic=recipe.reference_arithmetic)
            # Quantize the original high precision transpose, never a 1D QDQ result.
            qwt = quantize(weight.T.contiguous(), weight=recipe.weight_2d, reference_arithmetic=recipe.reference_arithmetic)
            dx = gemm(qg, qwt, chunk_blocks=recipe.chunk_blocks).to(x.dtype).reshape(ctx.input_shape)
            recipe.record("dgrad")
            if recipe.stochastic_rounding:
                recipe.record("sr_dgrad")
        if ctx.needs_input_grad[1]:
            xt, gt = recipe.backward_operands(x, g)
            qxt = quantize(xt, reference_arithmetic=recipe.reference_arithmetic)
            qgt = quantize(gt, stochastic=recipe.stochastic_rounding, generator=recipe.stochastic_generator,
                           reference_arithmetic=recipe.reference_arithmetic)
            dw = gemm(qgt, qxt, chunk_blocks=recipe.chunk_blocks)
            recipe.record("wgrad")
            if recipe.stochastic_rounding:
                recipe.record("sr_wgrad")
            # Preserve TE/Megatron's FP32 main_grad accumulation contract.
            parameter = ctx.weight_parameter
            if hasattr(parameter, "main_grad") and hasattr(parameter, "grad_added_to_main_grad"):
                parameter.main_grad.add_(dw)
                parameter.grad_added_to_main_grad = True
                dw = torch.zeros_like(parameter)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = g.float().sum(0).to(ctx.bias_dtype)
        return dx, dw, db, None


def nvfp4_linear(x, weight, bias=None, *, recipe):
    return _NVFP4Linear.apply(x, weight, bias, recipe)
