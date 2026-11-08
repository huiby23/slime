"""Candidate fused forward C4V8 integration; never installs itself globally.

All quantization boundaries and experiment-defined tree reductions remain.
Backward replays the same STE arithmetic, with fused pointwise QDQ; it is
NOT low-bit backward. This module is not production validated merely because
its component quantizers passed tests. Its launch policy/source gate is external.
"""
import math
import torch
import device_attention as reference

AttentionConfig = reference.AttentionConfig


def _apis():
    from forward_api import qdq_forward, pre_p_forward, post_pv_forward
    from nvfp4_triton import nvfp4_qdq
    return qdq_forward, pre_p_forward, post_pv_forward, nvfp4_qdq


def _normalize_api():
    from forward_api import normalize_forward
    return normalize_forward


class _MX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, masked):
        mask = torch.isneginf(x) if masked else torch.zeros_like(x, dtype=torch.bool)
        ctx.save_for_backward(mask)
        result = _apis()[0](x.float().contiguous(), allow_mask=masked, check=True)[0]
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad.masked_fill(mask, 0), None


def _mx(x, masked=False):
    return _MX.apply(x, masked)


def _tree(x, operation):
    x = _mx(x, operation == 'max')
    identity = -math.inf if operation == 'max' else 0.0
    while x.shape[-1] > 1:
        if x.shape[-1] % 2:
            x = torch.cat((x, torch.full_like(x[..., :1], identity)), -1)
        left, right = x[..., 0::2], x[..., 1::2]
        x = _mx(torch.maximum(left, right) if operation == 'max' else left + right,
                operation == 'max')
    return x


def _pre_reference(scores, maximum, denominator):
    scores = _mx(scores, True)
    new_m = _mx(torch.maximum(maximum, _tree(scores, 'max')), True)
    md, sd = maximum - new_m, scores - new_m
    if not bool((torch.isfinite(md) | torch.isneginf(maximum)).all()):
        raise FloatingPointError('Ordinary maximum subtraction overflow during STE replay')
    if not bool((torch.isfinite(sd) | torch.isneginf(scores)).all()):
        raise FloatingPointError('Ordinary score subtraction overflow during STE replay')
    alpha = _mx(torch.exp(_mx(md, True)))
    p = _mx(torch.exp(_mx(sd, True)))
    new_l = _mx(_mx(denominator * alpha) + _tree(p, 'sum'))
    if not bool((torch.isfinite(new_l) & (new_l > 0)).all()):
        raise FloatingPointError('Invalid denominator during STE replay')
    return new_m, new_l, alpha, p


def _post_reference(old, contribution, alpha):
    return (_mx(_mx(old * alpha) + _mx(contribution)),)


def _replay(ctx, function, gradients):
    needs = ctx.needs_input_grad
    with torch.enable_grad():
        inputs = [x.detach().requires_grad_(needed) for x, needed in zip(ctx.saved_tensors, needs)]
        outputs = function(*inputs)
        selected = [(out, grad) for out, grad in zip(outputs, gradients)
                    if out.requires_grad and grad is not None]
        active = [x for x, needed in zip(inputs, needs) if needed]
        if not selected:
            return tuple(torch.zeros_like(x) if needed else None for x, needed in zip(inputs, needs))
        result = torch.autograd.grad([x[0] for x in selected], active,
                                     [x[1] for x in selected], allow_unused=False)
        iterator = iter(result)
        return tuple(next(iterator) if needed else None for needed in needs)


class _Pre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, maximum, denominator):
        ctx.save_for_backward(scores, maximum, denominator)
        outputs = _apis()[1](scores.contiguous(), maximum.contiguous(), denominator.contiguous(), check=False)
        ctx.mark_non_differentiable(outputs[-1])
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gm, gl, ga, gp, unused_flags):
        return _replay(ctx, _pre_reference, (gm, gl, ga, gp))


class _Post(torch.autograd.Function):
    @staticmethod
    def forward(ctx, old, contribution, alpha):
        ctx.save_for_backward(old, alpha)
        outputs = _apis()[2](old.contiguous(), contribution.contiguous(), alpha.contiguous(), check=False)
        ctx.mark_non_differentiable(outputs[-1])
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient, unused_flags):
        old, alpha = ctx.saved_tensors
        needs = ctx.needs_input_grad
        # Identity STE makes this Jacobian independent of quantized values.
        # Preserve the original FP32 surrogate; this is not low-bit backward.
        if gradient is None:
            return tuple(torch.zeros_like(alpha if i == 2 else old) if needed else None
                         for i, needed in enumerate(needs))
        return (gradient * alpha if needs[0] else None,
                gradient if needs[1] else None,
                (gradient * old).sum_to_size(alpha.shape) if needs[2] else None)


class _Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, numerator, denominator):
        ctx.save_for_backward(numerator, denominator)
        outputs = _normalize_api()(numerator.contiguous(), denominator.contiguous(), check=False)
        ctx.mark_non_differentiable(outputs[-1])
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient, unused_flags):
        numerator, denominator = ctx.saved_tensors
        # Identity STE through the final MXFP8 QDQ. Gradients of division stay
        # high precision; no derivative of rounding/scales is introduced.
        gn = gradient / denominator if ctx.needs_input_grad[0] else None
        gd = (-(gradient * numerator) / denominator.square()).sum_to_size(denominator.shape) if ctx.needs_input_grad[1] else None
        return gn, gd


def attention(q, k, v, config=AttentionConfig(query_block=64, key_block=128), *, query_offset=0):
    if q.device.type == 'cpu':
        return reference.attention(q, k, v, config, query_offset=query_offset)
    if q.device.type not in ('npu', 'privateuseone') or any(x.device != q.device for x in (k, v)):
        raise ValueError('Matching NPU tensors required')
    if torch.npu.matmul.allow_hf32:
        raise RuntimeError('C4V8 FP32 reference accumulation requires allow_hf32=False')
    if config.vector != 'mxfp8' or not config.quantize_operands or not config.causal or config.reduction != 'mxfp8_tree_reduction':
        raise ValueError('Only the explicitly complete forward C4V8 baseline is implemented')
    if any(x.ndim != 4 or x.dtype != q.dtype or not x.is_floating_point() for x in (q, k, v)):
        raise ValueError('Expected matching floating BHND Q/K/V')
    b, h, n, d = q.shape
    nk = k.shape[2]
    if (min(q.shape) < 1 or k.shape != v.shape or k.shape[0] != b or k.shape[-1] != d
            or k.shape[1] < 1 or h % k.shape[1] or not isinstance(query_offset, int)
            or query_offset < 0 or query_offset+n > nk):
        raise ValueError('Invalid causal GQA inputs')
    # The standalone staged decode preserves each logical 128-key boundary.
    # Route only final-position inference; training/other offsets retain the
    # differentiable path below. Its checked API rejects invalid arithmetic.
    if (n == 1 and d == 128 and config.key_block == 128 and query_offset == nk - 1
            and q.dtype in (torch.float32, torch.bfloat16)
            and not any(x.requires_grad for x in (q, k, v))):
        from staged_decode import staged_decode_forward
        return staged_decode_forward(q, k, v, check=True)[0]
    if any(not bool(torch.isfinite(x).all()) for x in (q, k, v)):
        raise FloatingPointError('Nonfinite attention input')
    cube = _apis()[3]
    qq, kk, vv = (cube(x.float()) for x in (q, k, v))
    repeats = h // k.shape[1]
    if repeats != 1:
        kk, vv = kk.repeat_interleave(repeats, 1), vv.repeat_interleave(repeats, 1)
    result, flags = [], []
    for qs in range(0, n, config.query_block):
        qe = min(n, qs+config.query_block)
        qi = qq[:, :, qs:qe]
        state = (b, h, qe-qs, 1)
        maximum = torch.full(state, -math.inf, dtype=torch.float32, device=q.device)
        denominator = torch.zeros(state, dtype=torch.float32, device=q.device)
        accumulator = torch.zeros((b, h, qe-qs, d), dtype=torch.float32, device=q.device)
        stop = query_offset+qe
        qpos = torch.arange(query_offset+qs, query_offset+qe, device=q.device)[:, None]
        for ks in range(0, stop, config.key_block):
            ke = min(stop, ks+config.key_block)
            scores = qi @ kk[:, :, ks:ke].transpose(-1, -2) / math.sqrt(d)
            kpos = torch.arange(ks, ke, device=q.device)[None, :]
            mask = kpos > qpos
            # Capture ordinary nonfinite GEMM results before structural masking.
            flags.append((~torch.isfinite(scores.masked_fill(mask, 0))).any().reshape(1).to(torch.int32))
            scores = scores.masked_fill(mask, -math.inf)
            maximum, denominator, alpha, probabilities, f = _Pre.apply(scores, maximum, denominator)
            flags.append((f != 0).any().reshape(1).to(torch.int32))
            contribution = cube(probabilities) @ vv[:, :, ks:ke]
            accumulator, f = _Post.apply(accumulator, contribution, alpha)
            flags.append((f != 0).any().reshape(1).to(torch.int32))
        normalized, f = _Normalize.apply(accumulator, denominator)
        flags.append((f != 0).any().reshape(1).to(torch.int32))
        result.append(normalized.to(q.dtype))
    if bool((torch.cat(flags) != 0).any()):
        raise FloatingPointError('Fused attention encountered invalid data/arithmetic; no fallback')
    output = torch.cat(result, 2)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError('Nonfinite fused attention result')
    return output


def metadata():
    return dict(experimental=True, production_ready=False,
                scope='FA NVFP4/MXFP8 complete forward boundary QDQ; BF16 backbone',
                backward='High-precision identity-STE with query-tile replay; not quantized backward',
                decode_execution='Checked staged_decode for final-position inference at D128/K128; generic path for training/other offsets; whole_decode is not routed',
                errors='Per-stage device flags collected and checked before accepting any attention output',
                hardware='Ascend Triton numerical simulation; not native NVFP4/MXFP8 hardware')
