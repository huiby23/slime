"""Small, canonicalized tensors for frozen cross-engine diagnostics only."""

import hashlib
import re
import torch


def canonical_name(name):
    match = re.search(r"(?:decoder\.)?layers\.(\d+)\.(.+)", name)
    if not match:
        return "head" if name.endswith(("output_layer", "lm_head")) else None
    layer, rest = match.groups()
    if int(layer) not in (0, 13, 27):
        return None
    for suffix, label in [("linear_qkv", "qkv"), ("qkv_proj", "qkv"),
                          ("linear_proj", "o"), ("o_proj", "o"),
                          ("linear_fc1", "gate_up"), ("gate_up_proj", "gate_up"),
                          ("linear_fc2", "down"), ("down_proj", "down")]:
        if rest.endswith(suffix):
            return f"layer{layer}.{label}"
    return None


def canonical_qkv(tensor, weight=False):
    """Megatron's eight [Q,Q,K,V] groups -> SGLang's contiguous [Q,K,V]."""
    if weight:
        view = tensor.reshape(8, 4, 128, tensor.shape[-1])
        return torch.cat([part.reshape(-1, tensor.shape[-1]) for part in view.split([2, 1, 1], dim=1)], 0)
    view = tensor.reshape(-1, 8, 4, 128)
    return torch.cat([part.reshape(tensor.shape[0], -1) for part in view.split([2, 1, 1], dim=2)], -1)


def capture(name, x, weight, output, valid_rows, megatron=False):
    matrix = x.detach().reshape(-1, x.shape[-1])
    result = output.detach().reshape(-1, output.shape[-1])
    valid_rows = min(valid_rows, matrix.shape[0])
    indices = sorted(set(torch.linspace(0, valid_rows - 1, min(32, valid_rows)).round().long().tolist()))
    if megatron and name.endswith(".qkv"):
        result = canonical_qkv(result)
        weight = canonical_qkv(weight.detach(), weight=True)
    weight_bytes = weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    record = dict(rows=matrix.shape[0], valid_rows=valid_rows, indices=indices,
                  input=matrix[indices].cpu().clone(),
                  output=result[indices].cpu().clone() if name != "head" else None,
                  input_amax=float(matrix.float().abs().max()),
                  valid_input_amax=float(matrix[:valid_rows].float().abs().max()),
                  weight_sha256=hashlib.sha256(weight_bytes).hexdigest())
    if name.startswith("layer0."):
        record["reference_input"] = matrix[:min(32, valid_rows)].cpu().clone()
        record["reference_weight"] = weight.detach().cpu().clone()
    return record
