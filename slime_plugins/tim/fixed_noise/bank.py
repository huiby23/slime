"""Load a frozen CPU noise bank without touching any global RNG stream."""

import hashlib
import json
from pathlib import Path

import torch


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sample_stats(tensor, maximum=4096):
    x = tensor.detach().reshape(-1)[:maximum].float()
    if not x.numel():
        return dict(sampled_elements=0, finite=True)
    return dict(
        sampled_elements=x.numel(),
        finite=bool(torch.isfinite(x).all()),
        rms=float(x.square().mean().sqrt()),
        max_abs=float(x.abs().max()),
        nonzero_elements=int(torch.count_nonzero(x)),
    )


class FixedNoiseBank:
    def __init__(self, manifest_path, scale):
        self.path = Path(manifest_path)
        self.manifest_sha256 = file_sha(self.path)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("schema_version") != 1 or self.manifest.get("passed") is not True:
            raise ValueError("Noise bank manifest must be a completed schema-version 1 bank")
        if self.manifest.get("source_alias_equalityverified") is not True:
            raise ValueError("Noise bank has not verified tied source aliases")
        seed = self.manifest.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("Noise bank seed must be a nonnegative 63-bit integer")
        self.scale = float(scale)
        if not 0 <= self.scale <= 1:
            raise ValueError("Noise scale must be finite and between 0 and 1")
        self.entries = self.manifest["tensors"]
        self.aliases = self.manifest.get("canonical_aliases", {})
        self._deltas = {}
        self.records = {}

    def canonical(self, name):
        if name.startswith(("layers.", "embed_tokens.", "norm.")):
            name = "model." + name
        return self.aliases.get(name, name)

    def delta(self, name, shape):
        name = self.canonical(name)
        entry = self.entries[name]
        if tuple(entry["shape"]) != tuple(shape):
            raise ValueError("Noise shape mismatch at " + name)
        reused = name in self._deltas
        if not reused:
            path = (self.path.parent / entry["file"]).resolve()
            path.relative_to(self.path.parent.resolve())
            if file_sha(path) != entry["sha256"]:
                raise ValueError("Noise bank file changed: " + name)
            eta = torch.load(path, map_location="cpu", weights_only=True)
            if eta.dtype != torch.float32 or tuple(eta.shape) != tuple(shape):
                raise ValueError("Noise bank tensor type/shape mismatch: " + name)
            if tensor_sha(eta) != entry["eta_tensor_sha256"]:
                raise ValueError("Noise bank payload fingerprint mismatch: " + name)
            if not bool(torch.isfinite(eta).all()):
                raise ValueError("Nonfinite noise bank tensor: " + name)
            # This CPU tensor is private to the worker. Scale exactly once;
            # no sampling or current-weight-dependent amplitude is used.
            delta = eta.mul_(self.scale)
            self._deltas[name] = delta
            self.records[name] = dict(
                bank_sha256=entry["sha256"],
                delta_fingerprint=tensor_sha(delta),
                shape=list(shape),
                numel=delta.numel(),
                intended_delta_sample_stats=sample_stats(delta),
                loads=0,
            )
        self.records[name]["loads"] += 1
        return name, self._deltas[name], reused

    def transform(self, name, fresh):
        if fresh.dtype != torch.bfloat16:
            raise ValueError("Fixed-noise experiment requires fresh BF16 payloads: " + name)
        canonical, delta, reused = self.delta(name, fresh.shape)
        # The source may be a view of the trainer's CUDA IPC buffer. It must
        # NEVER be modified, including through .data or a detached alias.
        noisy = (fresh.float() + delta.to(device=fresh.device)).to(dtype=fresh.dtype)
        count = min(fresh.numel(), 4096)
        realized = noisy.reshape(-1)[:count].float() - fresh.reshape(-1)[:count].float()
        self.records[canonical]["realized_bf16_delta_sample_stats"] = sample_stats(realized)
        return canonical, noisy, reused
