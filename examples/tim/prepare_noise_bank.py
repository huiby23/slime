#!/usr/bin/env python3
"""CPU-only immutable eta=Z*initial_weight bank for fixed additive sampler noise.

Generate one relative perturbation from the initial local weights. Store its
unscaled FP32 base so a runtime sigma changes only its fixed amplitude,
without using updated weights or changing any global RNG stream.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

ALIASES = {"lm_head.weight": "model.embed_tokens.weight"}


def expected_names(config):
    layers = config.get("num_hidden_layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("Expected a positive dense Qwen3 layer count")
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    leaves = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    names.update("model.layers.{}.{}".format(layer, leaf) for layer in range(layers) for leaf in leaves)
    return names


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_tensor(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def tensor_seed(seed, canonical_name):
    value = hashlib.sha256((str(seed) + ":" + canonical_name).encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") & ((1 << 63) - 1)


def make_eta(initial, seed, canonical_name):
    if initial.device.type != "cpu" or not initial.is_floating_point():
        raise ValueError("Noise-bank input must be a floating-point CPU tensor")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(tensor_seed(seed, canonical_name))
    eta = torch.randn(initial.shape, dtype=torch.float32, device="cpu", generator=generator)
    eta.mul_(initial.to(dtype=torch.float32))
    return eta


def tensor_stats(tensor):
    value = tensor.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite source/noise tensor")
    std, mean = torch.std_mean(value, unbiased=False)
    minimum, maximum = torch.aminmax(value)
    mean, std = float(mean), float(std)
    return dict(
        mean=mean,
        std=std,
        rms=math.sqrt(mean * mean + std * std),
        min=float(minimum),
        max=float(maximum),
        zero_elements=value.numel() - int(torch.count_nonzero(value)),
    )


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_tensor(path, tensor):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("wb") as stream:
        # File-object serialization uses a stable archive prefix independent of
        # the temporary filename. The raw-tensor hash also records the contents.
        torch.save(tensor, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_identity(model):
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    for key, expected in {"model_type": "qwen3", "tie_word_embeddings": True}.items():
        if config.get(key) != expected:
            raise ValueError("Unexpected model config {}={!r}".format(key, config.get(key)))
    canonical_names = expected_names(config)
    files = sorted(model.glob("*.safetensors"))
    if not files:
        raise ValueError("No safetensors weights in " + str(model))
    return dict(
        model_path=str(model),
        config_sha256=sha_file(config_path),
        expected_canonical_names=sorted(canonical_names),
        safetensors_files={p.name: dict(sha256=sha_file(p), bytes=p.stat().st_size) for p in files},
    )


def verify_bank(output, source, seed):
    expected = set(source["expected_canonical_names"])
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("passed") is not True
        or manifest.get("seed") != seed
        or manifest.get("canonical_aliases") != ALIASES
        or manifest.get("raw_tensor_count") not in (len(expected), len(expected) + 1)
        or manifest.get("canonical_tensor_count") != len(expected)
        or manifest.get("source_alias_equalityverified") is not True
    ):
        raise ValueError("Existing bank has an invalid/incompatible manifest")
    for key in ("config_sha256", "safetensors_files"):
        if manifest["source"].get(key) != source.get(key):
            raise ValueError("Existing bank was prepared from a different source: " + key)
    entries = manifest["tensors"]
    if set(entries) != expected or any(name in entries for name in ALIASES):
        raise ValueError("Existing bank canonical tensor coverage is invalid")
    elementcount = 0
    for name, entry in sorted(entries.items()):
        path = output / entry["file"]
        path.resolve(strict=True).relative_to(output.resolve())
        if sha_file(path) != entry["sha256"]:
            raise ValueError("Corrupted tensor file: " + name)
        eta = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(eta, torch.Tensor)
            or eta.device.type != "cpu"
            or eta.dtype != torch.float32
            or list(eta.shape) != entry["shape"]
            or eta.numel() != entry["numel"]
            or not bool(torch.isfinite(eta).all())
            or sha_tensor(eta) != entry["eta_tensor_sha256"]
            or entry["tensor_seed"] != tensor_seed(seed, name)
        ):
            raise ValueError("Invalid tensor evidence: " + name)
        elementcount += eta.numel()
        del eta
    if elementcount != manifest["elementcount"]:
        raise ValueError("Existing bank element count is invalid")
    return manifest


def prepare(model, output, seed):
    if torch.cuda.is_initialized():
        raise RuntimeError("Noise-bank preparation must not initialize CUDA")
    random_state = torch.get_rng_state().clone()
    source = source_identity(model)
    expected = set(source["expected_canonical_names"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "manifest.json").exists():
            manifest = verify_bank(output, source, seed)
            reused = True
        else:
            if output.exists():
                # Never trust individual leftovers or discard a partial bank.
                # An empty pre-created destination is safe to remove atomically.
                output.rmdir()
            pending = output.parent / ("." + output.name + ".preparing-" + uuid.uuid4().hex)
            (pending / "tensors").mkdir(parents=True, exist_ok=False)
            with ExitStack() as stack:
                handles = {
                    name: stack.enter_context(safe_open(str(model / name), framework="pt", device="cpu"))
                    for name in source["safetensors_files"]
                }
                locations = {}
                for filename, handle in handles.items():
                    for name in handle.keys():
                        if name in locations:
                            raise ValueError("Duplicate source tensor: " + name)
                        locations[name] = filename
                if set(locations) - set(ALIASES) != expected:
                    raise ValueError("Source parameter names differ from expected tied dense Qwen3 coverage")
                for alias, canonical in ALIASES.items():
                    if alias not in locations:
                        # HF may serialize a tied tensor only under its canonical name.
                        continue
                    left = handles[locations[alias]].get_tensor(alias)
                    right = handles[locations[canonical]].get_tensor(canonical)
                    if left.dtype != right.dtype or not torch.equal(left, right):
                        raise ValueError("Tied source alias is unequal: " + alias)
                    del left, right
                entries = {}
                for index, name in enumerate(sorted(set(locations) - set(ALIASES))):
                    initial = handles[locations[name]].get_tensor(name).contiguous()
                    eta = make_eta(initial, seed, name)
                    relative = "tensors/" + hashlib.sha256(name.encode()).hexdigest() + ".pt"
                    path = pending / relative
                    source_stats, eta_stats = tensor_stats(initial), tensor_stats(eta)
                    save_tensor(path, eta)
                    entries[name] = dict(
                        file=relative,
                        sha256=sha_file(path),
                        eta_tensor_sha256=sha_tensor(eta),
                        source_sha256=sha_tensor(initial),
                        tensor_seed=tensor_seed(seed, name),
                        shape=list(initial.shape),
                        dtype=str(eta.dtype),
                        source_dtype=str(initial.dtype),
                        numel=initial.numel(),
                        source_stats=source_stats,
                        eta_stats=eta_stats,
                    )
                    print(
                        json.dumps(
                            dict(
                                event="tensor_prepared",
                                index=index + 1,
                                total=len(expected),
                                name=name,
                                numel=initial.numel(),
                            )
                        ),
                        flush=True,
                    )
                    del eta, initial
            if set(entries) != expected:
                raise ValueError("Unexpected canonical coverage")
            if source_identity(model) != source:
                raise ValueError("Source weights/config changed during preparation")
            if not torch.equal(random_state, torch.get_rng_state()) or torch.cuda.is_initialized():
                raise RuntimeError("Preparation changed global RNG or initialized CUDA")
            manifest = dict(
                schema_version=1,
                passed=True,
                seed=seed,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                source=source,
                raw_tensor_count=len(locations),
                canonical_tensor_count=len(expected),
                elementcount=sum(x["numel"] for x in entries.values()),
                source_alias_equalityverified=True,
                canonical_aliases=ALIASES,
                formula="eta_i = Z_i * W_initial_i; delta_i = sigma * eta_i; sampler_t = trainer_t + delta_i",
                distribution="Independent standard normal elements, separate CPU generator per canonical name",
                tensor_seed_derivation='SHA256(str(seed)+":"+canonical_name), first8bytes bigendian masked to63bits',
                global_cpu_rng_unchanged=True,
                cuda_initialized=False,
                scale_with_current_weights=False,
                accumulate_across_syncs=False,
                protocol_note="One name-keyed relative Gaussian offset from initial weights; sigma is applied once by each runtime bank.",
                tensors=entries,
            )
            atomic_json(pending / "manifest.json", manifest)
            # Publish only after every tensor and the complete manifest exists.
            pending.rename(output)
            reused = False
    if not torch.equal(random_state, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise RuntimeError("Preparation/verification changed global RNG or initialized CUDA")
    return dict(
        passed=True,
        reused=reused,
        output=str(output),
        manifest_sha256=sha_file(output / "manifest.json"),
        seed=seed,
        canonical_tensor_count=manifest["canonical_tensor_count"],
        elementcount=manifest["elementcount"],
        global_cpu_rng_unchanged=True,
        cuda_initialized=False,
        gpu_launches=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model", dest="model", type=Path, required=True)
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, required=True)
    parser.add_argument(
        "--sigma", type=float, default=0.05, help="Runtime scale; the immutable bank stores unscaled eta"
    )
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be a nonnegative63-bit integer")
    if not math.isfinite(args.sigma) or not 0 <= args.sigma <= 1:
        parser.error("--sigma must be finite and between 0 and 1")
    result = prepare(args.model.resolve(strict=True), args.output.resolve(), args.seed)
    result["runtime_scale"] = args.sigma
    result["runtime_manifest"] = str(args.output.resolve() / "manifest.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
