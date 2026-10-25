"""Export model tensors from native DCP checkpoints on CPU, without optimizer state."""

import argparse
import hashlib
import importlib.util
import io
import json
import pickle
import re
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file


def common_state(directory, metadata):
    if (directory / "common.pt").exists():
        return torch.load(directory / "common.pt", map_location="cpu", weights_only=False)
    key = next(k for k in metadata.state_dict_metadata if k.startswith("common_state/"))
    index = next(k for k in metadata.storage_data if k.fqn == key)
    storage = metadata.storage_data[index]
    with (directory / storage.relative_path).open("rb") as stream:
        stream.seek(storage.offset)
        value = torch.load(io.BytesIO(stream.read(storage.length)), map_location="cpu", weights_only=False)
        # MCore's ShardedObject stores a list of local object shards.
        return value[0] if isinstance(value, list) and len(value) == 1 else value


def export(source, origin, output):
    assert not torch.cuda.is_available(), "Diagnostic export must run without visible CUDA devices"
    assert not output.exists(), "Use a fresh export directory"
    reader = dcp.FileSystemReader(source)
    metadata = reader.read_metadata()
    common = common_state(source, metadata)
    args = common["args"]
    tensors = {
        key: torch.empty(value.size, dtype=value.properties.dtype)
        for key, value in metadata.state_dict_metadata.items()
        if isinstance(value, dcp.metadata.TensorStorageMetadata)
        and not key.startswith("optimizer")
        and "_state" not in key
    }
    dcp.state_dict_loader._load_state_dict(
        tensors, storage_reader=reader, planner=dcp.DefaultLoadPlanner(), no_dist=True
    )
    # Load the pinned mapper directly; no model or GPU initialization is needed.
    mapper_path = Path("/workspace/slime/slime/backends/megatron_utils/megatron_to_hf/qwen2.py")
    spec = importlib.util.spec_from_file_location("diagnostic_qwen_mapper", mapper_path)
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    converted = {}
    for key, tensor in tensors.items():
        if ".layers." in key and not re.search(r"\.layers\.\d+\.", key):
            assert tensor.shape[0] == args.num_layers
            parts = [(key.replace(".layers.", f".layers.{i}."), tensor[i]) for i in range(args.num_layers)]
        else:
            parts = [(key, tensor)]
        for name, value in parts:
            if name in ("embedding.word_embeddings.weight", "output_layer.weight"):
                value = value[: args.vocab_size]
            for hf_name, hf_value in mapper.convert_qwen2_to_hf(args, "module.module." + name, value):
                assert hf_name not in converted, hf_name
                converted[hf_name] = hf_value.contiguous()
    config = json.loads((origin / "config.json").read_text())
    expected = {}
    for path in origin.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as weights:
            expected.update({key: weights.get_slice(key).get_shape() for key in weights.keys()})
    if config.get("tie_word_embeddings"):
        if "lm_head.weight" in converted:
            assert torch.equal(converted["lm_head.weight"], converted["model.embed_tokens.weight"])
        if "lm_head.weight" in expected:
            # Serialize the trained tied weight twice when the origin uses both
            # names; never fill the head from the untrained HF initialization.
            converted["lm_head.weight"] = converted["model.embed_tokens.weight"].clone()
        else:
            converted.pop("lm_head.weight", None)
    assert set(converted) == set(expected), (set(expected) - set(converted), set(converted) - set(expected))
    assert all(list(value.shape) == expected[key] for key, value in converted.items())
    assert all(value.dtype == torch.bfloat16 and torch.isfinite(value).all() for value in converted.values())
    output.mkdir(parents=True)
    for path in origin.iterdir():
        if path.is_file() and not path.name.endswith((".safetensors", ".bin")) and "index.json" not in path.name:
            shutil.copy2(path, output / path.name)
    save_file(converted, output / "model.safetensors", metadata={"format": "pt"})
    fingerprint = hashlib.sha256()
    for name, tensor in sorted(converted.items()):
        fingerprint.update(name.encode())
        fingerprint.update(tensor.view(torch.uint8).numpy().tobytes())
    report = {
        "passed": True, "source": str(source), "iteration": common["iteration"],
        "completed_updates": common["iteration"] + 1, "model_tensors": len(converted),
        "parameters": sum(value.numel() for value in converted.values()) - (
            converted["lm_head.weight"].numel() if config.get("tie_word_embeddings") and "lm_head.weight" in converted else 0),
        "serialized_numel": sum(value.numel() for value in converted.values()),
        "parameter_sha256": fingerprint.hexdigest(), "tied_embeddings": config.get("tie_word_embeddings"),
        "strict_names_shapes_dtypes": True, "optimizer_exported": False,
    }
    (output / "export-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.input, args.origin, args.output)
