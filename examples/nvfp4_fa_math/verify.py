"""Verify 100 native updates, 20 recovery points, 21 evaluations and the final HF export."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from slime_plugins.nvfp4_fa.records import checkpoint, grad_norm, write_json


def owners(run):
    """A child inherits only the successfully restored prefix of its parent."""
    lineage = []
    end = 100
    seen = set()
    while run is not None:
        if run in seen:
            raise ValueError("Run lineage contains a cycle")
        seen.add(run)
        cfg = json.loads((run / "run.json").read_text())
        start = cfg["completed_on_entry"]
        lineage.append((start, end, run))
        end, run = start, (Path(cfg["parent"]) if cfg["parent"] else None)
    if end != 0:
        raise ValueError("Lineage does not start at step zero")
    return list(reversed(lineage))


def verify(run):
    import torch
    from safetensors import safe_open

    if json.loads((run / "exit.json").read_text())["returncode"] != 0:
        raise ValueError("Last driver did not exit successfully")
    chain = owners(run)
    updates, saved, curve = {}, {}, {}
    for start, end, path in chain:
        for step in range(start + 1, end + 1):
            updates[step] = grad_norm(path, step)
            if not step % 5:
                marker = json.loads((path / "saved" / f"step_{step:03d}.json").read_text())
                saved[step] = checkpoint(path, step)
                if marker["iteration"] != saved[step]["iteration"]:
                    raise ValueError(f"Wrong checkpoint receipt for step {step}")
        for step in range(start, end + 1, 5):
            file = path / "evaluation" / f"step_{step:03d}.json"
            if file.exists():
                record = json.loads(file.read_text())
                if record["step"] != step or record["questions"] != 500 or len(record["answers"]) != 500:
                    raise ValueError(f"Invalid MATH-500 record at step {step}")
                curve[step] = {k: record[k] for k in ("accuracy", "truncation_rate")}
    if (
        set(updates) != set(range(1, 101))
        or set(saved) != set(range(5, 101, 5))
        or set(curve) != set(range(0, 101, 5))
    ):
        raise ValueError("Incomplete update, checkpoint, or validation trajectory")
    hf = Path(saved[100]["hf"])
    base = Path(json.loads((run / "run.json").read_text())["model"])

    def index(path):
        return json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    initial, final = index(base), index(hf)
    if initial.keys() != final.keys():
        raise ValueError("Final HF index differs from the original parameter set")
    changed = []
    hashes = {}
    for shard in sorted(set(final.values())):
        with (hf / shard).open("rb") as stream:
            hashes[shard] = hashlib.file_digest(stream, "sha256").hexdigest()
        with safe_open(hf / shard, framework="pt") as output:
            for name in output.keys():
                tensor = output.get_slice(name)
                with safe_open(base / initial[name], framework="pt") as input_file:
                    original = input_file.get_slice(name)
                    if original.get_shape() != tensor.get_shape() or original.get_dtype() != tensor.get_dtype():
                        raise ValueError(f"Changed parameter layout: {name}")
                    width = max(1, 4194304 // max(1, math.prod(tensor.get_shape()[1:])))
                    for i in range(0, tensor.get_shape()[0], width):
                        block = tensor[i : i + width]
                        if not bool(block.isfinite().all()):
                            raise ValueError(f"Nonfinite parameter: {name}")
                        if i == 0 and not torch.equal(block[:1], original[:1]):
                            changed.append(name)
    if not changed:
        raise ValueError("No changed weights found relative to the original model")
    report = dict(
        updates=100,
        grad_norms=updates,
        checkpoints=20,
        curve=curve,
        final_hf=str(hf),
        tensors=len(final),
        sampled_changed_tensors=changed,
        sha256=hashes,
    )
    write_json(run / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    result = verify(parser.parse_args().run.resolve())
    print(result["final_hf"], result["curve"][100])
