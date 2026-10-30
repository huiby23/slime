"""Download only the pinned official BF16 initialization, with an audit record."""

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def publish_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    status = {"model": args.model, "source": args.source, "destination": str(args.destination), "completed": False}
    publish_status(args.report, status)
    try:
        metadata_path = args.report.parent / ("model-info-" + args.source + ".json")
        if args.source == "modelscope":
            url = f"https://modelscope.cn/api/v1/models/{args.model}/repo/files?Revision=master&Recursive=true"
        else:
            url = "https://huggingface.co/api/models/" + args.model + "?blobs=true"
        if metadata_path.exists():
            raw = metadata_path.read_bytes()
        else:
            raw = subprocess.check_output(["curl", "-fLsS", "--connect-timeout", "10", "--max-time", "60", url])
            metadata_path.write_bytes(raw)
        info = json.loads(raw)
        if args.source == "modelscope":
            if not info.get("Success"):
                raise RuntimeError("ModelScope metadata request did not succeed")
            files = info["Data"]["Files"]
            info = {
                "sha": hashlib.sha256(raw).hexdigest(),
                "siblings": [
                    {
                        "rfilename": f["Path"],
                        "size": f["Size"],
                        "lfs": {"sha256": f["Sha256"]},
                        "revision": f["Revision"],
                    }
                    for f in files
                    if f["Type"] == "blob"
                ],
            }
            # Existing HF metadata may be reused only when it is byte-identical
            # to the pinned official mirror artifacts.
            for f in info["siblings"]:
                existing = args.destination / f["rfilename"]
                if existing.is_file() and existing.suffix == ".json":
                    if hashlib.sha256(existing.read_bytes()).hexdigest() != f["lfs"]["sha256"]:
                        raise RuntimeError(f"Existing initialization metadata differs: {existing.name}")
        status["revision"] = info["sha"]
        status["started_at"] = datetime.now(timezone.utc).isoformat()
        publish_status(args.report, status)
        args.destination.mkdir(parents=True, exist_ok=True)
        patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
        selected = [
            f for f in info["siblings"] if any(fnmatch.fnmatch(f["rfilename"], pattern) for pattern in patterns)
        ]

        def download(file):
            name = file["rfilename"]
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe model artifact path")
            destination = args.destination / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            expected = file.get("size")
            if not destination.exists() or (expected is not None and destination.stat().st_size != expected):
                if args.source == "modelscope":
                    url = f'https://modelscope.cn/api/v1/models/{args.model}/repo?Revision={file["revision"]}&FilePath={quote(name)}'
                else:
                    url = f'https://huggingface.co/{args.model}/resolve/{info["sha"]}/{quote(name)}?download=true'
                subprocess.run(
                    [
                        "curl",
                        "-fLsS",
                        "--retry",
                        "3",
                        "--connect-timeout",
                        "10",
                        "--max-time",
                        "7200",
                        "--speed-limit",
                        "1024",
                        "--speed-time",
                        "60",
                        "--continue-at",
                        "-",
                        "--output",
                        str(partial),
                        url,
                    ],
                    check=True,
                )
                if expected is not None and partial.stat().st_size != expected:
                    raise RuntimeError(f"Incomplete model artifact: {name}")
                partial.replace(destination)
            digest = file.get("lfs", {}).get("sha256")
            if digest:
                hasher = hashlib.sha256()
                with destination.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024**2), b""):
                        hasher.update(block)
                actual = hasher.hexdigest()
                if actual != digest:
                    raise RuntimeError(f"Weight checksum mismatch: {name}")
            print(f"verified {name}", flush=True)

        # Pin and fetch metadata before large weights, so configuration preflight
        # can proceed while weights download. Curl also honors the host proxy.
        for file in selected:
            if not file["rfilename"].endswith(".safetensors"):
                download(file)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(download, [f for f in selected if f["rfilename"].endswith(".safetensors")]))
        config = json.loads((args.destination / "config.json").read_text())
        if config.get("model_type") not in ("qwen3", "qwen3_moe") or config.get("quantization_config"):
            raise ValueError("Expected original, unquantized Qwen3 dense or MoE initialization")
        index = json.loads((args.destination / "model.safetensors.index.json").read_text())
        missing = [name for name in set(index["weight_map"].values()) if not (args.destination / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing weight shards: {missing}")
        status.update(
            completed=True,
            weight_bytes=index["metadata"]["total_size"],
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        publish_status(args.report, status)


if __name__ == "__main__":
    main()
