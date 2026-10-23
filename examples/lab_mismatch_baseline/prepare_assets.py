#!/usr/bin/env python3
"""Prepare revision-pinned Qwen3 weights and non-overlapping GSM8K splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import urllib.parse
import urllib.request

from runtime import data_root

GSM_REPO = "openai/grade-school-math"
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "slime-lab-mismatch-baseline"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target, expected_sha256=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_sha256 is None or sha256(target) == expected_sha256):
        return
    temporary = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            print("Downloading " + target.name, flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "slime-lab-mismatch-baseline"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as f:
                shutil.copyfileobj(response, f, length=4 * 1024**2)
            if expected_sha256 is not None and sha256(temporary) != expected_sha256:
                raise ValueError("SHA256 mismatch for " + target.name)
            temporary.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def make_record(row, split, index):
    if "####" not in row["answer"]:
        raise ValueError("GSM8K answer missing #### delimiter")
    label = row["answer"].rsplit("####", 1)[1].strip()
    question = row["question"].strip()
    return {
        "prompt": [{"role": "user", "content": question + PROMPT_SUFFIX}],
        "label": label,
        "metadata": {
            "source": "gsm8k", "source_split": split, "source_index": index,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        },
    }


def make_splits(train, test, seed=2026, dev_count=256):
    records = [make_record(row, "train", i) for i, row in enumerate(train)]
    random.Random(seed).shuffle(records)
    dev = records[:dev_count]
    train_records = records[dev_count:]
    train_hashes = {r["metadata"]["question_sha256"] for r in train_records}
    if train_hashes & {r["metadata"]["question_sha256"] for r in dev}:
        raise ValueError("Training/development questions overlap")
    test_records = [make_record(row, "test", i) for i, row in enumerate(test)]
    if train_hashes & {r["metadata"]["question_sha256"] for r in test_records}:
        raise ValueError("Training/test questions overlap")
    return {
        "train": train_records, "dev": dev, "test": test_records,
        "smoke_train": train_records[:64], "smoke_dev": dev[:8],
    }


def prepare_dataset():
    root = data_root() / "datasets/gsm8k"
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "source.json"
    if lock.exists():
        revision = json.loads(lock.read_text())["revision"]
    else:
        revision = get_json("https://api.github.com/repos/" + GSM_REPO + "/commits/master")["sha"]
        lock.write_text(json.dumps({"repo": GSM_REPO, "revision": revision}, indent=2) + "\n")
    sources = {}
    for split in ("train", "test"):
        source = root / ("original_" + split + ".jsonl")
        url = "https://raw.githubusercontent.com/" + GSM_REPO + "/" + revision + "/grade_school_math/data/" + split + ".jsonl"
        download(url, source)
        sources[split] = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    splits = make_splits(sources["train"], sources["test"])
    manifest = {"repo": GSM_REPO, "revision": revision, "split_seed": 2026, "files": {}}
    for name, rows in splits.items():
        target = root / (name + ".jsonl")
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["files"][name] = {"count": len(rows), "sha256": sha256(target)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def prepare_model(size):
    name = "Qwen3-" + size
    root = data_root() / "models" / name
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "source.json"
    if lock.exists():
        source = json.loads(lock.read_text())
        revision = source["revision"]
        info = get_json("https://huggingface.co/api/models/Qwen/" + name + "/revision/" + revision + "?blobs=true")
    else:
        info = get_json("https://huggingface.co/api/models/Qwen/" + name + "?blobs=true")
        revision = info["sha"]
        lock.write_text(json.dumps({"repo": "Qwen/" + name, "revision": revision}, indent=2) + "\n")
    manifest = {"repo": "Qwen/" + name, "revision": revision, "files": {}}
    allowed = {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja",
        "model.safetensors.index.json",
    }
    jobs = []
    for item in info["siblings"]:
        filename = item["rfilename"]
        if filename not in allowed and not filename.endswith(".safetensors"):
            continue
        expected = (item.get("lfs") or {}).get("sha256")
        url = "https://huggingface.co/Qwen/" + name + "/resolve/" + revision + "/" + urllib.parse.quote(filename)
        jobs.append((filename, url, expected))
    if not any(name.endswith(".safetensors") for name, _, _ in jobs):
        raise ValueError("No model weights listed by Hugging Face")
    def fetch(job):
        filename, url, expected = job
        target = root / filename
        download(url, target, expected)
        return filename, {"sha256": sha256(target), "bytes": target.stat().st_size}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for filename, details in pool.map(fetch, jobs):
            manifest["files"][filename] = details
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared " + name + " at revision " + revision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("0.6B", "1.7B", "both", "none"), default="0.6B")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()
    if not args.skip_data:
        prepare_dataset()
    sizes = ("0.6B", "1.7B") if args.model == "both" else (() if args.model == "none" else (args.model,))
    for size in sizes:
        prepare_model(size)


if __name__ == "__main__":
    main()
