"""Prepare and verify the byte-identical MATH inputs used by the 910B3 run."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

HASHES = {
    "train": "43bcc3c7de547eb4beaebda6fc3a0edab12b8838cd4ed0899e569466fe50a0ca",
    "validation": "613cc4b7629c69edbc50b5f9d6441d8536839812749be26d66518b50018340bb",
}
SOURCES = {
    "train": (
        "DigitalLearningGmbH/MATH-lighteval",
        "0530c78699ea5e8eb5530600900e1f328b48acad",
        "data/train-00000-of-00001.parquet",
        "eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933",
    ),
    "validation": (
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "test.jsonl",
        "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    ),
}
REPAIRS = {888: "2", 1011: "9", 5341: "0", 5343: "0"}
SUFFIX = "Please solve the problem and put the final answer in \\boxed{}."


def verify(root):
    for name, digest in HASHES.items():
        path = root / (name + ".jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Mismatched {name} split: {path}")
    return dict(HASHES)


def prepare(root, train_parquet=None, test_jsonl=None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from slime.rollout.rm_hub.math_utils import extract_answer

    if root.exists():
        return verify(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as staging:
        new = Path(staging) / "data"
        new.mkdir()
        for name, (repo, revision, filename, digest) in SOURCES.items():
            supplied = train_parquet if name == "train" else test_jsonl
            path = supplied or Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Changed upstream source: {path}")
            if name == "train":
                records = pq.read_table(path).to_pylist()
            else:
                records = [json.loads(line) for line in path.read_text().splitlines()]
            with (new / (name + ".jsonl")).open("w", encoding="utf-8") as stream:
                for i, row in enumerate(records):
                    if name == "train":
                        label = extract_answer(row["solution"]) or REPAIRS[i]
                        meta = dict(source="MATH/train", source_index=i, subject=row["type"], level=row["level"])
                    else:
                        label = row["answer"]
                        meta = dict(
                            source="MATH-500/test",
                            source_index=i,
                            unique_id=row["unique_id"],
                            subject=row["subject"],
                            level=f"Level {row['level']}",
                        )
                    prompt = [dict(role="user", content=row["problem"] + "\n" + SUFFIX)]
                    stream.write(
                        json.dumps(dict(prompt=prompt, label=label, metadata=meta), ensure_ascii=False) + "\n"
                    )
        verify(new)
        new.rename(root)
    return dict(HASHES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--test-jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.train_parquet, args.test_jsonl)))
