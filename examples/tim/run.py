#!/usr/bin/env python3
"""Prepare, CPU-preflight, or launch a frozen TIM experiment in Docker.

No downloads, package installs, or GPU jobs occur during plan/prepare/preflight.
The train action starts a fresh experiment; historical continuation is separate.
"""

import argparse
import hashlib
import json
import posixpath
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import gpu_count, load_config, native_command, runtime_environment

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_datasets(config, train_path, eval_path):
    train, evaluation = rows(train_path), rows(eval_path)
    policy = config["data"]
    if not train or not evaluation:
        raise ValueError("Training and evaluation datasets must be nonempty")
    for name, data in (("train", train), ("eval", evaluation)):
        expected = policy[name + "_questions"]
        if expected is not None and len(data) != expected:
            raise ValueError("Unexpected " + name + " question count: " + str(len(data)))
        if any("prompt" not in item or "label" not in item for item in data):
            raise ValueError("Every dataset row requires prompt and label")
    prompt_key = lambda item: json.dumps(item["prompt"], sort_keys=True, ensure_ascii=False)
    train_prompts, eval_prompts = set(map(prompt_key, train)), set(map(prompt_key, evaluation))
    if len(train_prompts) != len(train) or len(eval_prompts) != len(evaluation):
        raise ValueError("Dataset contains duplicate prompts")
    if train_prompts & eval_prompts:
        raise ValueError("Training/evaluation prompt overlap")
    train_indices = {item.get("metadata", {}).get("source_index") for item in train} - {None}
    eval_indices = {item.get("metadata", {}).get("source_index") for item in evaluation} - {None}
    if train_indices & eval_indices:
        raise ValueError("Training/evaluation source-index overlap")
    for item in train:
        metadata = item.get("metadata", {})
        count = metadata.get("screen_correct_count")
        if (
            metadata.get("screen_samples") != policy["screen_samples"]
            or type(count) is not int
            or not policy["min_correct"] <= count <= policy["max_correct"]
        ):
            raise ValueError("Training data does not satisfy the declared frozen screening policy")
    return {
        "train_questions": len(train),
        "eval_questions": len(evaluation),
        "holdout_overlap": 0,
        "train_sha256": sha(train_path),
        "eval_sha256": sha(eval_path),
        "screening": policy,
        "screening_applies_to": "training_only",
        "evaluation_is_not_difficulty_filtered": True,
    }


def model_mount(path):
    """Keep Hugging Face snapshot→blob symlinks valid without copying weights."""
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("--model-path must be a local model directory")
    root = path
    if path.parent.name == "snapshots" and (path.parent.parent / "blobs").is_dir():
        root = path.parent.parent
    for file in path.rglob("*"):
        if file.is_symlink():
            try:
                file.resolve(strict=True).relative_to(root)
            except (ValueError, FileNotFoundError) as error:
                raise ValueError("Model symlink escapes its mounted cache root: " + str(file)) from error
    destination = "/model"
    inside = destination if root == path else destination + "/" + str(path.relative_to(root))
    return root, inside


def check_model(path):
    config = json.loads((Path(path) / "config.json").read_text())
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
    }
    if any(config.get(key) != value for key, value in expected.items()) or config.get("num_experts", 0):
        raise ValueError("These presets require the dense Qwen3-0.6B architecture")
    files = sorted(Path(path).glob("*.safetensors"))
    if not files or not (Path(path) / "tokenizer_config.json").is_file():
        raise ValueError("Local safetensor weights and tokenizer assets are required")
    artifacts = {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}
    return {"architecture": expected, "files_sha256": artifacts}


def source_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def safe_extract(stream, destination):
    """Validate member and link destinations before extracting a Git archive."""
    for member in stream.getmembers():
        if (
            member.name.startswith("/")
            or ".." in Path(member.name).parts
            or not (member.isfile() or member.isdir() or member.issym() or member.islnk())
        ):
            raise ValueError("Invalid source archive member")
        if member.issym() or member.islnk():
            target = member.linkname
            if member.issym():
                target = posixpath.join(posixpath.dirname(member.name), target)
            normalized = posixpath.normpath(target)
            if target.startswith("/") or normalized == ".." or normalized.startswith("../"):
                raise ValueError("Source archive link escapes its root")
    stream.extractall(destination)


def copy_sources(output, source_repo, commit):
    """Archive the selected commit, then overlay only explicit research packages."""
    destination = output / "sources"
    destination.mkdir()
    archive = destination / "slime.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "-C", str(source_repo), "archive", commit], stdout=stream, check=True)
    pinned = destination / "slime"
    pinned.mkdir()
    with tarfile.open(archive) as stream:
        safe_extract(stream, pinned)
    archive.unlink()
    extensions = destination / "extensions/slime_plugins"
    extensions.mkdir(parents=True)
    (extensions / "__init__.py").write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
    )
    for package in ("tim", "nvfp4_sim"):
        shutil.copytree(
            REPO / "slime_plugins" / package,
            extensions / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copytree(
        Path(__file__).parent, destination / "entry", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    return source_hashes(destination)


def check_output_location(output, model, noise=None):
    roots = [Path(__file__).parent, REPO / "slime_plugins/tim", REPO / "slime_plugins/nvfp4_sim", REPO / ".git", model]
    if noise is not None:
        roots.append(noise)
    for root in roots:
        if root.resolve() == output or root.resolve() in output.parents:
            raise ValueError("Output must be outside copied source directories, model assets and noise bank")
    if output == REPO:
        raise ValueError("Repository root cannot be an experiment output")


def cpu_driver_library():
    """TE imports libcuda even when no GPU is exposed; mount only the library."""
    ldconfig = shutil.which("ldconfig")
    if ldconfig is None:
        return None
    result = subprocess.run(
        [ldconfig, "-p"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False
    )
    for line in result.stdout.splitlines():
        if line.strip().startswith("libcuda.so.1 ") and "=>" in line:
            path = Path(line.partition("=>")[2].strip())
            if path.is_file():
                return str(path.resolve(strict=True))
    return None


def make_plan(args):
    config = load_config(args.config, args.set)
    source_repo = Path(args.slime_repo).resolve()
    commit = git(source_repo, "rev-parse", config["runtime"]["slime_commit"] + "^{commit}")
    model = Path(args.model_path).resolve(strict=True)
    mount, inside = model_mount(model)
    train, evaluation = Path(args.train_data).resolve(strict=True), Path(args.eval_data).resolve(strict=True)
    output = Path(args.output_dir).expanduser().resolve()
    if any(str(p).find(",") >= 0 for p in (mount, train, evaluation, output)):
        raise ValueError("Docker bind-mount paths cannot contain commas")
    devices = [item.strip() for item in args.gpus.split(",") if item.strip()] if args.gpus else []
    if devices and (len(devices) != gpu_count(config) or len(devices) != len(set(devices))):
        raise ValueError("GPU list must contain the configured number of distinct devices")
    if any(not (item.isdigit() or item.startswith("GPU-")) for item in devices):
        raise ValueError("Use physical GPU indices or GPU UUIDs")
    noise = None
    if config["rollout"]["precision"] == "fixed_noise":
        if not args.noise_bank:
            raise ValueError("Fixed-noise mode requires a separately prepared --noise-bank directory")
        noise = Path(args.noise_bank).resolve(strict=True)
        if "," in str(noise) or not (noise / "manifest.json").is_file():
            raise ValueError("Invalid noise bank path")
    elif args.noise_bank:
        raise ValueError("--noise-bank is only valid for fixed-noise rollout")
    check_output_location(output, model, noise)
    return {
        "schema_version": 1,
        "config": config,
        "slime_commit": commit,
        "extension_commit": git(REPO, "rev-parse", "HEAD"),
        "paths": dict(
            model=str(model),
            model_mount=str(mount),
            model_container=inside,
            train=str(train),
            evaluation=str(evaluation),
            output=str(output),
            noise_bank=str(noise) if noise else None,
        ),
        "data": check_datasets(config, train, evaluation),
        "gpus": devices,
        "required_gpus": gpu_count(config),
        "cpu_driver_library": cpu_driver_library(),
        "training_command": native_command(config, inside),
        "environment": runtime_environment(config, inside),
        "fresh_initialization": True,
        "historical_resume": False,
    }


def prepare(args, plan):
    output = Path(plan["paths"]["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; previous artifacts are immutable")
    output.mkdir(parents=True, exist_ok=True)
    plan["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["model"] = check_model(Path(plan["paths"]["model"]))
    plan["source_files_sha256"] = copy_sources(output, Path(args.slime_repo), plan["slime_commit"])
    plan["source_bundle_sha256"] = hashlib.sha256(
        json.dumps(plan["source_files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    image = plan["config"]["runtime"]["image"]
    # Inspect never pulls. Persist the immutable local ID used for both phases.
    plan["image_id"] = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()
    write_json(output / "resolved-config.json", plan)
    write_json(
        output / "mismatch.json",
        dict(
            use_tis=False,
            use_rs=False,
            tis_level="token",
            rs_level="token",
            tis_mode="truncate",
            tis_lower_bound=0.5,
            tis_upper_bound=2.0,
            rs_lower_bound=None,
            rs_upper_bound=None,
            rs_veto_threshold=None,
            tis_batch_normalize=False,
        ),
    )
    write_json(
        output / "status.json",
        dict(
            state="prepared",
            completed_updates=0,
            gpu_jobs_launched=0,
            source_bundle_sha256=plan["source_bundle_sha256"],
        ),
    )
    return plan


def verify_prepared(plan):
    output = Path(plan["paths"]["output"])
    if source_hashes(output / "sources") != plan["source_files_sha256"]:
        raise RuntimeError("Frozen experiment source changed")
    if check_datasets(plan["config"], plan["paths"]["train"], plan["paths"]["evaluation"]) != plan["data"]:
        raise RuntimeError("Dataset changed after preparation")
    if check_model(Path(plan["paths"]["model"])) != plan["model"]:
        raise RuntimeError("Model assets changed after preparation")


def idle_gpu_uuids(devices):
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpu_rows = [tuple(part.strip() for part in line.split(",")) for line in text.splitlines()]
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    selected = []
    for device in devices:
        matches = [row for row in gpu_rows if device in row[:2]]
        if len(matches) != 1:
            raise ValueError("GPU not found: " + device)
        index, uuid, memory, utilization = matches[0]
        if uuid in {item.strip() for item in active} or int(memory) > 64 or int(utilization) != 0:
            raise RuntimeError("Selected GPU is busy; no experiment started: " + index)
        selected.append(uuid)
    if len(selected) != len(set(selected)):
        raise ValueError("GPU indices and UUIDs resolve to duplicate physical devices")
    return selected


def docker_command(plan, action):
    paths, runtime = plan["paths"], plan["config"]["runtime"]
    output = Path(paths["output"])
    identity = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--pull",
        "never",
        "--name",
        "tim-" + identity + "-" + action,
        "--label",
        "tim.run_id=" + identity,
        "--cpus",
        str(runtime["cpus"]),
        "--memory",
        str(runtime["memory_gib"]) + "g",
        "--pids-limit",
        "4096",
        "--shm-size",
        "8g",
    ]
    mounts = [
        (output, "/output", False),
        (output / "sources/slime", "/runtime/slime", True),
        (output / "sources/extensions", "/runtime/extensions", True),
        (output / "sources/entry", "/runtime/entry", True),
        (Path(paths["model_mount"]), "/model", True),
        (Path(paths["train"]), "/datasets/train.jsonl", True),
        (Path(paths["evaluation"]), "/datasets/eval.jsonl", True),
    ]
    if paths["noise_bank"]:
        mounts.append((Path(paths["noise_bank"]), "/noise-bank", True))
    if action == "preflight" and plan.get("cpu_driver_library"):
        mounts.append((Path(plan["cpu_driver_library"]), "/usr/lib/x86_64-linux-gnu/libcuda.so.1", True))
    for source, target, readonly in mounts:
        command.extend(
            ["--mount", "type=bind,src=" + str(source) + ",dst=" + target + (",readonly" if readonly else "")]
        )
    for key, value in plan["environment"].items():
        command.extend(["--env", key + "=" + value])
    if action == "preflight":
        command.extend(["--network", "none", "--env", "CUDA_VISIBLE_DEVICES=", "--env", "NVIDIA_VISIBLE_DEVICES=void"])
    else:
        if not plan["gpus"]:
            raise ValueError("train requires --gpus")
        # Quotes are part of Docker's CSV DeviceRequest grammar, not shell quoting.
        command.extend(["--gpus", '"device=' + ",".join(plan["gpus"]) + '"'])
    command.extend(
        ["--workdir", "/runtime/entry", plan["image_id"], "python", "/runtime/entry/container_driver.py", action]
    )
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "preflight", "train"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=JSON_VALUE")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--slime-repo", default=REPO, type=Path, help="Git repository containing the pinned Slime commit"
    )
    parser.add_argument("--noise-bank", type=Path)
    parser.add_argument("--gpus", default="", help="Physical GPU indices or UUIDs; train requires idle devices")
    args = parser.parse_args()
    plan = make_plan(args)
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    prepared = Path(plan["paths"]["output"]) / "resolved-config.json"
    if prepared.exists():
        if args.action == "prepare":
            raise FileExistsError("Preparation already exists")
        previous = json.loads(prepared.read_text())
        for key in ("config", "paths", "data", "training_command", "environment", "slime_commit"):
            if previous[key] != plan[key]:
                raise ValueError("Launch differs from prepared experiment: " + key)
        previous["gpus"] = plan["gpus"]
        plan = previous
    else:
        plan = prepare(args, plan)
    verify_prepared(plan)
    if args.action == "prepare":
        print("Prepared: " + str(prepared))
        return 0
    subprocess.run(docker_command(plan, "preflight"), check=True)
    if args.action == "train":
        if not args.gpus:
            parser.error("train requires --gpus")
        selected = idle_gpu_uuids(plan["gpus"])
        plan["gpus"] = selected
        write_json(
            Path(plan["paths"]["output"]) / "launch.json",
            dict(
                gpu_uuids=selected,
                image_id=plan["image_id"],
                source_bundle_sha256=plan["source_bundle_sha256"],
                command=docker_command(plan, "train"),
            ),
        )
        return subprocess.call(docker_command(plan, "train"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
