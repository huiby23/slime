#!/usr/bin/env python3
"""User-local Enroot runtime; no Docker daemon storage or system changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO.parent / "slime-data" / "lab-mismatch"
IMAGE = "slimerl/slime:v0.4.0-cu129"
CONTAINER = "slime-v040-cu129"
ASSETS = {
    "enroot.deb": (
        "https://github.com/NVIDIA/enroot/releases/download/v4.2.1/enroot_4.2.1-1_amd64.deb",
        "5955c51e88df0c5ed538d87abb5264609d4bad55fa8570745ed5d4ec66c8ab91",
    ),
    "jq": (
        "https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64",
        "020468de7539ce70ef1bceaf7cde2e8c4f2ca6c3afb84642aabc5c97d9fc2a0d",
    ),
    "zstd.deb": (
        "https://archive.ubuntu.com/ubuntu/pool/universe/libz/libzstd/zstd_1.4.4+dfsg-3ubuntu0.1_amd64.deb",
        "508bd86b6ddc5ee1ae7e6ecfcf86c6cdbde1256e2bf5aab35e44f585b6d19b8e",
    ),
}


def data_root():
    return Path(os.environ.get("LAB_DATA_ROOT", str(DEFAULT_DATA))).resolve()


def environment():
    root = data_root()
    prefix = root / "tools" / "enroot"
    env = os.environ.copy()
    env.update({
        "PATH": str(prefix / "usr/bin") + ":" + str(root / "downloads") + ":" + env.get("PATH", ""),
        "ENROOT_LIBRARY_PATH": str(prefix / "usr/lib/enroot"),
        "ENROOT_SYSCONF_PATH": str(prefix / "etc/enroot"),
        "ENROOT_CONFIG_PATH": str(root / "runtime/config"),
        "ENROOT_CACHE_PATH": str(root / "runtime/cache"),
        "ENROOT_DATA_PATH": str(root / "runtime/containers"),
        "ENROOT_RUNTIME_PATH": str(root / "runtime/run"),
        "ENROOT_TEMP_PATH": str(root / "runtime/tmp"),
        "ENROOT_MAX_PROCESSORS": "8",
        "ENROOT_MAX_CONNECTIONS": "4",
        "ENROOT_TRANSFER_RETRIES": "3",
    })
    for key in ("ENROOT_CONFIG_PATH", "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_RUNTIME_PATH", "ENROOT_TEMP_PATH"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def bootstrap():
    root = data_root()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        target = downloads / name
        if not target.exists():
            print("Downloading " + name, flush=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as f:
                shutil.copyfileobj(response, f)
            temporary.replace(target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Checksum mismatch: " + str(target))
    prefix = root / "tools/enroot"
    for name in ("enroot.deb", "zstd.deb"):
        subprocess.run(["dpkg-deb", "-x", str(downloads / name), str(prefix)], check=True)
    (downloads / "jq").chmod(0o755)
    env = environment()
    for command in ("enroot", "jq", "zstd", "mksquashfs", "unsquashfs", "parallel", "nvidia-container-cli"):
        if not shutil.which(command, path=env["PATH"]):
            raise RuntimeError("Missing prerequisite: " + command)
    return env


def setup():
    env = bootstrap()
    root = data_root()
    if shutil.disk_usage(root).free < 160 * 1024**3:
        raise RuntimeError("Need 160 GiB free under LAB_DATA_ROOT for image cache, rootfs and models.")
    image = root / "runtime/slime-v040-cu129.sqsh"
    if not image.is_file():
        subprocess.run(["enroot", "import", "-o", str(image), "docker://" + IMAGE], env=env, check=True)
    if not (Path(env["ENROOT_DATA_PATH"]) / CONTAINER).is_dir():
        subprocess.run(["enroot", "create", "--name", CONTAINER, str(image)], env=env, check=True)
    (root / "runtime/runtime.json").write_text(json.dumps({
        "image": IMAGE, "container": CONTAINER, "enroot": "4.2.1",
        "slime_checkout": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2) + "\n")
    print("Runtime prepared: " + str(root / "runtime/runtime.json"), flush=True)


def execute(command, gpus="0,1", writable=False):
    env = environment()
    if gpus not in ("none", "0", "1", "0,1"):
        raise ValueError("This experiment is authorized only for physical GPUs 0 and 1.")
    if not (Path(env["ENROOT_DATA_PATH"]) / CONTAINER).is_dir():
        raise RuntimeError("Runtime is not prepared; run runtime.py setup first.")
    env["NVIDIA_VISIBLE_DEVICES"] = gpus
    env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
    args = ["enroot", "start", "--root"]
    if writable:
        args.append("--rw")
    args += [
        "--mount", str(REPO) + ":/workspace/slime",
        "--mount", str(data_root()) + ":/data",
        "--env", "NVIDIA_VISIBLE_DEVICES=" + gpus,
        "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        CONTAINER,
    ] + command
    return subprocess.run(args, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bootstrap", "setup", "exec"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--writable", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "bootstrap":
        bootstrap()
    elif args.action == "setup":
        setup()
    else:
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            parser.error("exec requires a command after --")
        sys.exit(execute(command, args.gpus, args.writable))


if __name__ == "__main__":
    main()
