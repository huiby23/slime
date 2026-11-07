"""Generate corrected one-container commands for the four attention routes."""

import argparse
import json
from pathlib import Path


MODEL_HOST = "/data4/yaozhuang/models/Qwen3-4B-Base"
MODEL_CONTAINER = "/models/Qwen3-4B-Base"


ROUTES = ("c16", "naive-c8", "naive-c4", "attn-qat-c4")


def corrected_command(command):
    command = list(command)
    command[command.index("--prompt-data") + 1] = "/datasets/train.jsonl"
    for flag in ("--hf-checkpoint", "--load", "--tokenizer-path"):
        command[command.index(flag) + 1] = MODEL_CONTAINER
    index = command.index("--eval-prompt-data")
    del command[index + 1 : index + 3]
    command[index + 1 : index + 1] = ["gsm8k", "/workspace/w4a8/validation-50-old-study.jsonl"]
    return command


def shell_quote(value):
    import shlex

    return shlex.quote(str(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    for route in ROUTES:
        route_root = args.root / route
        command = corrected_command(json.loads((route_root / "command.json").read_text()))
        command_path = route_root / "command-corrected.json"
        command_path.write_text(json.dumps(command, indent=2) + "\n")
        index = command.index("--eval-prompt-data")
        assert command[index + 1 : index + 3] == ["gsm8k", "/workspace/w4a8/validation-50-old-study.jsonl"]
        assert command[index + 3] == "--n-samples-per-eval-prompt"

        env = dict(line.split("=", 1) for line in (route_root / "env.list").read_text().splitlines() if line)
        # The full repository mount is needed by train_best_last.py and Ray workers.
        env["PYTHONPATH"] = "/quant/slime_plugins:/workspace/slime:/root/Megatron-LM"
        env["W4A8_CHECKPOINT_ROOTS"] = json.dumps(["/workspace/w4a8/checkpoints/.snapshots"])
        env["W4A8_INITIAL_MODEL"] = MODEL_CONTAINER
        env["RAY_TMPDIR"] = "/workspace/w4a8/raytmp"
        name = f"yaozhuang-qwen3-4b-base-{route}"
        docker_args = [
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "lab.yaozhuang=qwen3-4b-attention",
            "--gpus",
            "device=$GPU",
            "--shm-size",
            "16g",
        ]
        for key, value in env.items():
            docker_args += ["--env", f"{key}={value}"]
        docker_args += [
            "--mount",
            f"type=bind,src=/home/yaozhuang/v16_attention/slime-qwen3-4b-w4a8,dst=/quant,readonly",
            "--mount",
            "type=bind,src=/home/yaozhuang/v16_attention/slime-qwen3-4b-w4a8,dst=/workspace/slime,readonly",
            "--mount",
            f"type=bind,src={args.root / 'datasets'},dst=/datasets,readonly",
            "--mount",
            f"type=bind,src={route_root},dst=/workspace/w4a8",
            "--mount",
            f"type=bind,src={MODEL_HOST},dst={MODEL_CONTAINER},readonly",
            "--workdir",
            "/workspace/slime",
            "--entrypoint",
            "python3",
            args.image,
        ]
        docker_args += command[1:]
        script = '#!/usr/bin/env bash\nset -euo pipefail\n: "${GPU:?Set GPU to an available host GPU index}"\n'
        script += "docker " + " ".join(
            '"device=${GPU}"' if value == "device=$GPU" else shell_quote(value) for value in docker_args
        ) + "\n"
        (route_root / "docker-command-corrected.sh").write_text(script)
        print(route, command[index : index + 5], name)


if __name__ == "__main__":
    main()
