"""Prepare/run an NVFP4 simulation using the recorded BF16 baseline command.

This host-side script needs only Python's standard library and Docker. It mounts
the original baseline checkout read-only and overlays only the simulator plugin.
By default it prepares inspectable configuration without starting a GPU job.
"""

import argparse
import copy
import json
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path("/data2/huibingyu/slime-v040")
STUDY = Path("/home/huibingyu/rl/slime-mismatch")
MODEL = Path("/data2/chenyihong/models/models--Qwen--Qwen3-0.6B")


def replace_option(command, option, value):
    if option in command:
        command[command.index(option) + 1] = str(value)
    else:
        command.extend((option, str(value)))


def assert_gpu_idle(gpu):
    pids = subprocess.check_output(
        ["nvidia-smi", "--id=" + str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    memory, utilization = map(
        int,
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + str(gpu),
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(","),
    )
    if pids or memory > 64 or utilization:
        raise RuntimeError("GPU %s is busy; no CUDA context or training job was started" % gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=STUDY / "single/control-seed42/config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument(
        "--quick-smoke", action="store_true", help="2 prompts x 2 responses, 64 generated tokens, 8 eval questions"
    )
    parser.add_argument("--run", action="store_true", help="Run after checking that the selected GPU is idle")
    parser.add_argument(
        "--preflight", action="store_true", help="Resolve CLI/YAML and both entry points in a CPU-only container"
    )
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results/status.json").exists():
        raise RuntimeError("This run already has results; choose a fresh output directory")
    baseline = json.loads(args.baseline_config.read_text())
    source = ASSETS / "code/slime"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != baseline["source_commit"]:
        raise RuntimeError("Original baseline source revision does not match its recorded config")
    config = copy.deepcopy(baseline)
    config["run_id"] = "nvfp4-simulation-" + output.name
    config["arm"] = "nvfp4-default-recipe"
    config["physical_gpus"] = [args.gpu]
    config["assigned_gpu_uuids"] = [
        subprocess.check_output(
            ["nvidia-smi", "--id=" + str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
    ]
    config["gpu_allocation_managed_by_sequence"] = False
    config["container_run_dir"] = "/nvfp4-run/results"
    config["host_run_dir"] = str(output / "results")
    config["training"]["updates"] = args.updates
    config["training"]["save_every_updates"] = 1
    config["training"]["precision"] = "NVFP4 default training recipe simulation"
    command = [
        value.replace(baseline["container_run_dir"], config["container_run_dir"])
        for value in baseline["training_command"]
    ]
    replace_option(command, "--num-rollout", args.updates)
    replace_option(command, "--save-interval", 1)
    replace_option(command, "--custom-model-provider-path", "nvfp4_sim.megatron.model_provider")
    replace_option(
        command, "--sglang-json-model-override-args", json.dumps({"architectures": ["NVFP4SimQwen3ForCausalLM"]})
    )
    original_custom_config = Path(command[command.index("--custom-config-path") + 1].replace("/study", str(STUDY), 1))
    # The recorded baseline stores JSON-compatible YAML. Retain all TIS settings.
    custom_config = json.loads(original_custom_config.read_text())
    custom_config.update(nvfp4_sim_chunk_blocks=8, nvfp4_sim_include_output_layer=True)
    (output / "mismatch.yaml").write_text(json.dumps(custom_config, indent=2) + "\n")
    replace_option(command, "--custom-config-path", "/nvfp4-run/mismatch.yaml")
    if args.quick_smoke:
        for option, value in (
            ("--rollout-batch-size", 2),
            ("--n-samples-per-prompt", 2),
            ("--global-batch-size", 4),
            ("--rollout-max-response-len", 64),
            ("--eval-max-response-len", 64),
            ("--eval-interval", 1),
        ):
            replace_option(command, option, value)
        index = command.index("--eval-prompt-data") + 2
        validation = Path(command[index].replace("/experiment", str(ASSETS), 1)).read_text().splitlines()[:8]
        (output / "validation.jsonl").write_text("\n".join(validation) + "\n")
        command[index] = "/nvfp4-run/validation.jsonl"
        config["training"].update(
            prompts_per_update=2,
            samples_per_prompt=2,
            global_batch_size=4,
            response_token_limit=64,
            validation_questions=8,
            eval_every_updates=1,
        )
    config["training_command"] = command
    env = config["environment"]
    for key, value in env.items():
        env[key] = (
            value.replace("/study/cache", "/nvfp4-run/cache")
            .replace("/study/tmp", "/nvfp4-run/tmp")
            .replace(baseline["container_run_dir"], config["container_run_dir"])
        )
    env["PYTHONPATH"] = "/quant/slime_plugins:/workspace/slime:/root/Megatron-LM:/study/scripts"
    env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "nvfp4_sim.sglang_models"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    for name in ("results/logs", "cache", "tmp/sg-a"):
        (output / name).mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    docker = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--gpus",
        "device=" + str(args.gpu),
        "--network",
        "none",
        "--shm-size",
        "8g",
        "--workdir",
        "/workspace/slime",
    ]
    for host, target, readonly in (
        (REPO, "/quant", True),
        (source, "/workspace/slime", True),
        (ASSETS, "/experiment", True),
        (STUDY, "/study", True),
        (MODEL, "/shared/qwen3-0.6B", True),
        (output, "/nvfp4-run", False),
    ):
        docker.extend(("--mount", "type=bind,src=%s,dst=%s%s" % (host, target, ",readonly" if readonly else "")))
    # Pass the Python path to the initial supervisor; it then propagates all
    # configured environment values to Ray and the SGLang worker processes.
    docker.extend(("--env", "PYTHONPATH=" + env["PYTHONPATH"], "--env", "PYTHONDONTWRITEBYTECODE=1"))
    docker.extend(
        (
            "--entrypoint",
            "python3",
            config["image_id"],
            "/study/scripts/run_online_training.py",
            "/nvfp4-run/config.json",
        )
    )
    manifest = {
        "baseline_commit": revision,
        "simulator_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "simulator_dirty": bool(
            subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
        ),
        "recipe": {
            "fprop": "NVFP4",
            "dgrad": "NVFP4",
            "wgrad": "NVFP4",
            "weight_tile": [16, 16],
            "activation_gradient_tile": [1, 16],
            "stochastic_gradient_rounding": True,
            "wgrad_rht": True,
            "include_output_layer": True,
            "gemm_accumulation": "FP32",
        },
        "docker_command": docker,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared " + str(output / "config.json"))
    print(shlex.join(docker))
    if args.preflight or args.run:
        preflight = docker.copy()
        # TE dynamically links libcuda even for CPU reference functions. Keep
        # driver libraries available, but mask every CUDA device in preflight.
        index = preflight.index("--entrypoint")
        preflight[index:] = [
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            "--entrypoint",
            "python3",
            config["image_id"],
            "/quant/examples/nvfp4_sim/preflight.py",
            "/nvfp4-run",
        ]
        with (output / "preflight.log").open("w") as log:
            subprocess.run(preflight, check=True, stdout=log, stderr=subprocess.STDOUT)
        print("CPU preflight passed: " + str(output / "configuration-check.json"))
    if args.run:
        assert_gpu_idle(args.gpu)
        subprocess.run(docker, check=True)


if __name__ == "__main__":
    main()
