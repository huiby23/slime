"""Prepare a four-GPU continuation from a complete best/last snapshot."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

from prepare_baseline import REPO, cli_options, replace_option
from watch_gpu import resume_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-attempt", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 6, 7])
    parser.add_argument("--dp-validation", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "results/status.json").exists()
    source = args.resume_attempt.resolve()
    config = json.loads((source / "config.json").read_text())
    checkpoint = resume_checkpoint(source)
    validation = json.loads(args.dp_validation.read_text())
    assert validation["passed"] and validation["world_size"] == 4 and validation["replicas_equal"]
    assert config["experiment_mode"] == "formal" and config["training"]["global_batch_size"] == 64
    assert config["training"]["samples_per_prompt"] == config["training"]["prompts_per_update"] == 8
    config = copy.deepcopy(config)
    config.update(
        run_id="nvfp4-dp4-" + output.name,
        host_run_dir=str(output / "results"),
        physical_gpus=args.gpus,
        parallel_nvfp4=True,
        checkpoint_policy="best-last",
        resume_checkpoint=checkpoint,
    )
    config["assigned_gpu_uuids"] = [
        subprocess.check_output(
            ["nvidia-smi", "--id=" + str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        for gpu in args.gpus
    ]
    config["training"].update(gpu_count=4, data_parallel_size=4, save_every_updates=1, checkpoint_policy="best-last")
    config["environment"].update(CUDA_VISIBLE_DEVICES="0,1,2,3", TENSORBOARD_DIR="/nvfp4-run/results/tensorboard")
    command = config["training_command"]
    original = cli_options(command)
    command[:] = [token for token in command if token not in ("--finetune", "--no-load-optim", "--no-load-rng")]
    command[1] = "/quant/examples/nvfp4_sim/train_best_last.py"
    for option, value in [
        ("--load", "/nvfp4-resume"),
        ("--ckpt-step", checkpoint["step"]),
        ("--start-rollout-id", checkpoint["completed_updates"]),
        ("--actor-num-gpus-per-node", 4),
        ("--rollout-num-gpus", 4),
        ("--num-gpus-per-node", 4),
        ("--save-interval", 1),
        ("--custom-eval-rollout-log-function-path", "nvfp4_sim.checkpoint_policy.eval_log_hook"),
        (
            "--sglang-json-model-override-args",
            json.dumps({"architectures": ["NVFP4SimQwen3ForCausalLM"], "nvfp4_sim_weight_cache": True}),
        ),
    ]:
        replace_option(command, option, value)
    if "--use-tensorboard" not in command:
        command.append("--use-tensorboard")
    for index, token in enumerate(config["ray_start_command"]):
        if token.startswith("--num-gpus="):
            config["ray_start_command"][index] = "--num-gpus=4"
        if token.startswith("--num-cpus="):
            config["ray_start_command"][index] = "--num-cpus=32"
    updated = cli_options(command)
    allowed = {
        "--load",
        "--finetune",
        "--no-load-optim",
        "--no-load-rng",
        "--actor-num-gpus-per-node",
        "--rollout-num-gpus",
        "--num-gpus-per-node",
        "--save-interval",
        "--sglang-json-model-override-args",
    }
    assert all(updated.get(key) == value for key, value in original.items() if key not in allowed)
    comparison = {
        "passed": True,
        "scope": "Continuation: unchanged global batch, loss, data, optimizer and NVFP4 recipe; DP=4, cached inference casts, best/last, TensorBoard",
        "changes": {
            key: {"old": original.get(key), "new": updated.get(key)}
            for key in original.keys() | updated.keys()
            if original.get(key) != updated.get(key)
        },
        "resume_checkpoint": checkpoint,
    }
    (output / "baseline-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "mismatch.yaml").write_bytes((source / "mismatch.yaml").read_bytes())
    for path in ["results/logs", "cache/xdg/torch/kernels", "cache/torchinductor", "cache/triton", "tmp/sg-a"]:
        (output / path).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.template / "manifest.json").read_text())
    docker = manifest["docker_command"]
    docker[docker.index("--gpus") + 1] = '"device=' + ",".join(config["assigned_gpu_uuids"]) + '"'
    for index, token in enumerate(docker):
        if token == "--mount" and "dst=/nvfp4-run" in docker[index + 1].split(","):
            docker[index + 1] = "type=bind,src=%s,dst=/nvfp4-run" % output
    name = "huibingyu-nvfp4-dp4-" + output.name
    docker[2:2] = [
        "--name",
        name,
        "--label",
        "lab.nvfp4-parallel=" + output.name,
        "--mount",
        "type=bind,src=%s,dst=/nvfp4-resume,readonly" % checkpoint["directory"],
    ]
    manifest.update(
        docker_command=docker,
        container=name,
        simulator_commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        dp_validation=str(args.dp_validation.resolve()),
        resume_checkpoint=checkpoint,
        physical_gpus=args.gpus,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight:
        preflight = docker.copy()
        preflight[preflight.index("--name") + 1] = name + "-preflight"
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
    print(
        json.dumps(
            {"output": str(output), "resume_completed_updates": checkpoint["completed_updates"], "gpus": args.gpus}
        )
    )


if __name__ == "__main__":
    main()
