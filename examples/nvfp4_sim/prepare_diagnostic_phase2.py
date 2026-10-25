"""Prepare and CPU-preflight frozen NVFP4 diagnostics; never update weights."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from watch_diagnostics import free_gpus, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    cli = parser.parse_args()
    root = cli.root.resolve()
    prior = cli.prior.resolve()
    root.mkdir(exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a new root for an already prepared experiment."
    (root / "jobs").mkdir(exist_ok=True)
    template = json.loads((prior / "cpu-container.json").read_text())
    command = copy.deepcopy(template["docker_command"])
    name = "huibingyu-nvfp4-phase2-cpu-20261026"
    command[command.index("--name") + 1] = name
    old_mount = "type=bind,src=" + str(prior) + ",dst=/diagnostic"
    command[command.index(old_mount)] = "type=bind,src=" + str(root) + ",dst=/diagnostic"
    index = command.index("--entrypoint")
    command[index:index] = ["--mount", "type=bind,src=" + str(prior) + ",dst=/prior-diagnostic,readonly"]
    inspected = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    if inspected.returncode:
        subprocess.run(command, check=True)
    else:
        assert any(m["Source"] == str(root) and m["Destination"] == "/diagnostic"
                   for m in json.loads(inspected.stdout)[0]["Mounts"])
    save(root / "cpu-container.json", dict(container=name, docker_command=command, cuda_devices_hidden=True))
    # Reuse the previous immutable artifacts through a read-only mount. Only
    # the export manifest is copied for the host queue's readiness check.
    link_code = """import shutil,pathlib
assert not __import__('torch').cuda.is_available()
r=pathlib.Path('/diagnostic');p=pathlib.Path('/prior-diagnostic')
(r/'hf/nvfp4-step100').mkdir(parents=True,exist_ok=True)
shutil.copy2(p/'hf/nvfp4-step100/export-validation.json',r/'hf/nvfp4-step100/export-validation.json')
"""
    subprocess.run(["docker", "exec", name, "python3", "-c", link_code], check=True)
    base = json.loads((prior / "jobs/nv100-bf16/job.json").read_text())
    base["model"] = base["model"].replace("/diagnostic/", "/prior-diagnostic/")
    plan = dict(formal_run=json.loads((prior / "plan.json").read_text())["formal_run"],
                cpu_container=name, origin=base["origin"], jobs=[], reservations=[],
                physical_gpu_allowlist=[1, 3, 4, 6, 7], max_concurrent_jobs=4, stop_on_job_failure=False,
                interpretation="Frozen diagnostics only. Zero optimizer updates. Original formal results remain immutable.")
    replay = json.loads((prior / "jobs/nv75-frozen-tis/job.json").read_text())
    replay["batch_path"] = replay["batch_path"].replace("/diagnostic/", "/prior-diagnostic/")
    replay["megatron_argv"] = [v.replace("/diagnostic/snapshots/", "/prior-diagnostic/snapshots/")
                               for v in replay["megatron_argv"]]
    for key in ["idle_observation", "launch_checked_at_utc", "replay_script_sha256", "assigned_physical_gpu",
                "expected_gpu_uuid", "physical_gpu", "reservation_host_pids"]:
        replay.pop(key, None)
    replay["checkpoint_selection"] = "Frozen weights after update 75; native iteration 74, batch for update 76."
    choose_code = """import torch,json
rows=sorted(torch.load('/prior-diagnostic/snapshots/train_75.pt',map_location='cpu',weights_only=False)['samples'],key=lambda r:r['rollout_position'])
groups=[list(range(i,i+8)) for i in range(0,64,8) if all(torch.as_tensor(r['advantages']).count_nonzero().item()>0 for r in rows[i:i+8])]
assert groups
print(json.dumps(groups[0]))
"""
    probe_positions = json.loads(subprocess.check_output(["docker", "exec", name, "python3", "-c", choose_code], text=True))
    variants = []
    for group, tracker, independent, stochastic in [
        ("legacy", False, False, True), ("tracker", True, False, True),
        ("isolated", True, True, True), ("deterministic", True, False, False),
    ]:
        for repeat in [False, True]:
            variants.append(dict(name=group + ("_repeat" if repeat else "_baseline"),
                reference_group=group, seed=1042, reference=not repeat, restore_tracker=tracker,
                independent_sr=independent, stochastic_rounding=stochastic,
                require_repeat=False))
    jobs = [("p2-rng-controls8", dict(replay, replay_positions=probe_positions, replay_cases=variants)),
            ("p2-rng-deterministic8", dict(replay, replay_positions=probe_positions, deterministic_math=True,
                replay_cases=[dict(c, require_repeat=c['name'].endswith('_repeat') and c['reference_group'] in ('isolated','deterministic'))
                              for c in variants if c['reference_group']!='legacy'])),
            ("p2-rng-tis64", dict(replay, deterministic_math=True))]
    for job_id, config in jobs:
        directory = root / "jobs" / job_id
        directory.mkdir()
        config.update(job_id=job_id, host_run_dir=str(directory), container_run_dir="/diagnostic/jobs/" + job_id)
        source = prior / "jobs/nv75-frozen-tis/mismatch.yaml"
        mismatch = json.loads(source.read_text()); mismatch['nvfp4_sim_reference_arithmetic'] = False
        (directory / "mismatch.yaml").write_text(json.dumps(mismatch, indent=2) + "\n")
        argv = config["megatron_argv"][:]
        argv[argv.index("--custom-config-path") + 1] = "/diagnostic/jobs/" + job_id + "/mismatch.yaml"
        config["megatron_argv"] = argv
        save(directory / "job.json", config)
        entry = dict(job_id=job_id, kind="gradient_replay", script="/quant/examples/nvfp4_sim/replay_diagnostic.py")
        if job_id.endswith("tis64"):
            entry["export_dependency"] = "jobs/p2-rng-deterministic8/gradient-comparison.json"
        plan["jobs"].append(entry)
    candidates = []
    for suffix, precision, overrides in [
        ("weight64", "nvfp4", {"nvfp4_sim_quantize_activations": False}),
        ("activation64", "nvfp4", {"nvfp4_sim_quantize_weights": False}),
        ("headbf16-64", "nvfp4", {"nvfp4_sim_include_output_layer": False}),
        ("full64", "nvfp4", {}), ("bf16-64", "bf16", {}),
        ("reference64", "nvfp4", {"nvfp4_sim_reference_arithmetic": True}),
        ("reference512", "nvfp4", {"nvfp4_sim_reference_arithmetic": True}),
    ]:
        job_id = "p2-" + suffix
        directory = root / "jobs" / job_id
        directory.mkdir()
        config = dict(base, job_id=job_id, precision=precision, model_overrides=overrides,
                      output="/diagnostic/jobs/" + job_id, dataset_indices=list(range(0, 512, 8)),
                      allow_pending_model=False)
        config["model_overrides"] = dict({"nvfp4_sim_reference_arithmetic": False}, **overrides)
        if suffix == "reference512":
            config.pop("dataset_indices")
        for key in ["physical_gpu", "expected_gpu_uuid", "reservation_host_pids"]:
            config.pop(key, None)
        save(directory / "job.json", config)
        entry = dict(job_id=job_id, kind="evaluation", script="/quant/examples/nvfp4_sim/evaluate_diagnostic.py")
        if suffix == "reference512":
            entry["export_dependency"] = "jobs/p2-reference64/evaluation.json"
        plan["jobs"].append(entry)
        if suffix in ("weight64", "activation64", "headbf16-64"):
            candidates.append(job_id)
    plan["screen_promotion"] = dict(candidate_jobs=candidates, confirmation_job_id="p2-confirm512")
    for component in ["megatron", "sglang"]:
        job_id = "p2-align-" + component
        directory = root / "jobs" / job_id
        directory.mkdir()
        config = dict(replay, **{k: v for k, v in base.items() if k != "job_id"})
        config.update(job_id=job_id, component=component, replay_positions=[0, 1, 8, 9],
            output="/diagnostic/jobs/" + job_id, container_run_dir="/diagnostic/jobs/" + job_id,
            model="/prior-diagnostic/hf/nvfp4-step75", precision="nvfp4", model_overrides={},
            megatron_result="/diagnostic/jobs/p2-align-megatron/megatron-alignment.json")
        mismatch = json.loads((prior / "jobs/nv75-frozen-tis/mismatch.yaml").read_text())
        mismatch['nvfp4_sim_reference_arithmetic'] = False
        (directory / "mismatch.yaml").write_text(json.dumps(mismatch, indent=2) + "\n")
        config["model_overrides"] = {"nvfp4_sim_reference_arithmetic": False}
        argv = config["megatron_argv"][:]
        argv[argv.index("--custom-config-path") + 1] = "/diagnostic/jobs/" + job_id + "/mismatch.yaml"
        config["megatron_argv"] = argv
        save(directory / "job.json", config)
        entry = dict(job_id=job_id, kind="alignment", script="/quant/examples/nvfp4_sim/alignment_diagnostic.py")
        if component == "sglang":
            entry["export_dependency"] = "jobs/p2-align-megatron/megatron-alignment.json"
        plan["jobs"].append(entry)
    save(root / "plan.json", plan)
    for job in plan["jobs"]:
        flag = "--check-only" if job["kind"] == "gradient_replay" else "--preflight"
        with (root / "jobs" / job["job_id"] / "preflight.log").open("w") as log:
            subprocess.run(["docker", "exec", name, "python3", job["script"],
                "/diagnostic/jobs/" + job["job_id"] + "/job.json", flag], stdout=log, stderr=subprocess.STDOUT, check=True)
        print("PREFLIGHT_PASSED", job["job_id"], flush=True)
    save(root / "source-version.json", {"git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "prepared_at": time.time(),
        "note": "Code hashes and commit will be recorded again after validation, before launching."})
    print("READY", root, flush=True)


if __name__ == "__main__":
    main()
