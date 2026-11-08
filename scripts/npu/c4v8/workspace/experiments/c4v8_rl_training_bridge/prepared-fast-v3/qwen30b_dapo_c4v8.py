"""Isolated C4V8 forward-QDQ/STE DAPO GRPO recipe; launch gates required."""
import argparse
from datetime import datetime, timezone
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo
from recipe_support import prepare_run_context, register_driver_backend, verify_resolved_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/root/hby")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--raw-rollout-backup-dir", default=None)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-name", default="aime2024")
    parser.add_argument("--sglang-max-running-requests", type=int, default=16)
    parser.add_argument("--sglang-chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--parent-completed-optimizer-steps", type=int, default=0)
    parser.add_argument("--parent-run-manifest", default=None)
    parser.add_argument("--lineage-target-optimizer-steps", type=int, default=None)
    parser.add_argument("--validation-trajectory-dir", default=None)
    parser.add_argument("--offline-parent-run-manifest", default=None,
                        help="Continue only source77c0d7 at40 with explicit model/evidence lineage")
    parser.add_argument("--offline-checkpoint-validation", action="store_true",
                        help="Evaluate saved HF checkpoints after training; disable all native evaluation")
    parser.add_argument("--keep-all-weight-versions", action="store_true")
    parser.add_argument("--hf-checkpoint-every-steps", type=int, default=0,
                        help="Save independent HF-only snapshots every N successful optimizer updates")
    parser.add_argument("--sglang-max-total-tokens", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--prompts", type=int, default=5)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument("--response-length", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--sglang-context-length", type=int, default=8192)
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.13)
    parser.add_argument("--offload-fraction", type=float, default=0.33)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validate-arguments", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    options, extra = parser.parse_known_args()
    if options.hf_checkpoint_every_steps < 0:
        parser.error("--hf-checkpoint-every-steps cannot be negative")
    sglang_logprobs_environment = {
        "SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK": "1",
        "SGLANG_LOGITS_PROCESSER_CHUNK_SIZE": "64",
    }
    # The job runtime environment is inherited by ServingCluster and SG workers.
    os.environ.update(sglang_logprobs_environment)
    # SG uses a private MemPool for discardable weights/KV; trainers retain VMM.
    sglang_allocator_config = "expandable_segments:False"
    os.environ["SLIME_SGLANG_NPU_ALLOC_CONF"] = sglang_allocator_config
    os.environ["MAX_COMPILE_CORE_NUMBER"] = "1"
    if "--full-checkpoint" in extra:
        parser.error("This first-version launcher exports complete HF weights and does not claim optimizer resume")
    workspace = Path(options.workspace).resolve()
    c4v8_environment, c4v8_audit = prepare_run_context(options, extra, __file__)
    repo = workspace / "slime"
    model = Path(options.model) if options.model else workspace / "models/Qwen3-30B-A3B"
    data = Path(options.data).resolve() if options.data else workspace / "data/dapo-math-17k/train.jsonl"
    evaluation = Path(options.eval_data).resolve() if options.eval_data else workspace / "data/aime-2024/eval.jsonl"
    if not (model / "config.json").is_file() or not data.is_file() or not evaluation.is_file():
        raise SystemExit("The original BF16 model and prepared DAPO train/AIME evaluation files must exist")
    with data.open() as source:
        train_questions = sum(bool(line.strip()) for line in source)
    with evaluation.open() as source:
        eval_questions = sum(bool(line.strip()) for line in source)
    if options.prompts * options.samples % options.minibatch:
        raise SystemExit("Samples per rollout must divide into complete optimizer minibatches")
    steps_per_rollout = options.prompts * options.samples // options.minibatch
    target_steps = options.rollouts * steps_per_rollout
    parent_steps = options.parent_completed_optimizer_steps
    if parent_steps < 0 or (parent_steps and (not options.model or not options.parent_run_manifest)):
        parser.error("A continuation requires explicit parent HF model, manifest, and nonnegative completed steps")
    if parent_steps and os.environ.get("SLIME_NPU_REPLAY0_BACKUP"):
        parser.error("Base-policy rollout replay cannot be reused after loading a trained parent policy")
    lineage_target_steps = parent_steps + target_steps
    if options.lineage_target_optimizer_steps is not None and options.lineage_target_optimizer_steps != lineage_target_steps:
        parser.error("Parent completed updates plus new updates must equal the explicit lineage target")
    expected_checkpoint_steps = [step for step in range(parent_steps + 1, lineage_target_steps + 1)
                                 if options.hf_checkpoint_every_steps and step % options.hf_checkpoint_every_steps == 0]
    validation_directory = Path(options.validation_trajectory_dir).resolve() if options.validation_trajectory_dir else None
    offline_lineage = None
    offline_parent = None
    if options.offline_parent_run_manifest:
        if (not options.offline_checkpoint_validation or parent_steps != 40 or options.rollouts != 3
                or target_steps != 60 or options.hf_checkpoint_every_steps != 5
                or options.prompts != 5 or options.samples != 16 or options.minibatch != 4
                or options.seed != 1234 or options.offload_fraction != 0.35
                or options.learning_rate != 1e-6 or options.sequence_length != 8192
                or lineage_target_steps != 100
                or Path(options.parent_run_manifest).resolve() != Path(options.offline_parent_run_manifest).resolve()):
            parser.error("The audited offline continuation is exactly source40 plus3x20 new updates with unchanged training math")
        helper_path = workspace / "setup/base100_every5_resume_lineage.py"
        specification = importlib.util.spec_from_file_location("slime_offline_resume_lineage", helper_path)
        offline_lineage = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(offline_lineage)
        offline_parent = offline_lineage.validate_parent_source(
            options.offline_parent_run_manifest, model, workspace=workspace, trajectory=validation_directory)
        if data != workspace / "data/dapo-math-17k/train.jsonl" or evaluation != workspace / "data/aime-2024/eval.jsonl":
            parser.error("The audited continuation must retain the original train/evaluation datasets")
    if options.offline_checkpoint_validation and (validation_directory is None
            or (parent_steps != 0 and offline_parent is None)
            or options.hf_checkpoint_every_steps < 1 or target_steps % options.hf_checkpoint_every_steps):
        parser.error("Offline validation requires a new base run or the explicitly audited source40 continuation")
    if validation_directory is not None and (steps_per_rollout != 20 or options.eval_name != "aime2024" or eval_questions != 30 or options.response_length != 4096):
        parser.error("The fixed AIME trajectory requires all 30 questions, 4096 response tokens, and 20 updates per rollout")
    if parent_steps:
        parent_index = model / "model.safetensors.index.json"
        if not parent_index.is_file():
            parser.error("The trained parent model must have a complete HF safetensors index")
        indexed_files = set(json.loads(parent_index.read_text())["weight_map"].values())
        if not indexed_files or any(not (model / name).is_file() for name in indexed_files):
            parser.error("The parent HF model is missing indexed weight shards")
        if not Path(options.parent_run_manifest).is_file():
            parser.error("The parent run manifest must exist")
    keep_all_versions = options.keep_all_weight_versions or validation_directory is not None
    version_environment = {"SLIME_NPU_KEEP_ALL_WEIGHT_VERSIONS": "1"} if keep_all_versions else {}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_id = options.version + "_" + stamp + "_" + uuid.uuid4().hex[:6]
    output = workspace / "runs/qwen3-30b-a3b-dapo-math-17k" / run_id
    output.mkdir(parents=True)
    c4v8_environment["HBY_C4V8_ROLLOUT_AUDIT_DIR"] = str(output / "backend-audit")
    c4v8_environment["HBY_C4V8_TRAIN_AUDIT_DIR"] = str(output / "trainer-backend-audit")
    os.environ.update(c4v8_environment)
    published = workspace / "models/finetunes" / ("Qwen3-30B-A3B-DAPO-Math-17k-" + run_id)
    published.parent.mkdir(parents=True, exist_ok=True)
    ray_tmp = workspace / ".r" / uuid.uuid4().hex[:8]
    ray_tmp.mkdir(parents=True)
    raw_backup_directory = Path(options.raw_rollout_backup_dir).resolve() if options.raw_rollout_backup_dir else output / "accepted-rollouts"
    replay_environment = {}
    replay_source_steps = 0
    if os.environ.get("SLIME_NPU_REPLAY0_BACKUP"):
        for name in ("SLIME_NPU_REPLAY0_BACKUP", "SLIME_NPU_REPLAY0_SOURCE_MANIFEST",
                     "SLIME_NPU_REPLAY0_ALLOW_CURSOR_RESTART", "SLIME_NPU_REPLAY0_SHA256"):
            value = os.environ.get(name)
            if not value:
                raise SystemExit(f"Single-use rollout replay requires {name}")
            replay_environment[name] = value
        replay_environment["SLIME_NPU_REPLAY0_AUDIT"] = str(output / "replay0-audit.json")
        os.environ.update(replay_environment)
        setup_path = str(workspace / "setup")
        sys.path.insert(0, setup_path)
        os.environ["PYTHONPATH"] = setup_path + os.pathsep + os.environ.get("PYTHONPATH", "")
        replay_environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
        replay_source_steps = len(list(Path(replay_environment["SLIME_NPU_REPLAY0_SOURCE_MANIFEST"]).parent.glob("grad_norm_actor_*.pt")))
    config = output / "sglang-loopback.json"
    config.write_text(json.dumps({"sglang": [{
        "name": "actor", "update_weights": True, "num_gpus_per_engine": 8,
        "server_groups": [{"worker_type": "regular", "num_gpus": 8,
                           "overrides": {"host": "127.0.0.1", "enable_weights_cpu_backup": False}}],
    }]}, indent=2) + "\n")
    model_args = subprocess.check_output([
        "bash", "-c", 'source "$1"; printf "%s\n" "${MODEL_ARGS[@]}"',
        "_", str(repo / "scripts/models/qwen3-30B-A3B.sh"),
    ]).decode().splitlines()
    # This pinned MindSpeed fused-permute wrapper uses the older dispatcher
    # constructor API. Native permutation preserves the mathematical operation.
    model_args = [flag for flag in model_args if flag != "--moe-permute-fusion"]
    arguments = model_args + [
        "--seed", str(options.seed), "--rollout-seed", str(options.seed), "--bf16", "--hf-checkpoint", str(model), "--load", str(model),
        "--ref-load", str(model),
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "8",
        "--num-gpus-per-node", "8", "--rollout-num-gpus", "8",
        "--rollout-num-gpus-per-engine", "8", "--colocate",
        "--no-offload-train", "--offload-rollout",
        "--disable-actor-cpu-backup",
        "--tensor-model-parallel-size", "4", "--pipeline-model-parallel-size", "2",
        "--expert-model-parallel-size", "4", "--expert-tensor-parallel-size", "1",
        "--context-parallel-size", "1", "--sequence-parallel", "--use-tp-pp-dp-mapping",
        "--no-initialization", "--accumulate-allreduce-grads-in-fp32",
        "--optimizer", "adam", "--optimizer-cpu-offload",
        "--optimizer-offload-fraction", str(options.offload_fraction),
        "--use-precision-aware-optimizer",
        "--no-pin-cpu-params", "--no-pin-cpu-grads",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--train-env-vars", json.dumps({"HCCL_IF_IP": os.environ["HCCL_IF_IP"], "HCCL_IF_BASE_PORT": "31100", "HCCL_HOST_SOCKET_PORT_RANGE": "auto", "HCCL_NPU_SOCKET_PORT_RANGE": "auto", "TENSORBOARD_DIR": str(output / "tensorboard"), "TORCHINDUCTOR_COMPILE_THREADS": "1", "TORCH_WARM_POOL": "0", "TE_PARALLEL_COMPILER": "2", "MAX_COMPILE_CORE_NUMBER": "1",
                       "SLIME_NPU_PREALLOCATE_ADAM": os.environ.get("SLIME_NPU_PREALLOCATE_ADAM", "0"),
                       "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"),
                       "HCCL_BUFFSIZE": os.environ.get("HCCL_BUFFSIZE", "128"),
                       "SLIME_NPU_TRIM_UNUSED_CPU_HEAP": os.environ.get("SLIME_NPU_TRIM_UNUSED_CPU_HEAP", "1"),
                       "SLIME_NPU_PREALLOCATE_ADAM_HELPER": str(workspace / "setup/preallocate-hybrid-adam.py"),
                       "SLIME_NPU_PREALLOCATE_REPORT_DIR": str(output / "optimizer-preallocation")} | version_environment | c4v8_environment),
        "--prompt-data", str(data), "--input-key", "messages", "--label-key", "label",
        "--apply-chat-template", "--apply-chat-template-kwargs", '{"enable_thinking": false}',
        "--rm-type", "math", "--advantage-estimator", "grpo",
        "--kl-coef", "0", "--entropy-coef", "0",
        "--num-rollout", str(options.rollouts), "--rollout-batch-size", str(options.prompts),
        "--n-samples-per-prompt", str(options.samples),
        "--over-sampling-batch-size", "1",
        "--global-batch-size", str(options.minibatch), "--micro-batch-size", "1",
        "--num-steps-per-rollout", str(steps_per_rollout),
        "--rollout-max-response-len", str(options.response_length),
        "--rollout-temperature", "1.0", "--rollout-top-p", "1.0", "--rollout-top-k", "-1",
        "--dynamic-sampling-filter-path",
        "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        "--lr", str(options.learning_rate), "--lr-decay-style", "constant", "--weight-decay", "0",
        "--attention-dropout", "0", "--hidden-dropout", "0",
        "--attention-softmax-in-fp32", "--attention-backend", "flash",
        "--seq-length", str(options.sequence_length), "--max-position-embeddings", "40960",
        "--sglang-config", str(config), "--router-prometheus-host", "127.0.0.1",
        "--sglang-device", "npu", "--sglang-attention-backend", "hby_c4v8_sim",
        "--sglang-prefill-attention-backend", "hby_c4v8_sim",
        "--sglang-decode-attention-backend", "hby_c4v8_sim",
        "--sglang-dtype", "bfloat16", "--sglang-kv-cache-dtype", "auto",
        "--sglang-python-executable", str(workspace / "experiments/c4v8_rollout_bridge/python-c4v8"),
        "--spec", "megatron_spec", "get_spec",
        "--sglang-sampling-backend", "pytorch", "--sglang-disable-cuda-graph",
        "--sglang-context-length", str(options.sglang_context_length),
        "--sglang-max-running-requests", str(options.sglang_max_running_requests), "--sglang-mem-fraction-static", str(options.sglang_mem_fraction),
        "--sglang-chunked-prefill-size", str(options.sglang_chunked_prefill_size),
        "--update-weight-mode", "full", "--update-weight-transport", "disk",
        "--update-weight-disk-dir", str(output / "weight-updates"), "--update-weight-disk-keep-files",
        "--update-weight-buffer-size", str(128 * 1024**2),
        "--custom-update-weight-post-write-path", "slime_plugins.npu_dapo.prune_old_weight_versions",
        "--rollout-data-transport", "object-store",
        "--ci-save-grad-norm", str(output / "grad_norm_{role}_{rollout_id}_{step_id}.pt"),
        "--eval-prompt-data", options.eval_name, str(evaluation),
        "--eval-input-key", "messages", "--eval-label-key", "label",
        "--n-samples-per-eval-prompt", "1",
        "--eval-max-response-len", str(options.response_length), "--eval-temperature", "0",
        "--use-tensorboard", "--tensorboard-dir", str(output / "tensorboard"),
    ]
    if options.offline_checkpoint_validation:
        arguments += ["--skip-eval-before-train"]
    else:
        arguments += ["--eval-interval", "1" if validation_directory is not None else str(options.rollouts)]
    if options.sglang_max_total_tokens is not None:
        arguments += ["--sglang-max-total-tokens", str(options.sglang_max_total_tokens)]
    if validation_directory is not None and not options.offline_checkpoint_validation:
        arguments += ["--custom-eval-rollout-log-function-path", "slime_plugins.npu_validation_trajectory.log_eval"]
    if extra[:1] == ["--"]:
        extra = extra[1:]
    arguments += extra
    if replay_environment:
        arguments += ["--rollout-function-path", "qwen30b_replay_once.generate_rollout",
                      "--eval-function-path", "slime.rollout.sglang_rollout.generate_rollout"]
    invocation = [str(repo / "train.py")] + arguments
    manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_driver_pid": os.getpid(),
        "ray_temporary_directory": str(ray_tmp), "ray_namespace": "qwen30b-" + run_id,
        "directory_timezone": "Asia/Shanghai", "initial_model": str(model),
        "train_data": str(data), "heldout_data": str(evaluation),
        "training_dataset": "DAPO-Math-17k", "train_questions": train_questions,
        "baseline_evaluation_skipped": "--skip-eval-before-train" in invocation,
        "response_length_limit": options.response_length, "enable_thinking": False,
        "pre_abort_raw_rollout_backup": str(raw_backup_directory),
        "optimizer_state_preallocation_requested": os.environ.get("SLIME_NPU_PREALLOCATE_ADAM") == "1",
        "optimizer_preallocation_reports": str(output / "optimizer-preallocation"),
        "npu_allocator_config": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
        "hccl_buffer_size_mb": os.environ.get("HCCL_BUFFSIZE"),
        "cpu_unused_heap_trim_requested": os.environ.get("SLIME_NPU_TRIM_UNUSED_CPU_HEAP") == "1",
        "rollout_source_order": "shuffled" if "--rollout-shuffle" in invocation else "fixed",
        "rollout_seed": options.seed, "over_sampling_batch_size": 1,
        "sglang_max_running_requests": options.sglang_max_running_requests,
        "sglang_chunked_prefill_size": options.sglang_chunked_prefill_size,
        "training_sequence_length": options.sequence_length,
        "sglang_context_length": options.sglang_context_length,
        "sglang_mem_fraction_static": options.sglang_mem_fraction,
        "sglang_max_total_tokens": options.sglang_max_total_tokens,
        "sglang_input_logprobs_chunk_tokens": 64,
        "sglang_worker_environment": sglang_logprobs_environment | {
            "PYTORCH_NPU_ALLOC_CONF": sglang_allocator_config, "MAX_COMPILE_CORE_NUMBER": "1"} | c4v8_environment,
        "rollout_offload": True, "train_offload": False,
        "sglang_weights_cpu_backup": False,
        "rollout_memory_policy": "tagged NPU weights/KV discard; disk reload before generation",
        "evaluation": {"name": options.eval_name, "questions": eval_questions, "temperature": 0, "mode": "greedy", "purpose": "post-training sequential saved-checkpoint audit; not a benchmark" if options.offline_checkpoint_validation else ("held-out validation trajectory; not a benchmark" if validation_directory else "post-training audit; not a benchmark")},
        "initial_policy_evaluation_step": parent_steps if validation_directory and not options.offline_checkpoint_validation else None,
        "initial_policy_evaluation_skipped": "--skip-eval-before-train" in invocation,
        "baseline_zero_evaluation": "post-training evaluation of unchanged official base in this new trajectory" if options.offline_checkpoint_validation else ("independent original model point in shared trajectory" if parent_steps and validation_directory else None),
        "synthetic_rewards": False, "reward": "standard math/boxed-answer matching",
        "precision": "BF16 backbone, FA NVFP4/MXFP8 forward QDQ, high-precision STE backward, FP32 Adam",
        "attention_simulation": c4v8_audit,
        "parallel": {"train_tp": 4, "train_pp": 2, "train_ep": 4, "train_etp": 1, "rollout_tp": 8},
        "optimizer_cpu_offload_fraction": options.offload_fraction,
        "actor_cpu_snapshot": False, "reference_model": False, "kl_coefficient": 0,
        "ray_scheduling_cpus": 16, "ray_worker_prestart": False,
        "ray_memory_usage_threshold": 0.98,
        "cann_kb_bank_workers_per_process": 1,
        "random_weight_initialization": False, "parameter_construction_device": "npu",
        "target_optimizer_steps": lineage_target_steps,
        "target_new_optimizer_steps": target_steps,
        "parent_completed_optimizer_steps": parent_steps,
        "parent_run_manifest": str(Path(options.parent_run_manifest).resolve()) if options.parent_run_manifest else None,
        "optimizer_state_restored": False,
        "source_cursor_restarted": bool(parent_steps), "exact_source_cursor_restored": False,
        "validation_trajectory_directory": str(validation_directory) if validation_directory else None,
        "validation_optimizer_step_offset": parent_steps if validation_directory and not options.offline_checkpoint_validation else None,
        "offline_checkpoint_validation": options.offline_checkpoint_validation,
        "validation_execution": "post_training_sequential_http" if options.offline_checkpoint_validation else "native_rollout_boundary",
        "validation_expected_global_steps": [0] + expected_checkpoint_steps if options.offline_checkpoint_validation else None,
        "validation_checkpoint_model_paths": ({"0": str(model)} | {
            str(step): str(output / "optimizer-step-checkpoints" / f"checkpoint_step_{step:06d}")
            for step in expected_checkpoint_steps}) if options.offline_checkpoint_validation else None,
        "baseline_zero_is_new_in_this_trajectory": options.offline_checkpoint_validation,
        "keep_all_weight_versions": keep_all_versions,
        "optimizer_step_hf_checkpoint_interval": options.hf_checkpoint_every_steps,
        "optimizer_step_hf_checkpoint_directory": str(output / "optimizer-step-checkpoints"),
        "optimizer_step_hf_checkpoint_format": "HF weights only; Adam/RNG/source not saved",
        "optimizer_step_hf_expected_global_steps": expected_checkpoint_steps,
        "optimizer_checkpoint_requested": False, "seed": options.seed,
        "optimizer_resume_available": False, "pin_cpu_params": False, "pin_cpu_grads": False,
        "hf_model_destination": str(published), "status": "prepared", "argv": invocation,
    }
    if offline_parent is not None:
        parent_report = output / "provenance/offline-parent40-validation.json"
        offline_lineage.atomic_json(parent_report, offline_parent)
        manifest.update(offline_lineage.continuation_fields(offline_parent, output))
        manifest.update(
            offline_parent_validation_report=str(parent_report),
            offline_parent_validation_report_sha256=hashlib.sha256(parent_report.read_bytes()).hexdigest(),
            offline_parent_lineage_helper=str(helper_path),
            offline_parent_lineage_helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
            sampling_execution_before40={"sglang_chunked_prefill_size": 1024,
                "sglang_max_running_requests": 16, "sglang_max_total_tokens": 20096,
                "sglang_mem_fraction_static": 0.135},
            sampling_execution_after40={"sglang_chunked_prefill_size": options.sglang_chunked_prefill_size,
                "sglang_max_running_requests": options.sglang_max_running_requests,
                "sglang_max_total_tokens": options.sglang_max_total_tokens,
                "sglang_mem_fraction_static": options.sglang_mem_fraction})
    if replay_environment:
        manifest.update(
            replay0={key: value for key, value in replay_environment.items() if key != "PYTHONPATH"},
            source_cursor_restarted=True, exact_source_cursor_restored=False,
            optimizer_state_restored=False,
            replayed_source_unsaved_steps_ignored=replay_source_steps,
            fresh_target_optimizer_steps=target_steps,
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("QWEN30B_RUN_DIR=" + str(output), flush=True)
    print("QWEN30B_HF_MODEL=" + str(published), flush=True)
    if options.print_command:
        print(json.dumps(invocation, indent=2))
        return
    import psutil
    if not options.validate_arguments and psutil.virtual_memory().available < 215 * 1024**3:
        raise RuntimeError("This full-parameter recipe needs at least 215 GiB available host RAM")
    import ray
    import torch
    import torch_npu
    register_driver_backend()
    from slime.utils.arguments import parse_args
    from slime.observability.logging_utils import configure_logger
    from train import main as train_main
    sys.argv = invocation
    args, restore_plan = parse_args(return_restore_plan=True)
    verify_resolved_config(args, options, c4v8_environment)
    args.optimizer_step_hf_save_interval = options.hf_checkpoint_every_steps
    args.optimizer_step_hf_save_dir = str(output / "optimizer-step-checkpoints")
    args.optimizer_step_hf_completed_offset = parent_steps
    if validation_directory is not None:
        if options.offline_checkpoint_validation:
            if args.eval_interval is not None or not args.skip_eval_before_train or args.custom_eval_rollout_log_function_path is not None:
                raise ValueError("Offline checkpoint validation must disable native interval/initial evaluation and custom eval logging")
        elif args.eval_interval != 1:
            raise ValueError("AIME trajectory must evaluate every completed rollout")
        args.validation_trajectory_dir = str(validation_directory)
        args.validation_optimizer_step_offset = None if options.offline_checkpoint_validation else parent_steps
        args.validation_dataset_sha256 = hashlib.sha256(evaluation.read_bytes()).hexdigest()
        args.validation_protocol_helper = str(Path(__file__).with_name("c4v8_validation_protocol.py"))
        args.validation_protocol_helper_sha256 = hashlib.sha256(Path(args.validation_protocol_helper).read_bytes()).hexdigest()
        manifest["validation_protocol_helper"] = args.validation_protocol_helper
        manifest["validation_protocol_helper_sha256"] = args.validation_protocol_helper_sha256
        args.validation_model_identity = {
            "run_id": run_id, "parent_hf_checkpoint": str(model.resolve()),
            "parent_completed_optimizer_steps": parent_steps,
            "parent_run_manifest": manifest["parent_run_manifest"],
            "optimizer_state_restored": False,
        }
        from slime_plugins.npu_validation_trajectory import validate_native_protocol
        _, _, validation_protocol, validation_protocol_sha = validate_native_protocol(args)
        manifest["validation_protocol"] = validation_protocol
        manifest["validation_protocol_sha256"] = validation_protocol_sha
    if offline_parent is not None:
        if (args.hf_checkpoint != str(model) or args.load != str(model)
                or args.num_rollout != 3 or args.num_steps_per_rollout != 20
                or args.global_batch_size != 4 or args.rollout_batch_size != 5
                or args.n_samples_per_prompt != 16 or args.optimizer_offload_fraction != 0.35
                or not args.bf16 or args.lr != 1e-6 or args.lr_decay_style != "constant"
                or args.seq_length != 8192 or args.rollout_temperature != 1.0
                or args.rollout_top_p != 1.0 or args.rollout_top_k != -1):
            raise ValueError("Resolved continuation arguments changed parent identity or update math")
        manifest["sampling_execution_after40"] = {
            "sglang_chunked_prefill_size": args.sglang_chunked_prefill_size,
            "sglang_max_running_requests": args.sglang_max_running_requests,
            "sglang_max_total_tokens": args.sglang_max_total_tokens,
            "sglang_mem_fraction_static": args.sglang_mem_fraction_static}
    manifest["resolved_eval_interval"] = args.eval_interval
    manifest["resolved_custom_eval_log_function"] = args.custom_eval_rollout_log_function_path
    manifest["resolved_skip_eval_before_train"] = args.skip_eval_before_train
    manifest["rollout_source_order"] = "shuffled" if args.rollout_shuffle else "fixed"
    manifest["rollout_seed"] = args.rollout_seed
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.offload_train or not args.offload_rollout or not args.disable_actor_cpu_backup:
        raise RuntimeError("This memory plan requires resident training, SG memory-saver offload, and no actor CPU duplicate")
    if not args.use_precision_aware_optimizer or not args.optimizer_cpu_offload:
        raise RuntimeError("This memory plan requires precision-aware Hybrid CPU Adam")
    if args.dist_ckpt_optim_fully_reshardable or args.save_debug_rollout_data is not None:
        raise RuntimeError("Whole-model CPU aggregation is outside this recipe's host memory budget")
    if args.custom_rm_path is not None or args.rm_type != "math":
        raise RuntimeError("This launcher requires real math rewards")
    if not args.use_distributed_optimizer or args.overlap_cpu_optimizer_d2h_h2d:
        raise RuntimeError("This recipe requires distributed Hybrid Adam with no CPU transfer overlap")
    if options.validate_arguments:
        keys = [
            "colocate", "offload_train", "offload_rollout", "disable_actor_cpu_backup",
            "use_cpu_initialization", "perform_initialization", "pin_cpu_params", "pin_cpu_grads",
            "rollout_seed", "rollout_shuffle", "over_sampling_batch_size", "rollout_batch_size", "n_samples_per_prompt", "num_rollout", "sglang_max_running_requests",
            "use_distributed_optimizer", "optimizer_cpu_offload", "optimizer_offload_fraction",
            "use_precision_aware_optimizer", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype",
            "overlap_cpu_optimizer_d2h_h2d", "tensor_model_parallel_size",
            "pipeline_model_parallel_size", "expert_model_parallel_size", "expert_tensor_parallel_size",
            "sequence_parallel", "use_tp_pp_dp_mapping", "global_batch_size", "num_steps_per_rollout",
            "custom_rm_path", "rm_type", "prompt_data", "seq_length", "sglang_context_length", "sglang_mem_fraction_static", "save", "save_debug_rollout_data",
            "dist_ckpt_optim_fully_reshardable", "sglang_tp_size", "train_env_vars",
            "hf_checkpoint", "load", "eval_interval", "skip_eval_before_train", "n_samples_per_eval_prompt",
            "custom_eval_rollout_log_function_path", "validation_trajectory_dir",
            "validation_optimizer_step_offset", "sglang_max_total_tokens", "sglang_chunked_prefill_size",
            "optimizer_step_hf_save_interval", "optimizer_step_hf_save_dir",
            "optimizer_step_hf_completed_offset",
        ]
        resolved = {key: str(getattr(args, key, None)) for key in keys}
        (output / "resolved-arguments.json").write_text(json.dumps(resolved, indent=2) + "\n")
        print("QWEN30B_ARGUMENTS_VALIDATED=" + json.dumps(resolved), flush=True)
        return
    if torch.npu.device_count() != 8:
        raise RuntimeError("Expose all eight NPUs before starting this recipe")
    os.environ["SLIME_NPU_PRE_ABORT_ROLLOUT_BACKUP_DIR"] = str(raw_backup_directory)
    os.environ["TENSORBOARD_DIR"] = str(output / "tensorboard")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["SLIME_HOST_IP"] = "127.0.0.1"
    configure_logger()
    try:
        ray.init(address="local", namespace="qwen30b-" + run_id,
                 num_cpus=16, resources={"NPU": 8},
                 runtime_env={"env_vars": {"SLIME_NPU_PRE_ABORT_ROLLOUT_BACKUP_DIR": str(raw_backup_directory)}
                              | sglang_logprobs_environment
                              | {"SLIME_SGLANG_NPU_ALLOC_CONF": sglang_allocator_config,
                                 "MAX_COMPILE_CORE_NUMBER": "1"} | replay_environment | version_environment | c4v8_environment},
                 _system_config={"prestart_worker_first_driver": False, "enable_worker_prestart": False,
                                 "memory_usage_threshold": 0.98},
                 object_store_memory=2 * 1024**3, include_dashboard=False,
                 _node_ip_address="127.0.0.1", _temp_dir=str(ray_tmp))
        manifest["status"] = "running"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        train_main(args, restore_plan)
        norms = [float(torch.load(path, map_location="cpu", weights_only=True))
                 for path in sorted(output.glob("grad_norm_actor_*.pt"),
                                    key=lambda path: tuple(int(item) for item in path.stem.rsplit("_", 2)[1:]))]
        if len(norms) != target_steps or any(not math.isfinite(norm) or norm <= 0 for norm in norms):
            raise RuntimeError(f"Expected {target_steps} real positive-gradient steps, got {norms}")
        checkpoint_paths = []
        for global_step in expected_checkpoint_steps:
            checkpoint = output / "optimizer-step-checkpoints" / f"checkpoint_step_{global_step:06d}"
            metadata = json.loads((checkpoint / "optimizer-step-checkpoint.json").read_text())
            checkpoint_index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
            shards = set(checkpoint_index["weight_map"].values())
            if (metadata.get("completed_optimizer_steps") != global_step
                    or metadata.get("complete") is not True or metadata.get("hf_only") is not True
                    or not shards or any(not (checkpoint / name).is_file() for name in shards)):
                raise RuntimeError(f"Missing complete optimizer-step HF snapshot: {checkpoint}")
            checkpoint_paths.append(str(checkpoint))
        manifest["optimizer_step_hf_checkpoints"] = checkpoint_paths
        if offline_parent is not None:
            manifest.update(offline_lineage.write_lineage_gradient_report(
                manifest, output, norms, workspace=workspace))
        final_weights = output / "weight-updates" / f"weight_v{options.rollouts + 1:06d}"
        index_path = final_weights / "model.safetensors.index.json"
        if not (final_weights / "config.json").is_file() or not index_path.is_file():
            raise RuntimeError("The trained full HF checkpoint is missing its config or safetensors index")
        indexed_shards = set(json.loads(index_path.read_text())["weight_map"].values())
        if not indexed_shards or any(not (final_weights / name).is_file() for name in indexed_shards):
            raise RuntimeError("The trained full HF checkpoint is missing indexed shards")
        if published.exists():
            raise FileExistsError(published)
        if keep_all_versions:
            # Keep the audited checkpoint path valid; publish immutable hardlinks
            # on the same filesystem, with ordinary copies across filesystems.
            temporary = published.with_name(published.name + ".partial")
            def link_or_copy(source, destination):
                try:
                    return os.link(source, destination)
                except OSError as error:
                    if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                        raise
                    return shutil.copy2(source, destination)
            shutil.copytree(final_weights, temporary, copy_function=link_or_copy)
            os.replace(temporary, published)
        else:
            try:
                os.replace(final_weights, published)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                temporary = published.with_name(published.name + ".partial")
                shutil.copytree(final_weights, temporary)
                os.replace(temporary, published)
        manifest.update(status="completed", optimizer_steps=parent_steps + len(norms),
                        new_optimizer_steps=len(norms), gradient_norms=norms,
                        gradient_norm_order="rollout_id,step_id numeric",
                        optimizer_resume_available=False,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("QWEN30B_TRAINED_HF_MODEL=" + str(published), flush=True)
    except BaseException as error:
        # Keep the original failure and record only artifacts that actually exist.
        # A complete HF file does not imply Adam/RNG/source state was saved.
        try:
            count = len(list(output.glob("grad_norm_actor_*.pt")))
            completed_hf = []
            for index in sorted((output / "weight-updates").glob("weight_v*/model.safetensors.index.json")):
                mapping = json.loads(index.read_text())["weight_map"]
                if mapping and all((index.parent / name).is_file() for name in set(mapping.values())):
                    completed_hf.append(str(index.parent))
            manifest.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(),
                            failure={"type": type(error).__name__, "message": str(error)},
                            gradient_norm_record_count=count,
                            complete_hf_weight_versions=completed_hf,
                            complete_optimizer_step_hf_checkpoints=[str(path.parent) for path in sorted(
                                (output / "optimizer-step-checkpoints").glob("checkpoint_step_*/optimizer-step-checkpoint.json"))
                                if json.loads(path.read_text()).get("complete") is True])
            temporary = output / "manifest.json.partial"
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(temporary, output / "manifest.json")
        except Exception as metadata_error:
            print("FAILED_MANIFEST_RECORD_ERROR=" + type(metadata_error).__name__, flush=True)
        raise
    finally:
        # One ownership snapshot for the synchronous post-run acceptance gate.
        # No recurring monitoring and no command lines/authentication data.
        try:
            owner = psutil.Process()
            identities = []
            for process in owner.children(recursive=True):
                try:
                    identities.append({"pid": process.pid, "create_time": process.create_time(),
                                       "name": process.name()})
                except psutil.NoSuchProcess:
                    pass
            snapshot = {"owner_pid": os.getpid(), "owner_create_time": owner.create_time(),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "ray_temporary_directory": str(ray_tmp), "processes": identities}
            inventory = output / "owned-processes-before-ray-shutdown.json"
            temporary = inventory.with_suffix(".json.partial")
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
            os.replace(temporary, inventory)
        except Exception as error:
            print("OWNED_PROCESS_SNAPSHOT_FAILED=" + type(error).__name__, flush=True)
        # This address='local' owner waits only for the processes it started.
        active_error = sys.exc_info()[0] is not None
        try:
            ray.shutdown(wait_for_processes=True)
        except Exception as error:
            if not active_error:
                raise
            # Keep the original training failure; report cleanup separately.
            print("OWNED_RAY_SHUTDOWN_FAILED_DURING_EXCEPTION=" + type(error).__name__, flush=True)


if __name__ == "__main__":
    main()
