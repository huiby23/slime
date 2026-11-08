"""Original Qwen3-30B-A3B, W16A16C8V16, 100 fresh DAPO GRPO updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ADAPTERS = HERE.parent / "c8v16_adapters"
sys.path.insert(0, str(ADAPTERS))
from c8v16_profile import BACKEND_NAME as BACKEND, CUBE_FORMAT, HIF4_DIR, IS_HIF4, PRECISION

POLICY_FILE = (HIF4_DIR if IS_HIF4 else HERE) / "precision-policy.json"
SPEC = ["c8v16_megatron_spec", "get_spec"]
VERSION = "c4v16-hif4-base100-every5" if IS_HIF4 else "c8v16-mxfp8-bf16-base100-every5"

# Locate Slime by source identity. Model/data/runs live in a separate workspace.
REPOSITORY = next((parent for parent in HERE.parents if (parent / "scripts/npu/qwen30b_dapo.py").is_file()), None)
if REPOSITORY is None:
    raise ImportError("Run train_c8v16.py from the C8V16 Slime checkout")


def dataset_identity(workspace):
    source = REPOSITORY / "scripts/npu/c4v8/data-inputs.json"
    inputs = json.loads(source.read_text())["inputs"]
    for entry in inputs:
        path = workspace / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Data fingerprint differs for {path}")


class AttentionRecipe:
    backend = BACKEND
    repository = REPOSITORY
    validation_helper = HERE / "aime_protocol.py"

    def __init__(self, workspace):
        self.workspace = workspace
        self.shim = ADAPTERS / "python-c8v16"
        self.policy = json.loads(POLICY_FILE.read_text())
        canonical = json.dumps(self.policy, sort_keys=True, separators=(",", ":")).encode()
        self.policy_sha = hashlib.sha256(canonical).hexdigest()
        if not os.access(self.shim, os.X_OK):
            raise FileNotFoundError(f"Missing executable SGLang runtime shim {self.shim}")
        from c8v16_profile import implementation_sha256
        self.implementation_sha = implementation_sha256()

    def environment(self, output):
        search_path = [str(REPOSITORY), str(ADAPTERS), str(HERE.parent / "c8v16_attention")]
        search_path.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
        return {
            "SLIME_FA_CUBE_FORMAT": CUBE_FORMAT,
            "HBY_C8V16_POLICY_FILE": str(POLICY_FILE),
            "HBY_C8V16_POLICY_SHA256": self.policy_sha,
            "HBY_C8V16_IMPLEMENTATION_SHA256": self.implementation_sha,
            "HBY_C8V16_TRAIN_AUDIT_DIR": str(output / "trainer-backend-audit"),
            "HBY_C8V16_ROLLOUT_AUDIT_DIR": str(output / "backend-audit"),
            "HBY_C8V16_REAL_PYTHON": str(Path(sys.executable).absolute()),
            "HBY_C8V16_WORKSPACE": str(self.workspace),
            "HBY_C8V16_REPOSITORY": str(REPOSITORY),
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "PYTHONPATH": os.pathsep.join(dict.fromkeys(path for path in search_path if path)),
        }

    def arguments(self):
        return [
            "--sglang-prefill-attention-backend", BACKEND,
            "--sglang-decode-attention-backend", BACKEND,
            "--sglang-dtype", "bfloat16", "--sglang-kv-cache-dtype", "auto",
            "--sglang-python-executable", str(self.shim),
            *(["--sglang-server-concurrency", "8", "--sglang-router-request-timeout-secs", "86400"] if IS_HIF4 else []),
            "--spec", *SPEC,
        ]

    def manifest(self):
        return {
            "precision": f"{PRECISION}: BF16 backbone, {CUBE_FORMAT.upper()} Cube operands, BF16 Vector boundaries with FP32 reductions, FP32 Adam",
            "attention_simulation": {
                "backend": BACKEND, "trainer_spec": SPEC, "policy": self.policy,
                "policy_sha256": self.policy_sha, "implementation_sha256": self.implementation_sha,
                "cube": ("HiF4 Q/K/P/V, S1P2 with E6M2 and 8/4 subgroups, block64 on contraction"
                         if IS_HIF4 else "QK/PV input QDQ: MXFP8 E4M3FN / E8M0, block32 along contraction"),
                "vector": "BF16 boundaries; FP32 max/sum and matrix multiplication accumulation",
                "backbone": "BF16", "kv_cache": "BF16", "backward": "high-precision STE",
            },
        }

    def register_driver_backend(self):
        from c8v16_bootstrap import register_cli_choice
        register_cli_choice()

    def verify_args(self, args, env):
        required = {
            "spec": SPEC, "bf16": True,
            "sglang_python_executable": str(self.shim),
            "sglang_attention_backend": BACKEND, "sglang_prefill_attention_backend": BACKEND,
            "sglang_decode_attention_backend": BACKEND, "sglang_dtype": "bfloat16",
            "sglang_kv_cache_dtype": "auto", "sglang_disable_cuda_graph": True,
            "rollout_batch_size": 5, "n_samples_per_prompt": 16, "global_batch_size": 4,
            "num_steps_per_rollout": 20, "num_rollout": 5, "rollout_shuffle": True,
            "rm_type": "math", "advantage_estimator": "grpo",
            "dynamic_sampling_filter_path": "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        }
        if IS_HIF4:
            required.update(sglang_server_concurrency=8, sglang_router_request_timeout_secs=86400)
        for field, expected in required.items():
            if getattr(args, field, None) != expected:
                raise ValueError(f"Unexpected resolved {PRECISION} {field}: {getattr(args, field, None)!r}")
        for field in ("sglang_quantization", "fp8", "fp4", "custom_rm_path", "load_debug_rollout_data", "save_debug_rollout_data"):
            if getattr(args, field, None):
                raise ValueError(f"Incompatible with the {PRECISION} recipe: {field}")
        for field in ("main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
            if str(getattr(args, field, None)).split(".")[-1] not in {"float32", "fp32"}:
                raise ValueError(f"Expected FP32 {field}")
        actual = args.train_env_vars
        if isinstance(actual, str):
            actual = json.loads(actual)
        if any(actual.get(key) != value for key, value in env.items()):
            raise ValueError("Trainer and rollout precision environments differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/root/hby"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-command", action="store_true", help="Render actual Slime argv/manifest without Ray (default)")
    mode.add_argument("--validate-arguments", action="store_true", help="Parse installed Slime/Megatron/SGLang config without training")
    mode.add_argument("--run", action="store_true", help="Start the 100-update job on the configured NPU host")
    options = parser.parse_args(argv)
    workspace = options.workspace.resolve()
    if any(key.startswith("SLIME_NPU_REPLAY0_") and value for key, value in os.environ.items()):
        raise ValueError("Fresh C8V16 requires newly generated on-policy answers")
    dataset_identity(workspace)
    for path in [str(ADAPTERS), str(REPOSITORY), str(REPOSITORY / "scripts/npu")]:
        sys.path.insert(0, path)
    from qwen30b_dapo import main as train_dapo
    flags = [
        "--workspace", str(workspace), "--version", VERSION,
        "--parent-completed-optimizer-steps", "0", "--lineage-target-optimizer-steps", "100",
        "--rollouts", "5", "--prompts", "5", "--samples", "16", "--minibatch", "4",
        "--validation-trajectory-dir", str(workspace / "evaluations" / VERSION),
        "--offline-checkpoint-validation", "--keep-all-weight-versions", "--hf-checkpoint-every-steps", "5",
        "--offload-fraction", "0.35", "--response-length", "4096", "--sequence-length", "8192",
        "--sglang-mem-fraction", "0.135", "--sglang-context-length", "8192",
        "--sglang-max-total-tokens", "12288", "--sglang-max-running-requests", "4", "--sglang-chunked-prefill-size", "256",
    ]
    if not options.run:
        flags.append("--validate-arguments" if options.validate_arguments else "--print-command")
    flags.extend(["--", "--rollout-shuffle", "--log-probs-chunk-size", "1024"])
    train_dapo(AttentionRecipe(workspace), flags)


if __name__ == "__main__":
    main()
