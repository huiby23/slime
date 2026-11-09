"""Build or launch the eight-A100 Qwen3-4B NVFP4-FA experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .data import HASHES, verify
from slime_plugins.nvfp4_fa.records import checkpoint, write_json

REPO = Path(__file__).resolve().parents[2]
MODEL_FILES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "model.safetensors.index.json": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    "config.json": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors": "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
}
FLAGS = """
--num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728 --num-attention-heads 32
--group-query-attention --num-query-groups 8 --kv-channels 128 --qk-layernorm
--swiglu --disable-bias-linear --normalization RMSNorm --norm-epsilon 1e-6
--position-embedding-type rope --rotary-base 1000000 --vocab-size 151936
--bf16 --actor-num-nodes 1 --actor-num-gpus-per-node 8 --rollout-num-gpus 8
--rollout-num-gpus-per-engine 1 --num-gpus-per-node 8 --colocate --offload-rollout --offload-train
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
--expert-model-parallel-size 1 --expert-tensor-parallel-size 1
--spec slime_plugins.nvfp4_fa.megatron get_spec
--seed 1234 --rollout-seed 1234 --rollout-shuffle --input-key prompt --label-key label
--apply-chat-template --rm-type math --advantage-estimator grpo --kl-coef 0 --kl-loss-coef 0
--entropy-coef 0 --num-rollout 100 --rollout-batch-size 5 --n-samples-per-prompt 8
--over-sampling-batch-size 1 --global-batch-size 40 --micro-batch-size 1 --num-steps-per-rollout 1
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--rollout-max-response-len 3070 --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1
--n-samples-per-eval-prompt 1 --eval-max-response-len 3070 --eval-temperature 0 --eval-top-k 1
--eval-top-p 1 --eval-interval 5 --save-interval 5
--custom-megatron-before-train-step-hook-path slime_plugins.nvfp4_fa.records.trainer_coverage
--custom-eval-rollout-log-function-path slime_plugins.nvfp4_fa.records.evaluation
--lr 1e-6 --weight-decay 0 --lr-decay-style constant --optimizer adam --use-distributed-optimizer
--accumulate-allreduce-grads-in-fp32 --distrib-optim-sharding-type fully_reshardable
--attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 --attention-backend flash
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1
--seq-length 8192 --max-position-embeddings 40960 --log-probs-chunk-size 1024
--sglang-attention-backend flashinfer --sglang-dtype bfloat16 --sglang-kv-cache-dtype auto
--sglang-disable-cuda-graph --sglang-disable-prefill-cuda-graph --sglang-disable-decode-cuda-graph
--sglang-disable-overlap-schedule --sglang-chunked-prefill-size 256 --sglang-context-length 8192
--sglang-max-running-requests 16 --sglang-max-total-tokens 81920 --sglang-mem-fraction-static 0.45
--use-tensorboard --sglang-router-request-timeout-secs 86400 --rollout-data-transport object-store
"""


def file_hash(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def identity():
    h = hashlib.sha256()
    for folder in (REPO / "slime_plugins/nvfp4_fa", Path(__file__).parent):
        for p in sorted(folder.rglob("*.py")):
            h.update(str(p.relative_to(REPO)).encode())
            h.update(p.read_bytes())
    for p in (REPO / "train.py", REPO / "slime/utils/arguments.py", REPO / "slime_plugins/nvfp4_sim/recipe.py"):
        h.update(p.read_bytes())
    return h.hexdigest()


def arguments(model, data, run, parent=None, step=0):
    ckpt = run / "checkpoints"
    args = shlex.split(FLAGS)
    args += [
        "--hf-checkpoint",
        str(model),
        "--tokenizer-path",
        str(model),
        "--load",
        str(parent / "checkpoints" if parent else model),
        "--save",
        str(ckpt),
        "--start-rollout-id",
        str(step),
        "--save-hf",
        str(run / "hf/rollout_{rollout_id:03d}"),
        "--tensorboard-dir",
        str(run / "tensorboard"),
        "--ci-save-grad-norm",
        str(run / "grad_norm/{role}_{rollout_id}_{step_id}.pt"),
        "--prompt-data",
        str(data / "train.jsonl"),
        "--eval-prompt-data",
        "math500",
        str(data / "validation.jsonl"),
        "--apply-chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
        "--sglang-json-model-override-args",
        json.dumps({"architectures": ["NVFP4FAQwen3ForCausalLM"]}),
    ]
    if parent:
        args += ["--ckpt-step", str(step - 1)]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "train"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path, help="Completed checkpoint in a previous attempt directory")
    opts = parser.parse_args()
    run, model, data = [p.resolve() for p in (opts.run_dir, opts.model, opts.data)]
    parent = opts.resume_from.resolve() if opts.resume_from else None
    step = 0
    protocol = identity()
    if parent:
        source = json.loads((parent / "run.json").read_text())
        if source["implementation"] != protocol:
            raise ValueError("Resume requires the same numerical and training code")
        markers = sorted((parent / "saved").glob("step_*.json"))
        if not markers:
            raise ValueError("No verified full checkpoint available")
        step = json.loads(markers[-1].read_text())["step"]
        checkpoint(parent, step)
    args = arguments(model, data, run, parent, step)
    env = dict(
        SGLANG_EXTERNAL_MODEL_PACKAGE="slime_plugins.nvfp4_fa.sglang_models",
        SLIME_DESTROY_WORLD_PROCESS_GROUP="0",
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        NVIDIA_TF32_OVERRIDE="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="2",
        PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]),
    )
    env["NVFP4_FA_RUN"] = str(run)
    env["TENSORBOARD_DIR"] = str(run / "tensorboard")
    cmd = [sys.executable, "-m", "examples.nvfp4_fa_math.driver", *args]
    plan = dict(
        command=cmd,
        model=str(model),
        model_hashes=MODEL_FILES,
        data=str(data),
        splits=HASHES,
        implementation=protocol,
        parent=str(parent) if parent else None,
        completed_on_entry=step,
        cuda_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        rollout_model=env["SGLANG_EXTERNAL_MODEL_PACKAGE"],
    )
    if opts.action == "plan":
        print(json.dumps(plan, indent=2))
        return
    verify(data)
    if any(file_hash(model / name) != digest for name, digest in MODEL_FILES.items()):
        raise ValueError("Model differs from the original Qwen3-4B BF16 weights")
    if run.exists():
        raise FileExistsError(f"Use a fresh --run-dir, including for resume: {run}")
    run.mkdir(parents=True)
    (run / "grad_norm").mkdir()
    write_json(run / "run.json", plan)
    with (run / "driver.log").open("w", buffering=1) as log:
        proc = subprocess.run(cmd, env=os.environ | env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    write_json(run / "exit.json", {"returncode": proc.returncode})
    proc.check_returncode()


if __name__ == "__main__":
    main()
