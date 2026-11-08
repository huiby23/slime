"""Time the real TP8 HiF4 serving path with fixed prompts and no optimizer updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_hbm():
    raw = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    values = [int(v) for v in re.findall(r"([0-9]+) / 65536\s*\|", raw)]
    if len(values) != 8:
        raise RuntimeError("Cannot identify all eight 910B HBM readings")
    return raw, values


def measure(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = args.repo.resolve()
    workspace = Path("/root/hby")
    actual_pilot = repo / "scripts/npu/c4v8/workspace/experiments/c4v8_30b_sg_pilot/pilot.py"
    pilot = load_file("hif4_server_pilot", actual_pilot)
    recipe_dir = repo / "scripts/npu/c8v16/workspace/experiments/c8v16_recipe"
    sys.path.insert(0, str(recipe_dir))
    os.environ["SLIME_FA_CUBE_FORMAT"] = "hif4"
    from train_c8v16 import AttentionRecipe

    recipe = AttentionRecipe(workspace)
    rows = json.loads((workspace / "experiments/c4v8_30b_sg_pilot/dapo-first-two-nonthinking.json").read_text())["prompts"]
    env = os.environ | recipe.environment(out)
    env.update(ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", MAX_COMPILE_CORE_NUMBER="1",
               PYTORCH_NPU_ALLOC_CONF="expandable_segments:False", HCCL_BUFFSIZE="128", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", ASCEND_USE_FIA="False", SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK="1",
               SGLANG_LOGITS_PROCESSER_CHUNK_SIZE="64", TRITON_CACHE_DIR=str(workspace / "cache/triton-hif4"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = dict(model_path=str(workspace / "models/Qwen3-30B-A3B"), attention_backend=recipe.backend,
                  prefill_attention_backend=recipe.backend, decode_attention_backend=recipe.backend,
                  device="npu", dtype="bfloat16", kv_cache_dtype="auto", host="127.0.0.1", port=port,
                  tp_size=8, dp_size=1, pp_size=1, nnodes=1, context_length=8192, max_total_tokens=12288,
                  max_running_requests=4, chunked_prefill_size=256, mem_fraction_static=.135, enable_memory_saver=True,
                  disable_cuda_graph=True, enable_torch_compile=False, enable_mixed_chunk=False,
                  disable_overlap_schedule=True, skip_server_warmup=True, stream_interval=1, watchdog_timeout=600)
    report = dict(server=server, implementation=recipe.implementation_sha, policy=recipe.policy_sha,
                  samples=[], optimizer_updates=0, fixture="length-controlled repeated DAPO prompt token IDs")
    save = lambda: (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    raw, initial = read_hbm()
    if raw.count("No running processes found in NPU") != 8:
        raise RuntimeError("At least one NPU already has a process")
    peak = initial.copy()
    end = threading.Event()
    def monitor():
        while not end.wait(2):
            try:
                _, usage = read_hbm()
                peak[:] = map(max, peak, usage)
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                report["hbm_error"] = str(exc)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    command = [str(recipe.shim), "-m", "slime.backends.sglang_utils.runtime"]
    with (out / "server.log").open("w") as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, cwd=repo)
    owner = pilot.smoke.proc(child.pid)
    if owner["pgrp"] != child.pid or owner["sid"] != child.pid:
        raise RuntimeError("Unexpected SG ownership")
    report["owner"] = owner
    save()
    watcher.start()
    try:
        child.stdin.write(json.dumps(dict(server_args=server, parent_pid=os.getpid())))
        child.stdin.close()
        for _ in range(280):
            if child.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {child.returncode}")
            try:
                if pilot.smoke.fetch(port, "/health", timeout=2)["status"] == 200:
                    break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Server startup")
        for length in args.lengths:
            requests = []
            for index in range(args.batch):
                ids = rows[index % len(rows)]["input_ids"]
                # A unique first ID prevents sharing between benchmark requests.
                ids = ([1000 + index] + (ids * (1 + length // len(ids))))[:length]
                requests.append(dict(input_ids=ids, stream=True, return_logprob=True, logprob_start_len=0,
                                     sampling_params=dict(temperature=0., max_new_tokens=args.tokens, ignore_eos=True)))
            for repeat in range(3):
                tag = f"len{length}-repeat{repeat}"
                barrier = threading.Barrier(args.batch + 1)
                def one(pair):
                    i, request = pair
                    barrier.wait(timeout=10)
                    answer, metrics = pilot.stream_request(port, request, out / f"{tag}-r{i}.jsonl", time.monotonic()+1800, 1800)
                    meta = answer["meta_info"]
                    return dict(rank=i, output=meta["output_token_logprobs"], prompt=meta["input_token_logprobs"],
                                first=metrics["client_first_nonempty_chunk_seconds"], last=metrics["client_first_to_last_chunk_seconds"]+metrics["client_first_nonempty_chunk_seconds"],
                                first_count=metrics["first_chunk_completion_tokens"])
                with ThreadPoolExecutor(args.batch) as pool:
                    futures = [pool.submit(one, pair) for pair in enumerate(requests)]
                    start = time.monotonic()
                    barrier.wait(timeout=10)
                    answers = [future.result() for future in futures]
                    elapsed = time.monotonic() - start
                (out / f"{tag}-outputs.json").write_text(json.dumps(answers, allow_nan=False))
                report["samples"].append(dict(length=length, repeat=repeat, elapsed=elapsed,
                    tokens_s=args.batch*args.tokens/elapsed, per_request_decode=[(args.tokens-row["first_count"])/(row["last"]-row["first"]) for row in answers]))
                save()
        audit = [json.loads(p.read_text()) for p in (out / "backend-audit").glob("*.json")]
        for stage in ("first_extend", "first_decode"):
            if len({x["pid"] for x in audit if x.get("event")==stage}) != 8:
                raise RuntimeError(f"Missing eight actual scheduler events for {stage}")
        report["completed"] = True
    finally:
        end.set()
        watcher.join(timeout=20)
        report["peak_sampled_hbm_mib"] = peak
        if child.poll() is None:
            for sig, timeout in [(signal.SIGTERM, 20), (signal.SIGKILL, 8)]:
                current = pilot.smoke.proc(child.pid)
                if (current["startticks"] != owner["startticks"] or current["pgrp"] != child.pid
                        or current["sid"] != child.pid or "c8v16_runtime" not in current["cmdline"]):
                    raise RuntimeError("SG identity changed before cleanup")
                os.killpg(child.pid, sig)
                try:
                    child.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
        report["server_returncode"] = child.poll()
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[1024,3968])
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--tokens", type=int, default=12)
    measure(p.parse_args())
