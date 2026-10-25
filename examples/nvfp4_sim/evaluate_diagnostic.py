"""Fixed-validation SGLang evaluation for a frozen BF16 or NVFP4 weight export."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(job):
    from transformers import AutoTokenizer

    source = Path(job["dataset"])
    with source.open() as stream:
        data = [json.loads(line) for line in stream if line.strip()]
    assert len(data) == 512
    tokenizer = AutoTokenizer.from_pretrained(job["origin"], local_files_only=True)
    ids = [tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False) for row in data]
    assert max(map(len, ids)) + 1024 <= 2048
    assert job["precision"] in ("bf16", "nvfp4")
    assert Path(job["model"]).joinpath("config.json").is_file() or job.get("allow_pending_model", False)
    kwargs = dict(model_path=job["model"], tp_size=1, dtype="bfloat16", context_length=2048,
                  max_running_requests=16, mem_fraction_static=0.18, random_seed=42,
                  attention_backend="flashinfer", disable_prefill_cuda_graph=True,
                  disable_decode_cuda_graph=True, skip_server_warmup=True, log_level="warning")
    if job["precision"] == "nvfp4":
        kwargs["json_model_override_args"] = json.dumps({
            "architectures": ["NVFP4SimQwen3ForCausalLM"], "nvfp4_sim_weight_cache": True,
        })
    return data, ids, kwargs, hashlib.sha256(source.read_bytes()).hexdigest()


async def evaluate(engine, data, ids, output, status):
    from slime.rollout.rm_hub.math_utils import grade_answer_verl

    semaphore = asyncio.Semaphore(16)
    params = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "max_new_tokens": 1024}

    async def one(index):
        async with semaphore:
            response = await engine.async_generate(input_ids=ids[index], sampling_params=params)
            text = response["text"]
            meta = response["meta_info"]
            finish = meta.get("finish_reason", {})
            finish_type = finish.get("type") if isinstance(finish, dict) else str(finish)
            return {"index": index, "source_index": data[index]["metadata"]["source_index"],
                    "label": data[index]["label"], "text": text,
                    "correct": bool(grade_answer_verl(text, data[index]["label"])),
                    "response_tokens": meta["completion_tokens"], "truncated": finish_type == "length"}

    tasks = [asyncio.create_task(one(i)) for i in range(len(data))]
    rows = []
    with (output / "predictions.jsonl").open("w") as stream:
        for task in asyncio.as_completed(tasks):
            row = await task
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            status.update(state="evaluating", completed_questions=len(rows), correct=sum(x["correct"] for x in rows))
            save(output / "status.json", status)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "nvfp4_sim.sglang_models"
    if not args.preflight:
        import torch

        uuid = job["expected_gpu_uuid"]
        pids = subprocess.check_output(["nvidia-smi", "--id=" + uuid,
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        allowed = set(job.get("reservation_host_pids", []))
        if {int(pid) for pid in pids.splitlines() if pid.strip()} - allowed:
            raise SystemExit(75)
        assert torch.cuda.device_count() == 1
        actual = str(torch.cuda.get_device_properties(0).uuid)
        assert actual.lower().replace("gpu-", "") == uuid.lower().replace("gpu-", "")
        guard = torch.empty(1, device="cuda")
    data, ids, kwargs, dataset_hash = prepare(job)
    report = {"passed": True, "questions": len(data), "dataset_sha256": dataset_hash,
              "thinking": False, "max_response_tokens": 1024, "precision": job["precision"],
              "engine_kwargs": kwargs, "sampling": {"top_k": 1, "temperature": 1.0, "top_p": 1.0}}
    if args.preflight:
        from dataclasses import fields
        from sglang.srt.server_args import ServerArgs
        assert set(kwargs) <= {field.name for field in fields(ServerArgs)}
        save(output / "configuration-check.json", report)
        print(json.dumps(report), flush=True)
        return
    import sglang

    status = {"state": "loading_model", "job_id": job["job_id"], "completed_questions": 0,
              "planned_questions": 512, "physical_gpu": job["physical_gpu"], "started_at": time.time()}
    save(output / "status.json", status)
    engine = None
    try:
        engine = sglang.Engine(**kwargs)
        rows = asyncio.run(evaluate(engine, data, ids, output, status))
        result = dict(report, correct=sum(x["correct"] for x in rows),
                      accuracy=sum(x["correct"] for x in rows) / 512,
                      truncated_ratio=sum(x["truncated"] for x in rows) / 512,
                      mean_response_tokens=sum(x["response_tokens"] for x in rows) / 512,
                      wall_time_seconds=time.time() - status["started_at"])
        save(output / "evaluation.json", result)
        status.update(state="completed", successful_evaluation=True, result=result)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), successful_evaluation=False)
        raise
    finally:
        save(output / "status.json", status)
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
