"""Optional fresh NPU Hybrid Adam allocation before expensive rollout generation."""
import os



def trim_unused_cpu_heap():
    """Return only allocator-owned free glibc pages; live tensor storage is untouched."""
    import ctypes
    import psutil

    process = psutil.Process()
    report = {"requested": True,
              "before": {"host_available_bytes": psutil.virtual_memory().available,
                         "process_rss_bytes": process.memory_info().rss}}
    try:
        trim = getattr(ctypes.CDLL(None), "malloc_trim")
    except (AttributeError, OSError) as error:
        report.update(available=False, reason=str(error))
    else:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        report.update(available=True, return_code=int(trim(0)))
    report["after"] = {"host_available_bytes": psutil.virtual_memory().available,
                       "process_rss_bytes": process.memory_info().rss}
    return report


def maybe_preallocate_npu_adam(actor, args, role):
    if os.environ.get("SLIME_NPU_PREALLOCATE_ADAM") != "1":
        return
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    import psutil
    import torch
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    if role != "actor" or not torch.npu.is_available():
        raise RuntimeError("SLIME_NPU_PREALLOCATE_ADAM requires the NPU actor optimizer")
    helper_path = Path(os.environ["SLIME_NPU_PREALLOCATE_ADAM_HELPER"]).resolve()
    report_directory = Path(os.environ["SLIME_NPU_PREALLOCATE_REPORT_DIR"]).resolve()
    if not helper_path.is_file():
        raise FileNotFoundError(helper_path)
    spec = importlib.util.spec_from_file_location("slime_npu_fresh_adam_preallocator", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    hybrids = []
    seen = set()
    def visit(value):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, HybridDeviceOptimizer):
            hybrids.append(value)
            return
        children = getattr(value, "chained_optimizers", None)
        if children is not None:
            for child in children:
                visit(child)
            return
        children = getattr(value, "optimizers", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                visit(child)
            return
        child = getattr(value, "optimizer", None)
        if child is not None:
            visit(child)
    visit(actor.optimizer)
    if not hybrids:
        raise RuntimeError("No actual HybridDeviceOptimizer found for opt-in NPU preallocation")
    actor.clear_memory()
    heap_trim = (
        trim_unused_cpu_heap()
        if os.environ.get("SLIME_NPU_TRIM_UNUSED_CPU_HEAP") == "1"
        else {"requested": False}
    )
    torch.npu.synchronize()
    before = {
        "npu_allocated_bytes": torch.npu.memory_allocated(),
        "npu_reserved_bytes": torch.npu.memory_reserved(),
        "host_available_bytes": psutil.virtual_memory().available,
    }
    scheduler_before = actor.opt_param_scheduler.num_steps
    optimizer_id, model_id = id(actor.optimizer), id(actor.model)
    summaries = [
        helper.preallocate_hybrid_adam_state(value, preallocate_cpu_grads=True)
        for value in hybrids
    ]
    torch.npu.synchronize()
    assert optimizer_id == id(actor.optimizer) and model_id == id(actor.model)
    assert actor.opt_param_scheduler.num_steps == scheduler_before
    report = {
        "rank": torch.distributed.get_rank(), "role": role, "status": "allocated",
        "helper_path": str(helper_path),
        "helper_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        "performed_optimizer_updates": 0, "scheduler_num_steps": scheduler_before,
        "model_optimizer_identity_unchanged": True,
        "hybrid_instances": summaries, "before": before,
        "cpu_heap_trim": heap_trim,
        "allocation_environment": {
            "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
            "HCCL_BUFFSIZE": os.environ.get("HCCL_BUFFSIZE"),
        },
        "after": {
            "npu_allocated_bytes": torch.npu.memory_allocated(),
            "npu_reserved_bytes": torch.npu.memory_reserved(),
            "npu_free_total_bytes": list(torch.npu.mem_get_info()),
            "host_available_bytes": psutil.virtual_memory().available,
            "process_rss_bytes": psutil.Process().memory_info().rss,
        },
    }
    report_directory.mkdir(parents=True, exist_ok=True)
    path = report_directory / f"rank{report['rank']}.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)
    print(f"NPU_ADAM_PREALLOCATION_PASS rank={report['rank']} report={path}", flush=True)
