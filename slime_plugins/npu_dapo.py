"""Lifecycle helpers for real DAPO NPU GRPO. Rewards use the standard math scorer."""
from pathlib import Path
import json
import os
import shutil
import torch.distributed as dist


def prune_old_weight_versions(args, current_directory, rollout_engines):
    # Return inactive export buffers on every rank before SGLang reload/HCCL work.
    from slime.utils.memory_utils import clear_memory

    clear_memory()
    if os.environ.get("SLIME_NPU_TRIM_UNUSED_CPU_HEAP") == "1":
        from slime.utils.npu_optimizer_preallocation import trim_unused_cpu_heap

        report = trim_unused_cpu_heap()
        report.update(rank=dist.get_rank(), weight_directory=str(current_directory))
        directory = Path(current_directory).parent.parent / "post-write-memory" / Path(current_directory).name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank{dist.get_rank()}.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, path)
        print(f"NPU_POST_WRITE_CPU_HEAP_TRIM rank={dist.get_rank()} report={path}", flush=True)
    if dist.get_rank() != 0:
        return
    # Preserve every completed policy snapshot for held-out validation audits.
    # All-rank memory cleanup above remains mandatory in this mode.
    if os.environ.get("SLIME_NPU_KEEP_ALL_WEIGHT_VERSIONS") == "1":
        return
    current = Path(current_directory)
    version = int(current.name.removeprefix("weight_v"))
    for directory in current.parent.glob("weight_v[0-9]*"):
        if directory.is_dir() and not directory.is_symlink():
            if int(directory.name.removeprefix("weight_v")) < version - 1:
                shutil.rmtree(directory)
