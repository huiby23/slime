"""Transfer explicitly reserved GPUs to the training supervisor without a gap."""

import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import torch


def main():
    path = Path(sys.argv[1])
    config = json.loads(path.read_text())
    allowed = set(config["reservation_host_pids"])
    assert torch.cuda.device_count() == len(config["assigned_gpu_uuids"]) == 4
    for uuid in config["assigned_gpu_uuids"]:
        output = subprocess.check_output(
            ["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        )
        pids = {int(line) for line in output.splitlines() if line.strip()}
        assert pids <= allowed, (uuid, pids, allowed)
        memory = int(
            subprocess.check_output(
                ["nvidia-smi", "--id=" + uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
            )
        )
        assert memory <= 2048, (uuid, memory)
    actual = [str(torch.cuda.get_device_properties(i).uuid).lower().replace("gpu-", "") for i in range(4)]
    assert set(actual) == {uuid.lower().replace("gpu-", "") for uuid in config["assigned_gpu_uuids"]}
    # Keep these allocations live while the original supervisor starts Ray and
    # trainers. Reservation containers are released only after this marker.
    pins = []
    for index in range(4):
        with torch.cuda.device(index):
            pins.append(torch.empty(1, device="cuda"))
    torch.cuda.set_device(0)
    root = Path(config["container_run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "gpu-contexts-ready.json").write_text(
        json.dumps({"assigned_gpu_uuids": config["assigned_gpu_uuids"], "context_count": len(pins)}) + "\n"
    )
    assert config["gpu_allocation_managed_by_sequence"]
    sys.argv = ["/study/scripts/run_online_training.py", str(path)]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
