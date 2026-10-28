"""Read-only GPU idle observation. Never reserves or kills GPU processes."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 6])
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    since, last = {}, None
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        now = time.monotonic()
        try:
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10)
            processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True, timeout=10)
            busy = {line.split(",")[0].strip() for line in processes.splitlines() if "," in line}
            rows = []
            if last is not None and now - last > 15:
                since.clear()
            for line in raw.splitlines():
                index, uuid, memory, util = [s.strip() for s in line.split(",")]
                index = int(index)
                if index not in args.gpus:
                    continue
                idle = float(memory) <= 64 and float(util) == 0 and uuid not in busy
                if not idle:
                    since.pop(index, None)
                else:
                    since.setdefault(index, now)
                elapsed = now - since.get(index, now)
                rows.append(dict(index=index, uuid=uuid, memory_mib=float(memory), utilization=float(util),
                                 idle=idle, continuous_idle_seconds=elapsed, qualified=idle and elapsed > 300))
            report = {"at": datetime.now(timezone.utc).isoformat(), "gpus": rows,
                      "all_qualified": len(rows) == len(args.gpus) and all(r["qualified"] for r in rows)}
            last = now
        except Exception as exc:
            since.clear()
            report = {"at": datetime.now(timezone.utc).isoformat(), "all_qualified": False, "error": str(exc)}
        temp = args.output / "idle-status.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output / "idle-status.json")
        with (args.output / "idle-observations.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        time.sleep(5)


if __name__ == "__main__":
    main()
