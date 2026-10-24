"""Pause the existing driver during one train RPC, save last, and optionally hold.

Run inside that experiment's container. Pausing during the training RPC ensures
the source cursor and completed model update share the same batch boundary.
Only our recorded driver PID is signalled; trainer/serving actors remain alive.
"""

import argparse
import json
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import ray

from nvfp4_sim.checkpoint_policy import atomic_json, save_last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, default=Path("/nvfp4-run"))
    parser.add_argument("--hold-driver", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    root = args.attempt
    ray.init(address="127.0.0.1:6379", namespace="slime-training-recovery", log_to_driver=False)
    named = [
        item
        for item in ray.util.list_named_actors(all_namespaces=True)
        if item["namespace"] == "slime-training-recovery" and item["name"].endswith(":serving")
    ]
    assert len(named) == 1, named
    name = named[0]["name"]
    serving = ray.get_actor(name, namespace="slime-training-recovery")
    manager = ray.get_actor(name.removesuffix(":serving"), namespace="slime-training-recovery")
    handlers = ray.get(serving.__ray_call__.remote(lambda actor: actor.training_actors["actor"]))
    assert len(handlers) == 1, "This migration helper targets the existing single-rank job"
    started = time.monotonic()
    driver = None
    saved = False
    try:
        print("Waiting for a fresh trainer RPC boundary", flush=True)
        while time.monotonic() - started < args.timeout:
            status = json.loads((root / "results/status.json").read_text())
            text = re.sub(r"\x1b\[[0-9;]*m", "", (root / "results/logs/train.log").read_text(errors="replace"))
            begin = list(re.finditer(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*Timer train start", text))
            end = text.rfind("Timer train end")
            if begin:
                latest = begin[-1]
                timestamp = datetime.strptime(latest.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if latest.start() > end and 0 <= age < 120:
                    driver = int(status["trainer_pid"])
                    assert "train.py" in Path(f"/proc/{driver}/cmdline").read_text()
                    os.kill(driver, signal.SIGSTOP)
                    break
            time.sleep(1)
        if driver is None:
            raise TimeoutError("No safe training boundary was observed")
        print("Driver paused; waiting for its in-flight trainer RPC to finish", flush=True)
        model_args = ray.get(handlers[0].__ray_call__.remote(lambda actor: actor.args), timeout=600)
        rows = [
            json.loads(line)
            for line in (root / "results/optimizer-validation.jsonl").read_text().splitlines()
            if line.strip()
        ]
        rollout_id = rows[-1]["rollout_id"]

        class Group:
            def __init__(self):
                self.args, self._actor_handlers = model_args, handlers

            def save_model(self, index, force_sync=False):
                ray.get([actor.save_model.remote(index, force_sync=True) for actor in handlers])

        snapshot = save_last(model_args, rollout_id, Group(), None, manager, actor_trains=True)
        proof = {
            "completed_updates": rollout_id + 1,
            "rollout_id": rollout_id,
            "checkpoint": str(snapshot),
            "driver_pid": driver,
            "held_for_migration": args.hold_driver,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(root / "migration-last.json", proof)
        saved = True
        print(json.dumps(proof, indent=2), flush=True)
    finally:
        if driver is not None and (not args.hold_driver or not saved):
            os.kill(driver, signal.SIGCONT)
        ray.shutdown()


if __name__ == "__main__":
    main()
