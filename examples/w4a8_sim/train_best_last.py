"""Run pinned Slime with complete best/last publication across filesystems."""

import os
from pathlib import Path

import train as baseline
from w4a8_sim.checkpoint_policy import save_last


if __name__ == "__main__":
    args, restore_plan = baseline.parse_args(return_restore_plan=True)
    if args.save_interval != 1 or args.async_save or args.no_save_optim or args.no_save_rng:
        raise ValueError("W4A8 saves a complete last checkpoint after every optimizer update")
    baseline.save_checkpoint = save_last
    baseline.configure_logger()
    if os.environ.get("W4A8_ISOLATED_RAY") == "1":
        import ray
        temporary=Path(os.environ["RAY_TMPDIR"])
        temporary.mkdir(parents=True,exist_ok=True)
        ray.init(address="local",num_gpus=args.actor_num_gpus_per_node,
            num_cpus=32,object_store_memory=4*1024**3,include_dashboard=False,
            namespace=os.environ.get("W4A8_RUN_NAMESPACE","qwen3-4b"),_temp_dir=str(temporary))
        try:
            baseline.main(args, restore_plan)
        finally:
            ray.shutdown()
    else:
        baseline.main(args, restore_plan)
