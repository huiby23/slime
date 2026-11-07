"""Run pinned Slime with complete best/last publication across filesystems."""

import os
from pathlib import Path

# Cache directories may already be owned by a previous CPU preflight container.
# Create/access them as the training container user, before importing frameworks.
for _key in ["TMPDIR", "RAY_TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]:
    if os.environ.get(_key):
        Path(os.environ[_key]).mkdir(parents=True, exist_ok=True)

import train as baseline
from w4a8_sim.checkpoint_policy import save_last

if __name__ == "__main__":
    args, restore_plan = baseline.parse_args(return_restore_plan=True)
    if args.save_interval != 1 or args.async_save or args.no_save_optim or args.no_save_rng:
        raise ValueError("W4A8 saves a complete last checkpoint after every optimizer update")
    baseline.save_checkpoint = save_last
    baseline.configure_logger()
    # Slime's rollout logger reads this environment variable rather than the
    # Megatron --tensorboard-dir argument. Ray workers inherit it from raylet.
    os.environ["TENSORBOARD_DIR"] = args.tensorboard_dir
    if os.environ.get("W4A8_RESUME_METRICS") == "1":
        from w4a8_sim.metrics import prepare_resume

        prepare_resume(args)
    if os.environ.get("W4A8_ISOLATED_RAY") == "1":
        import ray

        temporary = Path(os.environ["RAY_TMPDIR"])
        temporary.mkdir(parents=True, exist_ok=True)
        ray.init(
            address="local",
            num_gpus=max(args.actor_num_gpus_per_node, args.rollout_num_gpus),
            num_cpus=32,
            object_store_memory=4 * 1024**3,
            include_dashboard=False,
            namespace=os.environ.get("W4A8_RUN_NAMESPACE", "qwen3-4b"),
            _temp_dir=str(temporary),
        )
        try:
            baseline.main(args, restore_plan)
        finally:
            ray.shutdown()
    else:
        baseline.main(args, restore_plan)
