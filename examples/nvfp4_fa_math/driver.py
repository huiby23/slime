"""Entrypoint inside the pinned CUDA image, using an isolated Slime Ray head."""

from functools import partial
import json
import os
from pathlib import Path

import ray
import torch
import train as slime_train
from slime_plugins.nvfp4_fa.records import verified_save


def main():
    run = Path(os.environ["NVFP4_FA_RUN"])
    job = json.loads((run / "run.json").read_text())
    if torch.cuda.device_count() != 8 or any("A100" not in torch.cuda.get_device_name(i) for i in range(8)):
        raise RuntimeError("Expose the eight allocated A100 GPUs to this container")
    torch.backends.cuda.matmul.allow_tf32 = False
    ray.init(
        address="local",
        num_gpus=8,
        include_dashboard=False,
        namespace="nvfp4-fa-math",
        object_store_memory=4 * 1024**3,
    )
    try:
        args, plan = slime_train.parse_args(return_restore_plan=True)
        if (
            args.no_save_optim
            or args.no_save_rng
            or args.async_save
            or args.global_batch_size != 40
            or args.rollout_batch_size * args.n_samples_per_prompt != 40
            or args.save_interval != 5
        ):
            raise ValueError("A complete synchronous save and one optimizer update per rollout are required")
        prior = job["completed_on_entry"]
        missing = None
        if prior:
            if args.no_load_optim or args.no_load_rng or args.finetune or args.ckpt_step != prior - 1:
                raise ValueError("Megatron would not restore the requested Adam/RNG checkpoint")
            if not (Path(job["parent"]) / "evaluation" / f"step_{prior:03}.json").is_file():
                missing = prior - 1
        slime_train.save_checkpoint = partial(verified_save, slime_train.save_checkpoint)
        slime_train.configure_logger()
        slime_train.main(args, plan, eval_restored_rollout=missing)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
