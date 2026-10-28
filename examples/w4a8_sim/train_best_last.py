"""Run pinned Slime with complete best/last publication across filesystems."""

import train as baseline
from w4a8_sim.checkpoint_policy import save_last


if __name__ == "__main__":
    args, restore_plan = baseline.parse_args(return_restore_plan=True)
    if args.save_interval != 1 or args.async_save or args.no_save_optim or args.no_save_rng:
        raise ValueError("W4A8 saves a complete last checkpoint after every optimizer update")
    baseline.save_checkpoint = save_last
    baseline.configure_logger()
    baseline.main(args, restore_plan)
