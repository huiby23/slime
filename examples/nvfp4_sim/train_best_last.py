"""Run the pinned Slime driver with best/last checkpoint publication."""

import train as baseline
from nvfp4_sim.checkpoint_policy import save_last

if __name__ == "__main__":
    args, restore_plan = baseline.parse_args(return_restore_plan=True)
    assert args.save_interval == 1 and not args.async_save
    assert not args.no_save_optim and not args.no_save_rng
    baseline.save_checkpoint = save_last
    baseline.configure_logger()
    baseline.main(args, restore_plan)
