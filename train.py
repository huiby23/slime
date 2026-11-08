import logging

import ray

from slime.data.checkpoint import save_checkpoint
from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.cleanup import Cleanup
from slime.utils.misc import should_run_periodic_action


def train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan):
    release_train = args.release_train
    # Disk publication restores serving weights after the HF write has completed.
    disk_full_weight_update = (
        args.update_weight_mode == "full" and args.update_weight_transport == "disk"
    )

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    # Opt-in audit context counts successful optimizer updates, not rollout IDs.
    completed_optimizer_steps = getattr(args, "validation_optimizer_step_offset", None)

    def evaluate(rollout_id):
        if completed_optimizer_steps is None:
            return ray.get(rollout_manager.eval.remote(rollout_id))
        return ray.get(rollout_manager.eval.remote(
            rollout_id, completed_optimizer_steps=completed_optimizer_steps
        ))

    if args.offload_rollout and not release_train and not disk_full_weight_update:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    resume_prepared = getattr(args, "hybrid_resume_in_round", False)
    if not resume_prepared:
        actor_model.update_weights()
    # Reattached async producers stay paused until restored weights are installed.
    # This startup notification resumes them once, before entering the train loop.
    ray.get(rollout_manager.training_ready.remote())

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout and not resume_prepared:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        evaluate(0)

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            evaluate(rollout_id)

        if resume_prepared:
            rollout_data_ref = None
            resume_prepared = False
        else:
            rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if completed_optimizer_steps is not None and actor_trains:
            completed_optimizer_steps += args.num_steps_per_rollout

        # Runtime completion releases queue capacity, without claiming that the
        # model/optimizer or this data progress have a durable joint checkpoint.
        ray.get(rollout_manager.training_completed.remote(rollout_id))

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            save_checkpoint(
                args,
                rollout_id,
                actor_model,
                critic_model,
                rollout_manager,
                actor_trains=actor_trains,
            )

        offload_train(actor_trains)
        if args.offload_rollout and not release_train and not disk_full_weight_update:
            ray.get(rollout_manager.onload_weights.remote())
        was_paused = ray.get(rollout_manager.pause_rollout_admission.remote())
        actor_model.update_weights()
        # Final evaluation uses the synchronized engines directly. Keep training
        # producers paused when no later rollout will consume their new work.
        if rollout_id + 1 < args.num_rollout:
            ray.get(rollout_manager.resume_rollout_admission.remote(was_paused))

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            evaluate(rollout_id)


def main(args, restore_plan):
    """Own the attempt's exit policy; actors only clean up their own resources."""
    with Cleanup() as cleanup:
        init_tracking(args)
        startup = None
        try:
            # Create or reattach rollout resources, then run this training attempt.
            startup = create_rollout_manager(args, restore_plan=restore_plan)
            train(
                startup.args, startup.placements, startup.manager, startup.num_rollout_per_epoch, startup.restore_plan
            )
        except BaseException:
            # Failures and interruptions retain serving and replay data for a manual
            # restart. Detach this driver's trainers instead of disposing the session.
            if startup is not None:
                try:
                    startup.close(failed=True)
                except Exception:
                    logging.getLogger(__name__).exception("Failed to clean up training attempt")
            raise
        else:
            startup.close(failed=False)
        finally:
            # Finish this driver's tracking on success, startup failure, or interruption.
            cleanup.run("finish driver tracking", finish_tracking, args)


if __name__ == "__main__":
    args, restore_plan = parse_args(return_restore_plan=True)
    configure_logger()
    main(args, restore_plan)
