"""Native BF16 driver with early evaluation and complete checkpoint retention.

The pinned Slime training loop and update order stay intact. Only its evaluation
cadence is extended: evaluate after every one of the first configured updates.
"""

import ast
import os
from pathlib import Path


def _configured_count(name):
    value = int(os.environ.get(name, "0"))
    if value < 0:
        raise ValueError(name + " must be nonnegative")
    return value


def runtime_protocol_check(args, source_path=None):
    """CPU-only config gate, optionally including the pinned native loop AST.

    Preflight callers should supply the selected Slime checkout's ``train.py``.
    The executable entry point always checks that file before training starts.
    """
    early = _configured_count("TIM_EARLY_EVAL_UPDATES")
    retain = _configured_count("TIM_RETAIN_EARLY_CHECKPOINTS")
    assert args.save_interval == 1 and not args.async_save
    assert not args.no_save_optim and not args.no_save_rng
    assert 0 <= early <= args.num_rollout
    assert 0 <= retain <= args.num_rollout
    assert retain >= early, "Early evaluated checkpoints must remain available"
    if early:
        assert args.eval_interval is not None and args.eval_interval > 0
        assert not args.skip_eval_before_train, "The early curve includes original W0"
    planned = []
    if args.eval_interval is not None:
        if not args.skip_eval_before_train:
            planned.append(0)
        planned.extend(
            completed
            for completed in range(1, args.num_rollout + 1)
            if completed <= early or completed % args.eval_interval == 0
        )
    report = {
        "early_eval_updates": early,
        "retain_early_checkpoints": retain,
        "evaluation_completed_updates_without_epoch_triggers": planned,
        "native_epoch_evaluation_triggers_preserved": True,
        "native_training_loop_unchanged": True,
        "tensorboard_completed_update_prefix": "eval_completed_updates/",
    }
    if source_path is not None:
        report["native_periodic_calls"] = validate_native_periodic_calls(source_path)
    return report


def make_periodic_action(native, eval_interval, early_updates):
    """Extend only eval calls; the pinned native save call passes num_rollout.

    In pinned train.py the checkpoint branch calls this function with four
    arguments (including total rollouts), and the evaluation branch with three.
    This keeps save behavior identical even when eval and save intervals match.
    """

    def periodic(rollout_id, interval, num_rollout_per_epoch=None, num_rollout=None):
        original = native(rollout_id, interval, num_rollout_per_epoch, num_rollout)
        is_eval = num_rollout is None and interval == eval_interval
        if is_eval and 1 <= rollout_id + 1 <= early_updates:
            return True
        return original

    return periodic


def validate_native_periodic_calls(source_path):
    """Fail before training if the pinned loop's eval/save distinction changes."""
    tree = ast.parse(Path(source_path).read_text())
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    calls = sorted(
        (
            node
            for node in ast.walk(train)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "should_run_periodic_action"
        ),
        key=lambda node: node.lineno,
    )
    expected = [
        ["rollout_id", "args.save_interval", "num_rollout_per_epoch", "args.num_rollout"],
        ["rollout_id", "args.eval_interval", "num_rollout_per_epoch"],
    ]
    actual = [[ast.dump(argument, include_attributes=False) for argument in call.args] for call in calls]
    wanted = [
        [ast.dump(ast.parse(argument, mode="eval").body, include_attributes=False) for argument in arguments]
        for arguments in expected
    ]
    if actual != wanted or any(call.keywords for call in calls):
        raise RuntimeError("Pinned train.py periodic-action call contract changed")
    return {"save_call_arguments": 4, "eval_call_arguments": 3, "passed": True}


def main():
    """Run with the selected pinned Slime checkout and this repo on PYTHONPATH."""
    import train as baseline
    from slime_plugins.tim.bf16.checkpoint_policy import save_last
    from slime_plugins.tim.bf16.training_validation import expected_optimizer_settings

    args, restore_plan = baseline.parse_args(return_restore_plan=True)
    protocol = runtime_protocol_check(args, baseline.__file__)
    expected_optimizer_settings(args)
    baseline.save_checkpoint = save_last
    baseline.should_run_periodic_action = make_periodic_action(
        baseline.should_run_periodic_action, args.eval_interval, protocol["early_eval_updates"]
    )
    baseline.configure_logger()
    baseline.main(args, restore_plan)


if __name__ == "__main__":
    main()
