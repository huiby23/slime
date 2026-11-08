"""Real-Torch CPU proof that TIM's BF16 observer does not change AdamW.

Run in an existing PyTorch/pytest environment:
python -m pytest -o addopts='' tests/test_tim_optimizer_observer.py
"""

import json
import random
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.tim.bf16.training_validation import before_train_step


def arguments(tmp_path, lr):
    return SimpleNamespace(
        save=str(tmp_path / "checkpoints"),
        bf16=True,
        fp8=None,
        fp4=None,
        optimizer="adam",
        lr=lr,
        min_lr=lr,
        lr_decay_style="constant",
        lr_warmup_iters=0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
    )


def model_and_master():
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.linear_fc1 = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    projection = model.mlp.linear_fc1.weight
    master = torch.nn.Parameter(torch.linspace(0.5, 1.5, 64 * 64, dtype=torch.float32).reshape(64, 64))
    return model, projection, master


class MegatronLikeOptimizer:
    """Expose the same wrapper/param-group contract around real CPU AdamW."""

    def __init__(self, master, args):
        self.optimizer = torch.optim.AdamW(
            [master],
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        self.param_groups = self.optimizer.param_groups
        self.config = SimpleNamespace(decoupled_weight_decay=True)
        self.calls = 0

    def step(self):
        self.calls += 1
        norm = float(self.param_groups[0]["params"][0].grad.norm())
        self.optimizer.step()
        return True, norm, 0


@pytest.mark.parametrize("lr", [1e-6, 1e-5, 1e-3])
@pytest.mark.parametrize("active", [False, True])
def test_two_updates_exactly_match_unobserved_adamw_and_preserve_rng(tmp_path, lr, active):
    args = arguments(tmp_path, lr)
    model, projection, master = model_and_master()
    expected = torch.nn.Parameter(master.detach().clone())
    reference = torch.optim.AdamW(
        [expected],
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    observed = MegatronLikeOptimizer(master, args)
    projection_before = projection.detach().clone()
    torch_rng_before = torch.get_rng_state().clone()
    python_rng_before = random.getstate()
    wrapped = None

    for rollout_id in range(2):
        projection.main_grad = torch.full_like(projection, rollout_id + 1 if active else 0)
        gradient_before = projection.main_grad.clone()
        master.grad = projection.main_grad.float().clone()
        expected.grad = master.grad.clone()
        master_grad_before = master.grad.clone()

        before_train_step(args, rollout_id, 0, [model], observed, None)
        if wrapped is None:
            wrapped = observed.step
        assert observed.step is wrapped
        # Calling the hook twice must only refresh the IDs, never wrap step twice.
        before_train_step(args, rollout_id, 0, [model], observed, None)
        assert observed.step is wrapped
        outcome = observed.step()
        reference.step()

        assert outcome == (True, float(master_grad_before.norm()), 0)
        torch.testing.assert_close(master, expected, rtol=0, atol=0)
        torch.testing.assert_close(master.grad, master_grad_before, rtol=0, atol=0)
        torch.testing.assert_close(projection.main_grad, gradient_before, rtol=0, atol=0)
        torch.testing.assert_close(projection, projection_before, rtol=0, atol=0)
        actual_state, expected_state = observed.optimizer.state[master], reference.state[expected]
        assert actual_state.keys() == expected_state.keys()
        for name in expected_state:
            torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), torch_rng_before)
        assert random.getstate() == python_rng_before

    records = [json.loads(line) for line in (tmp_path / "optimizer-validation.jsonl").read_text().splitlines()]
    assert observed.calls == 2
    assert [row["rollout_id"] for row in records] == [0, 1]
    assert [row["step_id"] for row in records] == [0, 0]
    for row in records:
        assert row["optimizer_success"]
        assert row["grad_finite"] and row["master_delta_finite"]
        assert row["active_learning_step"] is active
        assert row["precision"] == "native_bf16"
        assert row["quantization_recipe_modules"] == 0
        assert row["optimizer"]["param_groups"][0]["lr"] == lr
        if active:
            assert row["master_changed_elements"] > 0
        else:
            assert row["global_grad_norm"] == row["projection_grad_norm"] == 0


def test_clean_bf16_observer_rejects_a_quantization_recipe_before_wrapping(tmp_path):
    args = arguments(tmp_path, 1e-6)
    model, projection, master = model_and_master()
    projection.main_grad = torch.ones_like(projection)
    model._nvfp4_recipe = object()
    optimizer = MegatronLikeOptimizer(master, args)
    original_step = optimizer.step
    with pytest.raises(AssertionError, match="Unexpected QDQ recipe"):
        before_train_step(args, 0, 0, [model], optimizer, None)
    assert optimizer.step == original_step
    assert not hasattr(optimizer, "_bf16_validation_context")
    assert not (tmp_path / "optimizer-validation.jsonl").exists()
