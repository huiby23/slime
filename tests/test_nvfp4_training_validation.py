"""CPU checks that the optimizer observer preserves updates and step identity."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.nvfp4_sim.training_validation import KEYS, before_train_step


@pytest.mark.parametrize("active", [False, True])
def test_optimizer_observer_is_passive_and_does_not_double_wrap(tmp_path, active):
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.linear_fc1 = torch.nn.Linear(64, 64, bias=False)
    projection = model.mlp.linear_fc1.weight
    projection.main_grad = torch.ones_like(projection) if active else torch.zeros_like(projection)
    model._nvfp4_recipe = SimpleNamespace(counters=Counter())
    master = torch.nn.Parameter(projection.detach().clone())
    expected = torch.nn.Parameter(master.detach().clone())
    reference = torch.optim.AdamW([expected], lr=1e-3, weight_decay=0.1)

    class Optimizer:
        def __init__(self):
            self.base = torch.optim.AdamW([master], lr=1e-3, weight_decay=0.1)
            self.param_groups = self.base.param_groups
            self.calls = 0

        def step(self):
            self.calls += 1
            self.base.step()
            return True, 2.0 if active else 0.0, 0

    optimizer = Optimizer()
    args = SimpleNamespace(save=str(tmp_path / "checkpoints"))
    for step in range(2):
        before_train_step(args, step, 0, [model], optimizer, None)
        wrapped = optimizer.step
        before_train_step(args, step, 0, [model], optimizer, None)
        assert optimizer.step == wrapped
        master.grad = projection.main_grad.clone()
        expected.grad = projection.main_grad.clone()
        for key in KEYS:
            model._nvfp4_recipe.counters[key] += 113
        result = optimizer.step()
        reference.step()
        assert result == (True, 2.0 if active else 0.0, 0)
        torch.testing.assert_close(master, expected, rtol=0, atol=0)
    records = [json.loads(line) for line in (tmp_path / "optimizer-validation.jsonl").read_text().splitlines()]
    assert [record["rollout_id"] for record in records] == [0, 1]
    assert all(record["active_learning_step"] == active for record in records)
    assert all(record["master_changed_elements"] > 0 for record in records)  # AdamW decay alone also changes weights.
    assert optimizer.calls == 2
