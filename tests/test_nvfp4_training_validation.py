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


@pytest.mark.parametrize("master_numel", [128, 8_000_001])
def test_observer_accepts_dp_rank_without_medium_master(tmp_path, monkeypatch, master_numel):
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.linear_fc1 = torch.nn.Linear(64, 64, bias=False)
    model.mlp.linear_fc1.weight.main_grad = torch.ones_like(model.mlp.linear_fc1.weight)
    model._nvfp4_recipe = SimpleNamespace(counters=Counter())
    master = torch.nn.Parameter(torch.ones(master_numel))
    expected = torch.nn.Parameter(master.detach().clone())
    reference = torch.optim.AdamW([expected], lr=1e-3)

    class Optimizer:
        def __init__(self):
            self.base = torch.optim.AdamW([torch.nn.Parameter(torch.empty(0)), master], lr=1e-3)
            self.param_groups = self.base.param_groups

        def step(self):
            self.base.step()
            return True, 2.0, 0

    optimizer = Optimizer()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    before_train_step(SimpleNamespace(save=str(tmp_path / "checkpoints")), 10, 0, [model], optimizer, None)
    master.grad = torch.ones_like(master)
    expected.grad = master.grad.clone()
    for key in KEYS:
        model._nvfp4_recipe.counters[key] += 113
    assert optimizer.step() == (True, 2.0, 0)
    reference.step()
    torch.testing.assert_close(master, expected, rtol=0, atol=0)
    record = json.loads((tmp_path / "optimizer-validation-rank-3.jsonl").read_text())
    assert record["rank"] == 3 and record["world_size"] == 4
    assert record["active_learning_step"]
    assert record["master"]["observed_numel"] == min(master_numel, 1_048_576)
    assert record["master_changed_elements"] == record["master"]["observed_numel"]
