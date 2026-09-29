"""Regression checks for skipped AMP steps and correctly averaged gradients."""

import pytest
import torch

from cellvit.train.optimization import optimizer_step


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan")])
def test_amp_overflow_skips_update_and_recovers(bad_value):
    parameter = torch.nn.Parameter(torch.tensor([0.1]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    scaler.scale(parameter.square().sum()).backward()
    parameter.grad.fill_(bad_value)

    updated, norm = optimizer_step(optimizer, scaler)
    assert not updated and norm is None
    assert parameter.item() == pytest.approx(0.1)
    assert scaler.get_scale() == 64.0

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square().sum()).backward()
    updated, norm = optimizer_step(optimizer, scaler)
    assert updated and norm == pytest.approx(0.2)
    assert parameter.item() == pytest.approx(0.08)


def test_fp32_nonfinite_gradient_still_fails():
    parameter = torch.nn.Parameter(torch.tensor([0.1]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(FloatingPointError, match="AMP disabled"):
        optimizer_step(optimizer, scaler)
    assert parameter.item() == pytest.approx(0.1)


def test_partial_accumulation_matches_a_mean_batch_update():
    parameter = torch.nn.Parameter(torch.tensor([0.1]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    # Planned group: 8 wells. Actual tail: one batch of 2 and one of 1.
    for targets in (torch.tensor([0.0, 0.1]), torch.tensor([0.3])):
        loss = (parameter - targets).square().mean()
        scaler.scale(loss * len(targets) / 8).backward()
    updated, _ = optimizer_step(optimizer, scaler, gradient_multiplier=8 / 3)
    # Mean target is 0.4 / 3; compare with the full-batch analytic gradient.
    assert updated
    assert parameter.item() == pytest.approx(0.1 - 0.1 * 2 * (0.1 - 0.4 / 3))
