import math

import pytest
import torch

import mlp2_trajectory_robust_objective as objective


def test_centered_energy_is_fit_scale_and_rejects_constant_target():
    target = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    assert objective.centered_target_energy(target).item() == pytest.approx(2.5)
    with pytest.raises(ValueError):
        objective.centered_target_energy(torch.ones(3, 2))


def test_balanced_loss_gives_each_background_half_weight():
    target = torch.zeros(2, 2)
    native = torch.ones(2, 2)
    c512 = torch.full((2, 2), 2.0)
    loss, report = objective.balanced_background_loss(
        native, target, c512, target, 1.0, 4.0,
    )
    assert report["native_normalized_mse"].item() == pytest.approx(1.0)
    assert report["c512_normalized_mse"].item() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(1.0)


def test_background_duplication_does_not_change_weight():
    target = torch.zeros(2, 3)
    native = torch.ones(2, 3)
    c512 = torch.full((2, 3), 3.0)
    first, _ = objective.balanced_background_loss(
        native, target, c512, target, 1.0, 9.0,
    )
    second, _ = objective.balanced_background_loss(
        native.repeat(7, 1), target.repeat(7, 1), c512, target, 1.0, 9.0,
    )
    assert second.item() == pytest.approx(first.item())


def test_background_swap_is_invariant():
    target = torch.zeros(2, 3)
    native = torch.ones(2, 3)
    c512 = torch.full((2, 3), 3.0)
    first, _ = objective.balanced_background_loss(
        native, target, c512, target, 1.0, 3.0,
    )
    second, _ = objective.balanced_background_loss(
        c512, target, native, target, 3.0, 1.0,
    )
    assert second.item() == pytest.approx(first.item())


def test_gradients_reach_both_backgrounds():
    native = torch.tensor([1.0], requires_grad=True)
    c512 = torch.tensor([2.0], requires_grad=True)
    loss, _ = objective.balanced_background_loss(
        native, torch.zeros(1), c512, torch.zeros(1), 1.0, 1.0,
    )
    loss.backward()
    assert native.grad.item() == pytest.approx(1.0)
    assert c512.grad.item() == pytest.approx(2.0)


def test_checkpoint_is_minimax_then_mean():
    assert objective.retain_checkpoint(0.8, 0.8, 0.7, 1.0, 1.0, 1.0)
    assert not objective.retain_checkpoint(0.6, 1.01, 0.7, 1.0, 1.0, 1.0)
    assert objective.retain_checkpoint(0.7, 1.0, 0.8, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        objective.robust_checkpoint_key(math.nan, 1.0)


def test_checkpoint_cannot_sacrifice_either_background():
    assert not objective.retain_checkpoint(0.9, 0.95, 0.2, 1.0, 0.2, 1.0)
    assert not objective.retain_checkpoint(0.2, 1.03, 0.2, 1.0, 0.2, 1.0)


def test_objective_rejects_bfloat16():
    with pytest.raises(ValueError):
        objective.normalized_mse(
            torch.ones(2, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.bfloat16), 1.0,
        )


@pytest.mark.parametrize("energy", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_energy_fails_closed(energy):
    with pytest.raises(ValueError):
        objective.normalized_mse(torch.ones(1), torch.zeros(1), energy)
