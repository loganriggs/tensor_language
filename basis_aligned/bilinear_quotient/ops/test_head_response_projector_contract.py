import pytest
import torch

from head_response_projector_contract import (
    HeadResponseProjectorError,
    absolute_projected_head_response,
    orthonormal_basis,
    principal_angle_cosines,
    projector_frobenius_distance,
)


def test_orthonormal_basis_and_fixed_unit_projection():
    raw = torch.tensor([[2.0], [0.0], [0.0]])
    basis = orthonormal_basis(torch, raw)
    assert torch.allclose(basis.T @ basis, torch.eye(1))
    base = torch.zeros(1, 2, 2, 3)
    donor = torch.ones_like(base)
    changed = absolute_projected_head_response(torch, base, donor, {1: raw})
    assert torch.equal(changed[..., 0, :], base[..., 0, :])
    assert torch.allclose(changed[..., 1, :], torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]))


def test_projection_is_absolute_not_added_to_live_response():
    base = torch.full((1, 1, 1, 2), 3.0)
    donor = torch.tensor([[[[5.0, 7.0]]]])
    changed = absolute_projected_head_response(torch, base, donor, {0: torch.tensor([[1.0], [0.0]])})
    assert torch.allclose(changed, torch.tensor([[[[5.0, 3.0]]]]))


def test_basis_gradient_is_live():
    raw = torch.tensor([[1.0], [1.0]], requires_grad=True)
    base = torch.zeros(1, 1, 1, 2)
    donor = torch.tensor([[[[2.0, 0.0]]]])
    absolute_projected_head_response(torch, base, donor, {0: raw})[..., 1].sum().backward()
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0.0


def test_fold_metrics_are_gauge_invariant():
    left = torch.tensor([[1.0], [0.0]])
    right = -left
    assert torch.allclose(projector_frobenius_distance(torch, left, right), torch.tensor(0.0))
    assert torch.allclose(principal_angle_cosines(torch, left, right), torch.tensor([1.0]))


@pytest.mark.parametrize("raw", [torch.zeros(3), torch.zeros(2, 3), torch.tensor([[float("nan")]])])
def test_invalid_basis_fails_closed(raw):
    with pytest.raises(HeadResponseProjectorError):
        orthonormal_basis(torch, raw)
