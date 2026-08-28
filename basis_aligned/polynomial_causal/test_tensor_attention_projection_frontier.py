from __future__ import annotations

import torch

import tensor_attention_projection_frontier as frontier


def test_activation_weighted_linear_replays_registered_orientation() -> None:
    torch.manual_seed(61)
    width, rank = 7, 3
    x = torch.randn(80, width)
    covariance = x.T.double() @ x.double() / len(x)
    weight = torch.randn(width, width)
    observed = frontier.activation_weighted_linear(
        covariance, weight, rank, ridge_fraction=1e-8,
    )
    regularized = covariance + 1e-8 * torch.diag(covariance).mean() * torch.eye(
        width, dtype=torch.float64,
    )
    coefficient = torch.linalg.solve(regularized, covariance @ weight.double().T)
    u, singular, vh = torch.linalg.svd(coefficient, full_matrices=False)
    expected = (x @ u[:, :rank].float()) @ (
        vh[:rank].T.float() * singular[:rank].float()
    ).T
    torch.testing.assert_close(observed(x), expected, atol=1e-5, rtol=1e-5)
    assert observed.stored_values == 2 * width * rank


def test_shared_basis_orders_largest_activation_eigenvalues() -> None:
    covariance = torch.diag(torch.tensor([1.0, 9.0, 4.0, 16.0]))
    basis = frontier.shared_activation_basis(covariance, 2)
    projector = basis @ basis.T
    expected = torch.diag(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    torch.testing.assert_close(projector, expected)


def test_registered_arm_classes_and_complete_cost_logic_are_frozen() -> None:
    assert frontier.ARM_SPECS == {
        "routing384": frontier.ArmSpec(384, None, False),
        "value384": frontier.ArmSpec(None, 384, False),
        "joint384": frontier.ArmSpec(384, 384, False),
        "joint512": frontier.ArmSpec(512, 512, False),
        "shared_qk384": frontier.ArmSpec(384, None, True),
    }
    source = frontier.Path(frontier.__file__).read_text()
    assert "require_production=False" in source
    assert "AttentionNativePoison" in source
    assert "literal_native_attention_calls" in source
    assert "stored_bits" in source
    assert "multiply_adds_per_production_forward" in source
    assert "normalized_recovery" in source
    assert "os.O_EXCL" in source
