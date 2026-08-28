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


def test_shared_weighted_factorization_recovers_common_rank_two_maps() -> None:
    torch.manual_seed(62)
    width, rank = 6, 2
    covariance = torch.diag(torch.tensor([1.0, 9.0, 4.0, 16.0, 2.0, 5.0])).double()
    encoder = torch.randn(width, rank)
    weights = {}
    for name in frontier.QK_NAMES:
        decoder = torch.randn(rank, width)
        # Linear weight orientation is the transpose of the row coefficient.
        weights[name] = (encoder @ decoder).T.float()
    bank = frontier.shared_activation_weighted_bank(covariance, weights, rank)
    x = torch.randn(30, width)
    for name in frontier.QK_NAMES:
        expected_coefficient = frontier._registered_coefficient(
            covariance, weights[name], ridge_fraction=frontier.RIDGE,
        )
        expected = x.double() @ expected_coefficient
        torch.testing.assert_close(
            bank(name, x).double(), expected, atol=2e-5, rtol=2e-5,
        )
    assert bank.stored_values == 5 * width * rank


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
