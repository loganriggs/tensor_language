from __future__ import annotations

import torch

import tensor_attention_projection_frontier as frontier
import tensor_attention_shared_qk_matched_control as matched


def weighted_error(
    covariance: torch.Tensor, coefficient: torch.Tensor, program,
) -> torch.Tensor:
    reconstructed = program.input_factor.T.double() @ program.output_factor.T.double()
    error = coefficient - reconstructed
    return torch.trace(error.T @ covariance.double() @ error)


def test_independent_weighted_fit_recovers_supported_rank_two_coefficient() -> None:
    torch.manual_seed(71)
    width, rank = 7, 2
    covariance = torch.diag(torch.tensor([1., 12., 3., 8., 2., 6., 4.])).double()
    coefficient = (
        torch.randn(width, rank) @ torch.randn(rank, width)
    ).double()
    # Invert the registered ridge map so its fitted coefficient is the rank-two target.
    scale = torch.diag(covariance).mean()
    regularized = covariance + frontier.RIDGE * scale * torch.eye(width).double()
    weight_t = torch.linalg.solve(covariance, regularized @ coefficient)
    program = matched.independent_activation_weighted_linear(
        covariance, weight_t.T.float(), rank,
    )
    torch.testing.assert_close(
        program.input_factor.T.double() @ program.output_factor.T.double(),
        coefficient, atol=2e-5, rtol=2e-5,
    )
    assert program.stored_values == 2 * width * rank


def test_weighted_fit_beats_ordinary_coefficient_svd_on_anisotropic_input() -> None:
    torch.manual_seed(72)
    width, rank = 8, 3
    covariance = torch.diag(torch.logspace(-2, 2, width)).double()
    weight = torch.randn(width, width)
    coefficient = frontier._registered_coefficient(covariance, weight)
    weighted = matched.independent_activation_weighted_linear(covariance, weight, rank)
    u, singular, vh = torch.linalg.svd(coefficient, full_matrices=False)
    ordinary = frontier.StoredLinear(
        input_factor=u[:, :rank].T.float(),
        output_factor=(vh[:rank].T * singular[:rank]).float(),
    )
    assert weighted_error(covariance, coefficient, weighted) <= weighted_error(
        covariance, coefficient, ordinary,
    ) + 1e-8


def test_protocol_binds_parent_and_zero_native_execution() -> None:
    source = matched.Path(matched.__file__).read_text()
    prereg = matched.PREREG.read_text()
    assert matched.NAMES == ("independent_weighted384", "shared_qk384_replay")
    assert "score_bank" in source and "AttentionNativePoison" in frontier.Path(
        frontier.__file__
    ).read_text()
    assert "replay_within_0.003" in source
    assert "sharing_fidelity_free_within_0.005" in source
    assert "tensor_attention_projection_frontier_results.json" in source
    assert "fewer degrees of freedom" in prereg
