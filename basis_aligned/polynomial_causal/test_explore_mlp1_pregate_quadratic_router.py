import torch

import explore_mlp1_pregate_quadratic_router as subject


def test_randomized_signed_factors_recover_indefinite_low_rank_toy():
    torch.manual_seed(4)
    batch, dimension, rank = 3, 24, 4
    basis = torch.linalg.qr(torch.randn(batch, dimension, rank)).Q
    # Re-orthogonalized subspace iteration must not collapse away the smaller signed
    # modes of this moderately ill-conditioned planted rank.
    values = torch.tensor([[9.0, -7.0, 4.0, -2.0]]).expand(batch, -1)
    matrices = (basis * values[:, None, :]) @ basis.transpose(1, 2)
    found, found_values = subject.explicit_randomized_signed_factors(matrices, rank, seed=8)
    reconstructed = (found * found_values[:, None, :]) @ found.transpose(1, 2)
    assert torch.allclose(reconstructed, matrices, atol=2e-4, rtol=2e-4)
    assert torch.allclose(
        found_values.abs().sort(dim=1, descending=True).values,
        values.abs().sort(dim=1, descending=True).values,
        atol=2e-4, rtol=2e-4,
    )


def test_randomized_factors_keep_largest_magnitude_signed_tail():
    torch.manual_seed(9)
    dimension = 30
    basis = torch.linalg.qr(torch.randn(1, dimension, 6)).Q
    values = torch.tensor([[11.0, -9.0, 7.0, -5.0, 0.3, -0.1]])
    matrix = (basis * values[:, None, :]) @ basis.transpose(1, 2)
    found, found_values = subject.explicit_randomized_signed_factors(matrix, 4, seed=2)
    assert torch.allclose(
        found_values.abs().sort(dim=1, descending=True).values,
        values[:, :4].abs(), atol=2e-3, rtol=2e-3,
    )
    reconstructed = (found * found_values[:, None, :]) @ found.transpose(1, 2)
    relative_error = (matrix - reconstructed).square().sum() / matrix.square().sum()
    analytic_tail = values[:, 4:].square().sum() / values.square().sum()
    assert torch.allclose(relative_error, analytic_tail, atol=2e-4, rtol=2e-3)


def test_quadratic_score_uses_signed_modes():
    factors = torch.eye(3).reshape(1, 3, 3)
    values = torch.tensor([[2.0, -3.0, 0.5]])
    states = torch.tensor([[1.0, 2.0, 4.0]])
    score = subject.quadratic_scores(states, factors, values, rank=3)
    assert torch.equal(score, torch.tensor([[2.0 - 12.0 + 8.0]]))


def test_quadratic_score_axis_contract_is_not_hidden_by_square_toy():
    factors = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
        [[0.5, 1.0], [1.0, 0.0], [0.0, 2.0]],
    ])  # [atom=2,state=3,rank=2]
    values = torch.tensor([[2.0, -1.0], [3.0, 0.5]])
    states = torch.tensor([[1.0, 2.0, -1.0]])
    projected = torch.einsum("nd,adr->nar", states, factors)
    expected = (projected.square() * values).sum(-1)
    assert torch.equal(subject.quadratic_scores(states, factors, values, 2), expected)


def test_rank8_price_is_a_genuine_full_mlp_simplification():
    price = subject.router_price(8)
    assert price["bias_folded_stored_reals"] == 5_313_664
    assert price["executed_artifact_stored_reals"] == 5_314_816
    assert price["full_mlp_storage_saved_reals"] == 10_612_736
    assert price["full_mlp_storage_saved_fraction"] > 0.66
    assert price["router_score_and_decode_multiplies_per_token"] == 4_759_552
