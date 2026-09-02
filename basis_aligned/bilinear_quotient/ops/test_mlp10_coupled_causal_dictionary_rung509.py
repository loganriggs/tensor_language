import torch

import mlp10_coupled_causal_dictionary_rung509 as rung


def test_assignment_is_simplex_and_branch_swap_invariant():
    generator = torch.Generator().manual_seed(1)
    left = torch.randn(4, 8, 22, generator=generator)
    right = torch.randn(4, 8, 22, generator=generator)
    first = rung.assignment(left, right)
    second = rung.assignment(right, left)
    assert first.shape == (4, 253, 8)
    assert torch.allclose(first.sum(-1), torch.ones(4, 253), atol=1e-6)
    assert torch.allclose(first, second, atol=1e-6)


def test_prediction_dimensions_and_identity_matching():
    gates = torch.full((4, 253, 8), 1 / 8)
    responses = torch.randn(8, 34, generator=torch.Generator().manual_seed(2))
    prediction = rung.coupled_prediction(gates, responses)
    assert prediction.shape == (4, 253, 34)
    assert rung.best_permutation(responses, responses) == tuple(range(8))


def test_loss_rejects_bad_scale():
    gates = torch.full((4, 253, 8), 1 / 8)
    responses = torch.zeros(8, 34)
    prediction = rung.coupled_prediction(gates, responses)
    try:
        rung.standardized_loss(prediction, prediction, gates, torch.zeros(34))
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("zero scale was accepted")


def test_registered_maximum_forward_price():
    singleton_phase = 62 * (1 + 4 * (1 + 253))
    atom_phase = 62 * 4 * 8
    pair_confirmation = 62 * 4 * (8 * 7 // 2)
    assert singleton_phase == 63054
    assert atom_phase == 1984
    assert 2 * (singleton_phase + atom_phase) + pair_confirmation == 137020
