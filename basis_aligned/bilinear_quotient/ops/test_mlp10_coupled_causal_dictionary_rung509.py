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


def test_corrected_addendum_price_includes_two_pair_phases_and_captures():
    exact_phase = 62 * (1 + 1 + 4 * (1 + 253))
    atom_phase = 62 * (1 + 4 * (1 + 8))
    pair_phase = 62 * (1 + 4 * (1 + 28))
    assert exact_phase == 63116
    assert atom_phase == 2294
    assert 2 * (exact_phase + atom_phase) + 2 * pair_phase == 145328


def test_weighted_hidden_equals_explicit_unordered_sum():
    generator = torch.Generator().manual_seed(3)
    factors = {
        "left": torch.randn(2, 3, 22, 7, generator=generator),
        "right": torch.randn(2, 3, 22, 7, generator=generator),
    }
    weights = torch.rand(253, generator=generator)
    actual = rung._weighted_hidden(factors, weights)
    expected = sum(
        weights[index] * rung.parent._pair_hidden(factors, index)
        for index in range(253)
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_align_fits_recovers_atom_permutation_without_seed_selection():
    generator = torch.Generator().manual_seed(4)
    responses = torch.randn(8, 34, generator=generator)
    assignments = torch.softmax(torch.randn(4, 253, 8, generator=generator), -1)
    permutation = torch.tensor([3, 5, 0, 7, 1, 6, 2, 4])
    fits = []
    for index in range(6):
        if index == 0:
            order = torch.arange(8)
        else:
            order = permutation
        fits.append({
            "seed": 5090 + index % 3,
            "responses": responses[order],
            "assignments": assignments[:, :, order],
            "archetype_weights": torch.eye(8)[order],
            "anchor_indices": torch.arange(8)[order],
            "anchor_weights": torch.ones(8)[order],
            "scale": torch.ones(34),
            "loss": 0.0,
        })
    aligned, mean_gates, mean_responses, mean_scale = rung.align_fits(fits)
    for fit in aligned:
        torch.testing.assert_close(fit["responses"], responses)
        torch.testing.assert_close(fit["assignments"], assignments)
    torch.testing.assert_close(mean_gates, assignments)
    torch.testing.assert_close(mean_responses, responses)
    torch.testing.assert_close(mean_scale, torch.ones(34))


def test_heldout_forecast_passes_exact_planted_dictionary():
    generator = torch.Generator().manual_seed(5)
    gates = torch.softmax(torch.randn(4, 253, 8, generator=generator), -1)
    responses = torch.randn(8, 34, generator=generator)
    target = rung.coupled_prediction(gates, responses)
    row = rung.heldout_forecast(
        gates, responses, torch.ones(34), target, target)
    assert row["standardized_mse"] == 0.0
    assert row["holds"] is True


def test_frozen_archetypal_fit_fails_planted_identifiability_gate():
    audit = rung.synthetic_identifiability_audit()
    assert audit["model_loaded"] is False
    assert audit["model_outcomes_opened"] is False
    assert audit["cpu_fits"] == 6
    assert len(set(audit["expected_anchor_indices"])) == 8
    assert min(audit["expected_anchor_assignment_weights"]) > .99
    assert audit["summary"]["minimum_response_cosine"] < .90
    assert audit["summary"]["minimum_assignment_cosine"] < .80
    assert audit["summary"]["anchor_identity_matches"] < 48
    assert audit["summary"]["minimum_anchor_weight"] < .90
    assert audit["holds"] is False
    try:
        rung.require_identifiable_instrument(audit)
    except RuntimeError as error:
        assert "blocked" in str(error)
    else:
        raise AssertionError("failed synthetic identifiability gate did not block model execution")
