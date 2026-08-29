import math

import pytest
import torch

import selective_compilation_risk as risk


def test_document_cluster_bound_matches_hand_calculation():
    scores = torch.tensor([[0.1, 0.6, 0.9], [0.2, 0.7, 0.8], [0.3, 0.4, 1.0]])
    losses = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
    (bound,) = risk.simultaneous_bounds(scores, losses, (0.5,), delta=0.2)
    radius = math.sqrt(math.log(10.0) / 6.0)
    assert bound.accepted_positions == 5
    assert bound.empirical_coverage == pytest.approx(5 / 9)
    assert bound.empirical_conditional_risk == pytest.approx(1 / 5)
    assert bound.accepted_mass_lcb == pytest.approx(max(0.0, 5 / 9 - radius))
    assert bound.accepted_error_mass_ucb == pytest.approx(min(1.0, 1 / 9 + radius))


def test_within_document_duplication_does_not_fake_more_independent_samples():
    scores = torch.ones(20, 1)
    losses = torch.zeros(20, 1)
    narrow = risk.simultaneous_bounds(scores, losses, (0.0,))[0]
    duplicated = risk.simultaneous_bounds(
        scores.repeat(1, 1000), losses.repeat(1, 1000), (0.0,),
    )[0]
    assert duplicated.simultaneous_radius == narrow.simultaneous_radius


def test_simultaneous_penalty_grows_with_threshold_search():
    scores = torch.ones(40, 4)
    losses = torch.zeros(40, 4)
    one = risk.simultaneous_bounds(scores, losses, (0.0,))[0]
    many = risk.simultaneous_bounds(scores, losses, tuple(range(20)))[0]
    assert many.simultaneous_radius > one.simultaneous_radius


def test_ratio_bound_is_fail_closed_when_acceptance_lcb_is_zero():
    bound = risk.simultaneous_bounds(
        torch.zeros(10, 2), torch.zeros(10, 2), (1.0,),
    )[0]
    assert bound.accepted_positions == 0
    assert math.isinf(bound.conditional_risk_ucb)
    assert risk.select_max_coverage((bound,), maximum_risk=1.0) is None


def test_selector_uses_certified_not_empirical_coverage_and_strict_tie_break():
    common = dict(
        documents=100, positions_per_document=10, valid_positions=1000,
        empirical_conditional_risk=0.01, accepted_error_mass_ucb=0.01,
        conditional_risk_ucb=0.04, simultaneous_radius=0.02,
    )
    lower = risk.SelectiveBound(
        threshold=0.3, accepted_positions=800, empirical_coverage=0.8,
        accepted_mass_lcb=0.7, **common,
    )
    stricter = risk.SelectiveBound(
        threshold=0.7, accepted_positions=750, empirical_coverage=0.75,
        accepted_mass_lcb=0.7, **common,
    )
    assert risk.select_max_coverage(
        (lower, stricter), maximum_risk=0.05,
    ) is stricter


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_invalid_loss_is_rejected(bad):
    loss = torch.zeros(3, 2)
    loss[0, 0] = bad
    with pytest.raises(ValueError):
        risk.simultaneous_bounds(torch.ones(3, 2), loss, (0.0,))


def test_masked_positions_never_count_as_accepted_or_errors():
    scores = torch.ones(4, 3)
    losses = torch.ones(4, 3)
    valid = torch.tensor([[1, 0, 0]] * 4, dtype=torch.bool)
    bound = risk.simultaneous_bounds(scores, losses, (0.0,), valid=valid)[0]
    assert bound.accepted_positions == 4
    assert bound.valid_positions == 4
    assert bound.empirical_conditional_risk == 1.0
