import torch
import pytest

import compilation_mask_cut_rank_v1 as cut


def _observations(cost: torch.Tensor, ce_cost: torch.Tensor | None = None):
    if ce_cost is None:
        ce_cost = cost / 25.0
    if tuple(cost.shape) != (8, 8) or tuple(ce_cost.shape) != (8, 8):
        raise ValueError("synthetic costs must be 8 x 8")
    return {
        (i, j): cut.ObservedCell(
            top1_accuracy=0.90 - float(cost[i, j]) / 100.0,
            mean_ce=0.50 + float(ce_cost[i, j]),
        )
        for i in range(8) for j in range(8)
    }


def _low_rank_cost(rank: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(7, rank, generator=generator, dtype=torch.float64)
    right = torch.randn(7, rank, generator=generator, dtype=torch.float64)
    prefix = torch.linspace(0.0, 4.0, 8, dtype=torch.float64)
    suffix = torch.linspace(0.0, 3.0, 8, dtype=torch.float64)
    cost = prefix[:, None] + suffix[None, :]
    cost[1:, 1:] += 2.0 * left @ right.T
    return cost


def _develop(observations):
    return cut.prepare_development({
        cell: observations[cell] for cell in cut.DEVELOPMENT_CELLS
    })


def _finalize(development, observations):
    return cut.finalize_heldout(
        development,
        {cell: observations[cell] for cell in cut.HELDOUT_CELLS},
    )


def test_registry_is_complete_disjoint_connected_and_challenge_balanced():
    cut.validate_registry()
    assert len(cut.ANCHOR_CELLS) == 15
    assert len(cut.TRAIN_CELLS) == 28
    assert len(cut.VALIDATION_CELLS) == 10
    assert len(cut.HELDOUT_CELLS) == 11
    assert cut.inhomogeneous_tt_parameter_count(1) == 52
    assert cut.inhomogeneous_tt_parameter_count(2) == 192
    assert cut.inhomogeneous_tt_parameter_count(4) == 736


def test_anchoring_removes_all_additive_prefix_and_suffix_effects():
    prefix = torch.arange(8, dtype=torch.float64)[:, None]
    suffix = torch.arange(8, dtype=torch.float64)[None, :].square()
    interaction = cut.anchored_interaction(3.0 + prefix + suffix)
    assert torch.equal(interaction, torch.zeros_like(interaction))
    assert cut.spectral_tail_nre(interaction, 0) == 0.0


def test_rank_two_cut_passes_and_rank_three_cut_falsifies_rank_two():
    generator = torch.Generator().manual_seed(20260828)
    left = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    left[0].zero_()
    right[:, 0].zero_()
    rank_two = left[:, :2] @ right[:2]
    rank_three = left @ right
    assert cut.spectral_tail_nre(cut.anchored_interaction(rank_two), 2) < 1e-12
    assert cut.spectral_tail_nre(cut.anchored_interaction(rank_three), 2) > 1e-3


def test_cost_construction_uses_b0_and_opposite_top1_ce_signs():
    values = {
        (0, 0): cut.ObservedCell(0.8, 0.5),
        (1, 0): cut.ObservedCell(0.75, 0.7),
    }
    costs = cut.observed_costs(values)
    assert costs.top1_pp[(0, 0)] == 0.0
    assert costs.ce_nats[(0, 0)] == 0.0
    assert costs.top1_pp[(1, 0)] == pytest.approx(5.0)
    assert costs.ce_nats[(1, 0)] == pytest.approx(0.2)


def test_development_rejects_any_heldout_key_and_final_requires_exact_boundary():
    observations = _observations(_low_rank_cost(1))
    malformed = {cell: observations[cell] for cell in cut.DEVELOPMENT_CELLS}
    malformed[cut.HELDOUT_CELLS[0]] = observations[cut.HELDOUT_CELLS[0]]
    with pytest.raises(ValueError, match="exactly the frozen cells"):
        cut.prepare_development(malformed)
    development = _develop(observations)
    heldout = {cell: observations[cell] for cell in cut.HELDOUT_CELLS}
    heldout.pop(cut.HELDOUT_CELLS[-1])
    with pytest.raises(ValueError, match="exactly the frozen cells"):
        cut.finalize_heldout(development, heldout)


def test_train_scale_is_invariant_to_validation_outcomes():
    observations = _observations(_low_rank_cost(1))
    changed = dict(observations)
    for index, cell in enumerate(cut.VALIDATION_CELLS):
        value = changed[cell]
        changed[cell] = cut.ObservedCell(
            value.top1_accuracy - 0.001 * (index + 1),
            value.mean_ce + 0.001 * (index + 1),
        )
    first = _develop(observations)
    second = _develop(changed)
    assert first.top1_summary.train_scale == second.top1_summary.train_scale
    assert first.ce_summary.train_scale == second.ce_summary.train_scale


def test_rank_one_known_answer_recovers_heldout_interactions():
    observations = _observations(_low_rank_cost(1, seed=11))
    development = _develop(observations)
    result = _finalize(development, observations)
    # The frozen positive ridge grid can make rank 2 win validation by a tiny
    # regularization-bias margin even when the truth is rank 1.  Recovery, not a
    # post-hoc rank tie, is the known answer required here.
    assert development.top1_summary.selected_rank in {1, 2}
    assert min(
        value.validation_rmse
        for value in development.top1_summary.candidates if value.rank == 1
    ) < 0.02
    assert development.top1_summary.nontrivial_train_interaction is True
    assert result.promotive_decision is None
    assert result.bootstrap_complete is False
    assert result.top1.rank_signal_present is True
    assert result.top1.heldout_r2 is not None and result.top1.heldout_r2 > 0.99
    assert result.top1.interaction_nre is not None
    assert result.top1.interaction_nre < 0.10
    assert result.top1.full_grid_rank2_spectral_tail_nre < 1e-12


def test_rank_two_known_answer_beats_additive_baseline_without_promotion():
    observations = _observations(_low_rank_cost(2, seed=7))
    development = _develop(observations)
    result = _finalize(development, observations)
    assert development.top1_summary.selected_rank == 2
    assert result.top1.total_rmse < 0.1
    assert result.top1.interaction_nre is not None
    assert result.top1.interaction_nre < 0.02
    assert result.top1.rmse_ratio is not None and result.top1.rmse_ratio < 0.02
    assert result.ce.heldout_r2 is not None and result.ce.heldout_r2 > 0.999
    assert result.promotive_decision is None


def test_additive_case_is_typed_as_no_rank_signal_not_a_low_rank_success():
    prefix = torch.arange(8, dtype=torch.float64)[:, None] / 2
    suffix = torch.arange(8, dtype=torch.float64)[None, :] / 3
    observations = _observations(prefix + suffix)
    development = _develop(observations)
    result = _finalize(development, observations)
    assert development.top1_summary.selected_rank == 0
    assert development.top1_summary.nontrivial_train_interaction is False
    assert result.top1.rank_signal_present is False
    assert result.top1.interaction_nre is None
    assert result.promotive_decision is None


def test_high_rank_interaction_fails_the_rank_two_spectral_diagnostic():
    interaction = torch.eye(7, dtype=torch.float64)
    prefix = torch.linspace(0.0, 2.0, 8, dtype=torch.float64)
    suffix = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    cost = prefix[:, None] + suffix[None, :]
    cost[1:, 1:] += 2.0 * interaction
    observations = _observations(cost)
    result = _finalize(_develop(observations), observations)
    assert result.top1.full_grid_rank2_spectral_tail_nre > 0.50
    assert (
        result.top1.interaction_nre is None
        or result.top1.interaction_nre > 0.50
    )


def test_singleton_baseline_requires_exact_source_bound_currency():
    sites = {
        (kind, layer): 0.1
        for layer in range(1, 18) for kind in ("attn", "mlp")
    }
    frozen = cut.FrozenSingletonCosts(
        target="top1_pp", costs=sites, source_sha256="a" * 64,
    )
    observations = _observations(_low_rank_cost(1))
    development = cut.prepare_development(
        {cell: observations[cell] for cell in cut.DEVELOPMENT_CELLS},
        singleton_top1_pp=frozen,
    )
    assert development.top1_summary.singleton_baseline_source_sha256 == "a" * 64
    assert {
        value.name for value in development.top1_summary.baselines
    } >= {
        "literal_singleton_sum", "s1834_scaled_singleton_sum",
    }
    with pytest.raises(ValueError, match="target currency"):
        cut.prepare_development(
            {cell: observations[cell] for cell in cut.DEVELOPMENT_CELLS},
            singleton_ce_nats=frozen,
        )
