from __future__ import annotations

import pytest
import torch

import early_mlp_context_cross_v1 as cross


def test_registry_is_nested_and_partitions_every_cell() -> None:
    cross.validate_registry()
    assert len(cross.RANK3_DISCOVERY_CELLS) == 48
    assert len(cross.RANK4_VALIDATION_CELLS) == 7
    assert len(cross.RANK4_FIT_CELLS) == 55
    assert len(cross.HELDOUT_CELLS) == 9
    assert set(cross.RANK4_FIT_CELLS).isdisjoint(cross.HELDOUT_CELLS)
    assert cross.BOOTSTRAP_SEEDS == {
        "skip7000": 2026082803,
        "skip11000": 2026082804,
    }
    assert cross.BOOTSTRAP_DRAWS == 2_000
    assert cross.ALS_RESTARTS == 8
    assert cross.ALS_SEED == 2026082805


def test_exact_rank_three_interaction_is_recovered_without_heldout_access() -> None:
    generator = torch.Generator().manual_seed(20260828)
    left = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    right = torch.randn((3, 7), generator=generator, dtype=torch.float64)
    interaction = torch.zeros((8, 8), dtype=torch.float64)
    interaction[1:, 1:] = left @ right
    row = torch.randn((8, 1), generator=generator, dtype=torch.float64)
    column = torch.randn((1, 8), generator=generator, dtype=torch.float64)
    cost = row + column + interaction
    observed = {cell: float(cost[cell]) for cell in cross.RANK3_DISCOVERY_CELLS}
    result = cross.cross_prediction(observed, 3)
    assert torch.allclose(result.prediction, cost, atol=1e-10, rtol=1e-10)

    # Even a finite extra cell is a capability violation; poisoned cells cannot be read.
    forbidden = next(iter(set(cross.ALL_CELLS) - set(cross.RANK3_DISCOVERY_CELLS)))
    with pytest.raises(ValueError, match="stage capability"):
        cross.cross_prediction({**observed, forbidden: float("nan")}, 3)


def test_rank_four_prediction_reads_validation_but_not_heldout() -> None:
    generator = torch.Generator().manual_seed(17)
    cost = torch.randn((8, 8), generator=generator, dtype=torch.float64)
    observed = {cell: float(cost[cell]) for cell in cross.RANK4_FIT_CELLS}
    result = cross.cross_prediction(observed, 4)
    assert result.prediction.shape == (8, 8)
    with pytest.raises(ValueError, match="stage capability"):
        cross.cross_prediction({**observed, cross.HELDOUT_CELLS[0]: 0.0}, 4)


def test_singular_pivot_and_bad_scores_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="singular"):
        cross.cross_prediction(
            {cell: 0.0 for cell in cross.RANK4_FIT_CELLS}, 4,
        )
    with pytest.raises(ValueError):
        cross.score_prediction(
            {}, torch.zeros((8, 8)), "unknown",
        )


def test_exact_rank_four_interaction_is_recovered() -> None:
    generator = torch.Generator().manual_seed(2026082806)
    left = torch.randn((7, 4), generator=generator, dtype=torch.float64)
    right = torch.randn((4, 7), generator=generator, dtype=torch.float64)
    interaction = torch.zeros((8, 8), dtype=torch.float64)
    interaction[1:, 1:] = left @ right
    row = torch.randn((8, 1), generator=generator, dtype=torch.float64)
    column = torch.randn((1, 8), generator=generator, dtype=torch.float64)
    cost = row + column + interaction
    observed = {cell: float(cost[cell]) for cell in cross.RANK4_FIT_CELLS}
    result = cross.cross_prediction(observed, 4)
    assert torch.allclose(result.prediction, cost, atol=1e-10, rtol=1e-10)


def test_score_capabilities_reject_missing_extra_and_poisoned_cells() -> None:
    prediction = torch.zeros((8, 8), dtype=torch.float64)
    validation = {cell: 0.0 for cell in cross.RANK4_VALIDATION_CELLS}
    assert cross.score_prediction(
        validation, prediction, "rank3_validation",
    )["rmse"] == 0.0

    missing = dict(validation)
    missing.pop(cross.RANK4_VALIDATION_CELLS[0])
    with pytest.raises(ValueError, match="stage capability"):
        cross.score_prediction(missing, prediction, "rank3_validation")

    extra = {**validation, cross.HELDOUT_CELLS[0]: 0.0}
    with pytest.raises(ValueError, match="stage capability"):
        cross.score_prediction(extra, prediction, "rank3_validation")

    poisoned = dict(validation)
    poisoned[cross.RANK4_VALIDATION_CELLS[0]] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        cross.score_prediction(poisoned, prediction, "rank3_validation")
