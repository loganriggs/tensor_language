import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorial_causal_attribution import (
    analyze_cells,
    analyze_cube,
    mobius_coefficients,
    powerset,
    reconstruct_from_mobius,
)


GROUPS = ("attention", "mlp012", "deep")


def cube_from_terms(terms):
    return {
        arm: sum(value for term, value in terms.items() if set(term).issubset(arm))
        for arm in powerset(GROUPS)
    }


def test_mobius_round_trip_recovers_signed_interactions():
    terms = {
        (): 2.9,
        ("attention",): 0.1,
        ("mlp012",): 0.3,
        ("deep",): 0.2,
        ("attention", "mlp012"): 0.15,
        ("attention", "mlp012", "deep"): -0.04,
    }
    cube = cube_from_terms(terms)
    mobius = mobius_coefficients(GROUPS, cube)
    for arm in powerset(GROUPS):
        assert mobius[arm] == pytest.approx(terms.get(arm, 0.0))
    assert reconstruct_from_mobius(GROUPS, mobius) == pytest.approx(cube)


def test_shapley_splits_each_interaction_and_closes():
    terms = {
        (): 3.0,
        ("attention",): 1.0,
        ("mlp012",): 2.0,
        ("attention", "mlp012"): 6.0,
        ("attention", "mlp012", "deep"): 3.0,
    }
    result = analyze_cube(GROUPS, cube_from_terms(terms))
    assert result["shapley"]["attention"] == pytest.approx(1 + 3 + 1)
    assert result["shapley"]["mlp012"] == pytest.approx(2 + 3 + 1)
    assert result["shapley"]["deep"] == pytest.approx(1)
    assert result["shapley_closure_error"] == pytest.approx(0.0)
    assert result["interaction_l1"] == pytest.approx(9.0)


def test_cell_aggregation_uses_counts_not_mean_of_means():
    additive_a = {(): 1.0, ("attention",): 2.0}
    additive_b = {(): 4.0, ("attention",): 2.0}
    groups = ("attention",)
    result = analyze_cells(
        groups,
        {"rare": 1, "frequent": 3},
        {
            "rare": {arm: sum(value for term, value in additive_a.items() if set(term).issubset(arm))
                     for arm in powerset(groups)},
            "frequent": {arm: sum(value for term, value in additive_b.items() if set(term).issubset(arm))
                         for arm in powerset(groups)},
        },
    )
    assert result["weighted_total_effect"] == pytest.approx(2.0)
    assert result["cell_damage_shares"] == pytest.approx({"rare": 0.25, "frequent": 0.75})
    assert math.isclose(result["weighted_shapley_closure_error"], 0.0, abs_tol=1e-12)


def test_incomplete_cube_is_rejected():
    with pytest.raises(ValueError, match="cube mismatch"):
        analyze_cube(GROUPS, {(): 0.0, ("attention",): 1.0})
