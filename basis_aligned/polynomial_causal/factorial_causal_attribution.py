"""Exact Boolean-cube attribution for replacement-group causal audits.

For a set function v(S), where S is the set of replacement groups installed and
v(S) is a declared loss or response metric, the Mobius coefficients isolate every
interaction.  Dividing each interaction equally among its members gives the Shapley
allocation.  This module is CPU-only and deliberately keeps raw signed effects;
negative contributions and interactions must not be clipped into a flattering pie.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations


Arm = tuple[str, ...]


def canonical_arm(groups: Sequence[str], arm: Sequence[str]) -> Arm:
    chosen = set(arm)
    unknown = chosen.difference(groups)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    return tuple(group for group in groups if group in chosen)


def powerset(groups: Sequence[str]) -> list[Arm]:
    return [tuple(arm) for size in range(len(groups) + 1)
            for arm in combinations(groups, size)]


def validate_cube(groups: Sequence[str], values: Mapping[Arm, float]) -> dict[Arm, float]:
    if len(set(groups)) != len(groups):
        raise ValueError("group names must be unique")
    normalized = {canonical_arm(groups, arm): float(value) for arm, value in values.items()}
    expected = set(powerset(groups))
    missing = expected.difference(normalized)
    extra = set(normalized).difference(expected)
    if missing or extra or len(normalized) != len(values):
        raise ValueError(f"factorial cube mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return normalized


def mobius_coefficients(groups: Sequence[str], values: Mapping[Arm, float]) -> dict[Arm, float]:
    """Return m(T) such that v(S) = sum_{T subset S} m(T)."""
    cube = validate_cube(groups, values)
    coefficients: dict[Arm, float] = {}
    for arm in powerset(groups):
        total = 0.0
        for subset in powerset(arm):
            total += (-1.0) ** (len(arm) - len(subset)) * cube[canonical_arm(groups, subset)]
        coefficients[arm] = total
    return coefficients


def reconstruct_from_mobius(groups: Sequence[str], coefficients: Mapping[Arm, float]) -> dict[Arm, float]:
    mobius = validate_cube(groups, coefficients)
    return {
        arm: sum(mobius[canonical_arm(groups, subset)] for subset in powerset(arm))
        for arm in powerset(groups)
    }


def shapley_from_mobius(groups: Sequence[str], coefficients: Mapping[Arm, float]) -> dict[str, float]:
    mobius = validate_cube(groups, coefficients)
    return {
        group: sum(value / len(arm) for arm, value in mobius.items()
                   if arm and group in arm)
        for group in groups
    }


def analyze_cube(groups: Sequence[str], values: Mapping[Arm, float]) -> dict:
    cube = validate_cube(groups, values)
    mobius = mobius_coefficients(groups, cube)
    shapley = shapley_from_mobius(groups, mobius)
    empty = cube[()]
    full = cube[tuple(groups)]
    total = full - empty
    interaction_l1 = sum(abs(value) for arm, value in mobius.items() if len(arm) >= 2)
    return {
        "baseline": empty,
        "full": full,
        "total_effect": total,
        "mobius": {"+".join(arm) if arm else "baseline": value for arm, value in mobius.items()},
        "shapley": shapley,
        "shapley_closure_error": sum(shapley.values()) - total,
        "interaction_l1": interaction_l1,
        "interaction_l1_fraction_of_total": interaction_l1 / max(abs(total), 1e-30),
    }


def analyze_cells(
    groups: Sequence[str],
    cell_counts: Mapping[str, int],
    cell_values: Mapping[str, Mapping[Arm, float]],
) -> dict:
    """Analyze per-cell mean metrics and close their weighted full-arm total.

    Cell values must be mean loss/response metrics, not already weighted sums.
    Counts define the only permitted aggregation denominator.
    """
    if set(cell_counts) != set(cell_values):
        raise ValueError("cell counts and values must name the same cells")
    if any(count <= 0 for count in cell_counts.values()):
        raise ValueError("cell counts must be positive")
    cells = {cell: analyze_cube(groups, values) for cell, values in cell_values.items()}
    total_n = sum(cell_counts.values())
    weighted_total = sum(cell_counts[cell] * row["total_effect"] for cell, row in cells.items()) / total_n
    damage_numerators = {
        cell: cell_counts[cell] * row["total_effect"] for cell, row in cells.items()
    }
    denominator = sum(damage_numerators.values())
    shares = {cell: value / denominator if denominator else None
              for cell, value in damage_numerators.items()}
    weighted_shapley = {
        group: sum(cell_counts[cell] * row["shapley"][group] for cell, row in cells.items()) / total_n
        for group in groups
    }
    return {
        "groups": list(groups),
        "cell_counts": dict(cell_counts),
        "cells": cells,
        "weighted_total_effect": weighted_total,
        "cell_damage_shares": shares,
        "weighted_shapley": weighted_shapley,
        "weighted_shapley_closure_error": sum(weighted_shapley.values()) - weighted_total,
    }
