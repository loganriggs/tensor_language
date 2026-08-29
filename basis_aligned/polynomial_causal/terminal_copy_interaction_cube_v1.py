"""Pure CPU contract for the prospective four-head copy interaction cube.

This module defines the Boolean cube and mathematical reductions only.  Importing it
does not authorize rows, checkpoint access, model forwards, or outcome publication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import factorial_causal_attribution as factorial


HEADS = ("L5H5", "L7H3", "L8H3", "L8H4")
CELLS = ("positive", "matched_negative", "off_target")
SCALES = (0.25, 0.5, 0.75, 1.0)


def arms() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(arm)
        for size in range(len(HEADS) + 1)
        for arm in combinations(HEADS, size)
    )


def candidate_name(arm: Sequence[str]) -> str:
    canonical = factorial.canonical_arm(HEADS, arm)
    return "native" if not canonical else "+".join(canonical)


def candidate_names() -> tuple[str, ...]:
    return tuple(candidate_name(arm) for arm in arms())


def missing_from_e4() -> tuple[tuple[str, ...], ...]:
    """The ten pair/triple arms absent from the completed E4 screen."""

    return tuple(arm for arm in arms() if len(arm) in (2, 3))


def validate_effect_cube(values: Mapping[tuple[str, ...], float]) -> dict:
    cube = factorial.validate_cube(HEADS, values)
    if abs(cube[()]) > 1e-12:
        raise ValueError("native-baseline effect must be zero")
    return cube


def analyze_effect_cube(values: Mapping[tuple[str, ...], float]) -> dict:
    """Return exact signed interactions plus order-truncated full-arm predictions."""

    cube = validate_effect_cube(values)
    coefficients = factorial.mobius_coefficients(HEADS, cube)
    full = tuple(HEADS)
    singleton_sum = sum(cube[(head,)] for head in HEADS)
    excess = cube[full] - singleton_sum
    by_order = {
        order: sum(value for arm, value in coefficients.items() if len(arm) == order)
        for order in range(1, len(HEADS) + 1)
    }
    truncated = {
        order: sum(value for arm, value in coefficients.items() if len(arm) <= order)
        for order in range(1, len(HEADS) + 1)
    }
    reconstructed = factorial.reconstruct_from_mobius(HEADS, coefficients)
    if any(abs(reconstructed[arm] - cube[arm]) > 1e-10 for arm in cube):
        raise RuntimeError("Möbius transform failed exact reconstruction")
    return {
        "full_effect": cube[full],
        "singleton_sum": singleton_sum,
        "full_minus_singleton_sum": excess,
        "mobius_by_order": by_order,
        "full_prediction_through_order": truncated,
        "mobius": coefficients,
        "shapley": factorial.shapley_from_mobius(HEADS, coefficients),
    }


def analyze_scaled_full_curve(values: Mapping[float, float]) -> dict:
    """Describe deviation from a linear response to intervention amplitude.

    Values are effects relative to native at the four frozen positive amplitudes.
    The alpha=1 effect defines the linear secant.  This is descriptive; a nonlinear
    curve can arise from RMSNorm, attention, later MLPs, or output softmax geometry.
    """

    observed = {float(scale): float(value) for scale, value in values.items()}
    if set(observed) != set(SCALES):
        raise ValueError("scaled-full curve must contain exactly the frozen amplitudes")
    full = observed[1.0]
    linear = {scale: scale * full for scale in SCALES}
    residual = {scale: observed[scale] - linear[scale] for scale in SCALES}
    return {
        "observed": observed,
        "linear_secant_prediction": linear,
        "nonlinear_residual": residual,
        "max_abs_nonlinear_residual": max(abs(value) for value in residual.values()),
    }
