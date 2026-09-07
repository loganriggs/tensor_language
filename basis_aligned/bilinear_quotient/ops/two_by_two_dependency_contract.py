#!/usr/bin/env python3
"""Pure rowwise accounting for a two-factor causal dependency experiment.

Cell names encode ``upstream,downstream`` bits.  The helper operates on a
linear outcome such as answer-minus-foil margin.  Nonlinear outcomes such as
KL divergence must be reported per cell and are intentionally not accepted.
"""
from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence


CELLS = ("00", "01", "10", "11")


class DependencyFactorialError(ValueError):
    """The four cells cannot support exact rowwise factorial accounting."""


def _vector(value: object, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DependencyFactorialError(f"{label} must be a numeric sequence")
    result = []
    for index, item in enumerate(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise DependencyFactorialError(f"{label}[{index}] is not numeric") from error
        if not math.isfinite(number):
            raise DependencyFactorialError(f"{label}[{index}] is not finite")
        result.append(number)
    if not result:
        raise DependencyFactorialError(f"{label} is empty")
    return result


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "mean_absolute": statistics.fmean(abs(value) for value in values),
        "max_absolute": max(abs(value) for value in values),
    }


def decompose_dependency_factorial(
    cells: Mapping[str, Sequence[float]], *, base_zero_tolerance: float = 1e-6,
    closure_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Return main, conditional, and interaction responses for exact 2x2 cells.

    ``00`` is native/no intervention, ``01`` downstream only, ``10`` upstream
    only with the downstream computation live, and ``11`` both interventions.
    Inputs are changes from one shared native baseline, so ``00`` must be zero.
    """
    if set(cells) != set(CELLS):
        raise DependencyFactorialError(f"expected exactly cells {CELLS}")
    try:
        zero_bar, closure_bar = float(base_zero_tolerance), float(closure_tolerance)
    except (TypeError, ValueError) as error:
        raise DependencyFactorialError("tolerances must be numeric") from error
    if (not math.isfinite(zero_bar) or zero_bar < 0 or not math.isfinite(closure_bar)
            or closure_bar < 0):
        raise DependencyFactorialError("tolerances must be finite and nonnegative")

    values = {cell: _vector(cells[cell], cell) for cell in CELLS}
    sizes = {len(vector) for vector in values.values()}
    if len(sizes) != 1:
        raise DependencyFactorialError("all cells must contain the same rows")

    c00, c01, c10, c11 = (values[cell] for cell in CELLS)
    upstream_live = [x10 - x00 for x00, x10 in zip(c00, c10)]
    attention_without_upstream = [x01 - x00 for x00, x01 in zip(c00, c01)]
    upstream_with_fixed_attention = [x11 - x01 for x01, x11 in zip(c01, c11)]
    attention_with_upstream = [x11 - x10 for x10, x11 in zip(c10, c11)]
    interaction = [x11 - x10 - x01 + x00
                   for x00, x01, x10, x11 in zip(c00, c01, c10, c11)]
    reconstruction = [x00 + upstream + attention + joint
                      for x00, upstream, attention, joint
                      in zip(c00, upstream_live, attention_without_upstream, interaction)]
    closure_error = max(abs(observed - rebuilt)
                        for observed, rebuilt in zip(c11, reconstruction))
    base_error = max(abs(value) for value in c00)
    return {
        "cell_semantics": {
            "00": "no upstream projector; live attention-15",
            "01": "no upstream projector; complete attention-15 donor clamp",
            "10": "upstream projector; live attention-15",
            "11": "upstream projector; complete attention-15 donor clamp",
        },
        "components": {
            "upstream_live_attention": upstream_live,
            "attention15_without_upstream": attention_without_upstream,
            "upstream_with_fixed_attention15": upstream_with_fixed_attention,
            "attention15_with_upstream": attention_with_upstream,
            "mobius_interaction": interaction,
        },
        "summaries": {
            name: _summary(vector) for name, vector in {
                "upstream_live_attention": upstream_live,
                "attention15_without_upstream": attention_without_upstream,
                "upstream_with_fixed_attention15": upstream_with_fixed_attention,
                "attention15_with_upstream": attention_with_upstream,
                "mobius_interaction": interaction,
            }.items()
        },
        "base_zero_max_abs_error": base_error,
        "factorial_closure_max_abs_error": closure_error,
        "base_zero_pass": base_error <= zero_bar,
        "factorial_closure_pass": closure_error <= closure_bar,
    }


def vector_metrics(response: Sequence[float], target: Sequence[float]) -> dict[str, float | int]:
    """Score a linear response against a nonzero registered target vector."""
    observed, wanted = _vector(response, "response"), _vector(target, "target")
    if len(observed) != len(wanted):
        raise DependencyFactorialError("response and target must contain the same rows")
    xx = sum(value * value for value in observed)
    yy = sum(value * value for value in wanted)
    xy = sum(left * right for left, right in zip(observed, wanted))
    if yy == 0:
        raise DependencyFactorialError("target vector has zero norm")
    return {
        "count": len(observed),
        "signed_projection": xy / yy,
        "relative_squared_error": sum((left - right) ** 2
                                      for left, right in zip(observed, wanted)) / yy,
        "cosine": xy / math.sqrt(xx * yy) if xx else 0.0,
        "norm_ratio": math.sqrt(xx / yy),
        "direction_fraction": sum(left * right > 0 for left, right
                                  in zip(observed, wanted)) / len(observed),
    }
