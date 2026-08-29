"""Finite-sample risk bounds for a token-gated simplified program.

Each row is an independent source document; positions inside a document may be
arbitrarily dependent.  For every predeclared confidence threshold we bound both the
unconditional mass of accepted errors and the unconditional mass of accepted tokens.
Their ratio bounds conditional error among accepted tokens.  A union bound over both
quantities and every threshold makes subsequent threshold selection valid.

This module is pure CPU mathematics.  It does not load bilin18, rows, or outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class SelectiveBound:
    threshold: float
    documents: int
    positions_per_document: int
    accepted_positions: int
    valid_positions: int
    empirical_coverage: float
    empirical_conditional_risk: float
    accepted_mass_lcb: float
    accepted_error_mass_ucb: float
    conditional_risk_ucb: float
    simultaneous_radius: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim != 2 or tensor.shape[0] < 2 or tensor.shape[1] < 1:
        raise ValueError(f"{name} must have shape [at least 2 documents, positions]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains a non-finite value")
    return tensor


def simultaneous_bounds(
    scores: torch.Tensor,
    losses: torch.Tensor,
    thresholds: Sequence[float],
    *,
    valid: torch.Tensor | None = None,
    delta: float = 0.05,
) -> tuple[SelectiveBound, ...]:
    """Bound risk for every fixed threshold under independent documents.

    ``scores`` increase with trust in the simplified program. ``losses`` must lie in
    [0, 1]; examples include top-1 disagreement, task error, or a preregistered clipped
    and rescaled CE excess. ``valid`` may mask unsupported positions, but every
    document retains the same fixed denominator (the matrix width).

    If e_d and a_d are document-level accepted-error and acceptance proportions, then
    target selective risk is E[e_d] / E[a_d]. Hoeffding plus a 2K union bound yields
    simultaneous upper/lower bounds for K thresholds without assuming independent
    token positions or monotonic empirical risk.
    """

    score = _as_matrix(scores, "scores").double()
    loss = _as_matrix(losses, "losses").double()
    if score.shape != loss.shape:
        raise ValueError("scores and losses must have identical shape")
    if bool(((loss < 0) | (loss > 1)).any()):
        raise ValueError("losses must lie in [0, 1]")
    mask = torch.ones_like(score, dtype=torch.bool) if valid is None else torch.as_tensor(
        valid
    ).detach().cpu().bool()
    if mask.shape != score.shape:
        raise ValueError("valid must have the same shape as scores")
    grid = tuple(float(value) for value in thresholds)
    if not grid or any(not math.isfinite(value) for value in grid):
        raise ValueError("thresholds must be a nonempty finite sequence")
    if len(set(grid)) != len(grid):
        raise ValueError("thresholds must not repeat")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")

    n_documents, width = score.shape
    radius = math.sqrt(math.log(2.0 * len(grid) / delta) / (2.0 * n_documents))
    output: list[SelectiveBound] = []
    for threshold in grid:
        accepted = mask & (score >= threshold)
        accepted_error = accepted.double() * loss
        accepted_per_document = accepted.double().mean(dim=1)
        error_per_document = accepted_error.mean(dim=1)
        mean_acceptance = float(accepted_per_document.mean())
        mean_error = float(error_per_document.mean())
        acceptance_lcb = max(0.0, mean_acceptance - radius)
        error_ucb = min(1.0, mean_error + radius)
        conditional_ucb = (
            min(1.0, error_ucb / acceptance_lcb)
            if acceptance_lcb > 0 else math.inf
        )
        accepted_count = int(accepted.sum())
        error_sum = float(accepted_error.sum())
        output.append(SelectiveBound(
            threshold=threshold,
            documents=n_documents,
            positions_per_document=width,
            accepted_positions=accepted_count,
            valid_positions=int(mask.sum()),
            empirical_coverage=(accepted_count / max(int(mask.sum()), 1)),
            empirical_conditional_risk=(error_sum / accepted_count if accepted_count else math.inf),
            accepted_mass_lcb=acceptance_lcb,
            accepted_error_mass_ucb=error_ucb,
            conditional_risk_ucb=conditional_ucb,
            simultaneous_radius=radius,
        ))
    return tuple(output)


def select_max_coverage(
    bounds: Iterable[SelectiveBound],
    *,
    maximum_risk: float,
    minimum_accepted_mass: float = 0.0,
) -> SelectiveBound | None:
    """Select the highest certified coverage; return None when every gate fails."""

    if not 0 <= maximum_risk <= 1 or not 0 <= minimum_accepted_mass <= 1:
        raise ValueError("risk and accepted-mass gates must lie in [0, 1]")
    eligible = [
        item for item in bounds
        if item.conditional_risk_ucb <= maximum_risk
        and item.accepted_mass_lcb >= minimum_accepted_mass
    ]
    if not eligible:
        return None
    # Coverage is primary. A stricter threshold is the deterministic tie-breaker.
    return max(eligible, key=lambda item: (item.accepted_mass_lcb, item.threshold))
