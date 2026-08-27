"""CPU scoring for finite causal-response quotient candidates.

The scorer intentionally keeps consumer and background axes separate.  Mean response
distance is diagnostic only: acceptance uses pairwise worst-background distances and
requires every consumer to pass.  Model collection and candidate discovery live in a
separate, authority-bound program.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConsumerGate:
    epsilon_q95: float
    epsilon_max: float
    minimum_separation_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.epsilon_q95 < 0 or self.epsilon_max < 0:
            raise ValueError("consumer tolerances must be nonnegative")
        if self.epsilon_q95 > self.epsilon_max:
            raise ValueError("epsilon_q95 cannot exceed epsilon_max")
        if self.minimum_separation_ratio <= 0:
            raise ValueError("minimum_separation_ratio must be positive")


def score_worst_cell_equivalence(
    effect_sums: Mapping[str, np.ndarray],
    effect_counts: Mapping[str, np.ndarray],
    *,
    margins: Mapping[str, float],
    cell_names: Sequence[str],
    minimum_documents_per_cell: int = 30,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Simultaneous document-bootstrap equivalence test over consumers and cells.

    Inputs are per-document sums and counts with shape ``[document, cell]``.  A
    single bootstrap resample of documents is shared by all consumers, preserving
    their dependence.  Acceptance uses the 95% UCB of the nonlinear maximum
    ``effect / margin``; pooled or marginal averages cannot certify equivalence.
    """

    if not effect_sums:
        raise ValueError("at least one consumer is required")
    if set(effect_sums) != set(effect_counts) or set(effect_sums) != set(margins):
        raise ValueError("sums, counts, and margins must name exactly the same consumers")
    if minimum_documents_per_cell < 1:
        raise ValueError("minimum_documents_per_cell must be positive")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    names = list(cell_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("cell_names must be nonempty and unique")
    n_documents = n_cells = None
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for consumer in effect_sums:
        sums = np.asarray(effect_sums[consumer], dtype=np.float64)
        counts = np.asarray(effect_counts[consumer], dtype=np.float64)
        if sums.ndim != 2 or counts.shape != sums.shape:
            raise ValueError(f"consumer {consumer!r} arrays must share [document, cell] shape")
        if n_documents is None:
            n_documents, n_cells = sums.shape
        elif sums.shape != (n_documents, n_cells):
            raise ValueError("all consumers must share document and cell axes")
        if n_cells != len(names):
            raise ValueError("cell_names length must match the cell axis")
        if (counts < 0).any() or not np.isfinite(counts).all():
            raise ValueError(f"consumer {consumer!r} counts must be finite and nonnegative")
        if not np.isfinite(sums).all():
            raise ValueError(f"consumer {consumer!r} sums must be finite")
        if np.any((counts == 0) & (sums != 0)):
            raise ValueError(f"consumer {consumer!r} has nonzero sums with zero counts")
        margin = float(margins[consumer])
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError(f"consumer {consumer!r} margin must be finite and positive")
        arrays[consumer] = sums, counts

    assert n_documents is not None and n_cells is not None
    if n_documents < 2:
        raise ValueError("at least two documents are required")

    consumer_reports: dict[str, dict[str, object]] = {}
    support_pass = True
    point_max_ratio = -np.inf
    for consumer, (sums, counts) in arrays.items():
        support = (counts > 0).sum(axis=0)
        totals = counts.sum(axis=0)
        effects = np.divide(
            sums.sum(axis=0), totals, out=np.full(n_cells, np.nan), where=totals > 0
        )
        ratios = effects / float(margins[consumer])
        support_ok = support >= minimum_documents_per_cell
        support_pass = support_pass and bool(support_ok.all())
        if np.isfinite(ratios).any():
            point_max_ratio = max(point_max_ratio, float(np.nanmax(ratios)))
        consumer_reports[consumer] = {
            "margin": float(margins[consumer]),
            "cell_effects": {names[i]: float(effects[i]) for i in range(n_cells)},
            "cell_standardized_effects": {
                names[i]: float(ratios[i]) for i in range(n_cells)
            },
            "support_documents": {names[i]: int(support[i]) for i in range(n_cells)},
            "support_passes": {names[i]: bool(support_ok[i]) for i in range(n_cells)},
        }

    rng = np.random.default_rng(seed)
    bootstrap_max = np.full(n_bootstrap, -np.inf, dtype=np.float64)
    # Chunking bounds peak memory when the confirmatory run uses 10k replicates.
    chunk = min(512, n_bootstrap)
    offset = 0
    while offset < n_bootstrap:
        size = min(chunk, n_bootstrap - offset)
        indices = rng.integers(0, n_documents, size=(size, n_documents))
        local_max = np.full(size, -np.inf, dtype=np.float64)
        for consumer, (sums, counts) in arrays.items():
            sampled_sums = sums[indices].sum(axis=1)
            sampled_counts = counts[indices].sum(axis=1)
            effects = np.divide(
                sampled_sums,
                sampled_counts,
                out=np.full_like(sampled_sums, np.inf),
                where=sampled_counts > 0,
            )
            local_max = np.maximum(
                local_max, np.max(effects / float(margins[consumer]), axis=1)
            )
        bootstrap_max[offset : offset + size] = local_max
        offset += size

    simultaneous_ucb = _quantile(bootstrap_max, 0.95)
    passes = bool(support_pass and simultaneous_ucb < 1.0)
    return {
        "schema_version": 1,
        "n_documents": n_documents,
        "n_cells": n_cells,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": seed,
        "minimum_documents_per_cell": minimum_documents_per_cell,
        "consumers": consumer_reports,
        "point_max_standardized_effect": float(point_max_ratio),
        "simultaneous_95pct_ucb_max_standardized_effect": simultaneous_ucb,
        "support_passes": bool(support_pass),
        "equivalence_passes": passes,
        "acceptance_uses_pooled_average": False,
    }


def pointwise_dominates(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> bool:
    """Check the preregistered matched-price pointwise dominance condition."""

    c_consumers = candidate.get("consumers")
    b_consumers = baseline.get("consumers")
    if not isinstance(c_consumers, Mapping) or not isinstance(b_consumers, Mapping):
        raise ValueError("both reports must contain consumer mappings")
    if set(c_consumers) != set(b_consumers):
        raise ValueError("reports must contain the same consumers")
    for name in c_consumers:
        c_effects = c_consumers[name]["cell_standardized_effects"]
        b_effects = b_consumers[name]["cell_standardized_effects"]
        if set(c_effects) != set(b_effects):
            raise ValueError("reports must contain the same cells")
        if any(float(c_effects[cell]) > float(b_effects[cell]) for cell in c_effects):
            return False
    return float(candidate["point_max_standardized_effect"]) < float(
        baseline["point_max_standardized_effect"]
    )


def _validated_responses(
    responses: Mapping[str, np.ndarray],
    scales: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], int, int]:
    if not responses:
        raise ValueError("at least one consumer response is required")
    if set(responses) != set(scales):
        raise ValueError("response consumers and scale consumers must match exactly")

    checked: dict[str, np.ndarray] = {}
    n_states = n_backgrounds = None
    for name, raw in responses.items():
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(
                f"consumer {name!r} must have shape [state, background, feature]"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"consumer {name!r} contains non-finite responses")
        scale = float(scales[name])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"consumer {name!r} scale must be finite and positive")
        if n_states is None:
            n_states, n_backgrounds = array.shape[:2]
        elif array.shape[:2] != (n_states, n_backgrounds):
            raise ValueError("all consumers must share state and background axes")
        checked[name] = array / scale

    assert n_states is not None and n_backgrounds is not None
    if n_states < 2:
        raise ValueError("at least two states are required")
    if n_backgrounds < 1:
        raise ValueError("at least one background is required")
    return checked, n_states, n_backgrounds


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q, method="higher"))


def score_partition(
    responses: Mapping[str, np.ndarray],
    labels: Sequence[object],
    *,
    scales: Mapping[str, float],
    gates: Mapping[str, ConsumerGate],
    declared_price_bits: float,
    minimum_non_singleton_coverage: float = 0.90,
) -> dict[str, object]:
    """Score a proposed discrete quotient without averaging away failures.

    Each response tensor has shape ``[state, background, feature]``.  Distances are
    Euclidean over the feature axis.  For a pair of states, the authoritative
    distance is the maximum across backgrounds.  The report also includes the mean
    distance only as a diagnostic.
    """

    checked, n_states, n_backgrounds = _validated_responses(responses, scales)
    if set(checked) != set(gates):
        raise ValueError("every and only registered consumers must have a gate")
    if not np.isfinite(declared_price_bits) or declared_price_bits < 0:
        raise ValueError("declared_price_bits must be finite and nonnegative")
    if not 0 <= minimum_non_singleton_coverage <= 1:
        raise ValueError("minimum_non_singleton_coverage must lie in [0, 1]")

    lab = np.asarray(list(labels), dtype=object)
    if lab.shape != (n_states,):
        raise ValueError("labels must contain exactly one entry per state")
    if any(value is None for value in lab):
        raise ValueError("labels cannot contain None")

    unique, inverse, counts = np.unique(lab.astype(str), return_inverse=True,
                                        return_counts=True)
    non_singleton = counts[inverse] >= 2
    coverage = float(non_singleton.mean())

    within_pairs: list[tuple[int, int]] = []
    between_pairs: list[tuple[int, int]] = []
    for left in range(n_states):
        for right in range(left + 1, n_states):
            target = within_pairs if inverse[left] == inverse[right] else between_pairs
            target.append((left, right))

    consumer_reports: dict[str, dict[str, object]] = {}
    all_consumers_pass = True
    for name, array in checked.items():
        gate = gates[name]

        def distances(pairs: list[tuple[int, int]]) -> np.ndarray:
            if not pairs:
                return np.empty((0, n_backgrounds), dtype=np.float64)
            left = np.fromiter((pair[0] for pair in pairs), dtype=np.int64)
            right = np.fromiter((pair[1] for pair in pairs), dtype=np.int64)
            delta = array[left] - array[right]
            return np.sqrt(np.mean(delta * delta, axis=-1))

        within = distances(within_pairs)
        between = distances(between_pairs)
        within_pair_worst = within.max(axis=1) if within.size else np.empty(0)
        between_pair_worst = between.max(axis=1) if between.size else np.empty(0)

        within_q95 = _quantile(within_pair_worst, 0.95)
        within_max = float(within_pair_worst.max()) if within_pair_worst.size else float("nan")
        within_mean_diagnostic = float(within.mean()) if within.size else float("nan")
        between_median = float(np.median(between_pair_worst)) if between_pair_worst.size else float("nan")
        separation_ratio = (
            between_median / max(within_q95, np.finfo(np.float64).tiny)
            if np.isfinite(between_median) and np.isfinite(within_q95)
            else float("nan")
        )
        consumer_pass = bool(
            within_pair_worst.size
            and between_pair_worst.size
            and within_q95 <= gate.epsilon_q95
            and within_max <= gate.epsilon_max
            and separation_ratio >= gate.minimum_separation_ratio
        )
        all_consumers_pass = all_consumers_pass and consumer_pass
        consumer_reports[name] = {
            "n_within_pairs": len(within_pairs),
            "n_between_pairs": len(between_pairs),
            "within_mean_diagnostic_only": within_mean_diagnostic,
            "within_pair_worst_q95": within_q95,
            "within_worst": within_max,
            "between_pair_worst_median": between_median,
            "separation_ratio": separation_ratio,
            "gate": asdict(gate),
            "passes": consumer_pass,
        }

    coverage_pass = coverage >= minimum_non_singleton_coverage
    return {
        "schema_version": 1,
        "n_states": n_states,
        "n_backgrounds": n_backgrounds,
        "n_cells": int(unique.size),
        "n_singleton_cells": int((counts == 1).sum()),
        "non_singleton_state_coverage": coverage,
        "minimum_non_singleton_coverage": minimum_non_singleton_coverage,
        "coverage_passes": coverage_pass,
        "declared_price_bits": float(declared_price_bits),
        "consumers": consumer_reports,
        "passes_all_consumers": bool(all_consumers_pass),
        "causal_quotient_passes": bool(all_consumers_pass and coverage_pass),
        "acceptance_uses_mean_response": False,
    }


def rank_passing_candidates(reports: Mapping[str, Mapping[str, object]]) -> list[str]:
    """Return passing candidates ordered by externally declared bit price.

    This is deliberately not a scalar quality ranking: distortion gates are hard
    constraints, and price only orders candidates that passed every constraint.
    """

    eligible: list[tuple[float, str]] = []
    for name, report in reports.items():
        if bool(report.get("causal_quotient_passes", False)):
            price = float(report["declared_price_bits"])
            if not np.isfinite(price) or price < 0:
                raise ValueError(f"candidate {name!r} has invalid declared price")
            eligible.append((price, name))
    return [name for _, name in sorted(eligible, key=lambda item: (item[0], item[1]))]
