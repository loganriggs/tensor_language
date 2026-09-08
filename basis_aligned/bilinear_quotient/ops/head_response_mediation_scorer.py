"""Pure scoring for head/module response-mediation atlases."""

from __future__ import annotations

import math

import head_response_mediation_contract as mediation
from two_by_two_dependency_contract import vector_metrics


class MediationScoreError(ValueError):
    pass


def score_cells(cells, full_target):
    parts = mediation.decompose_margin_cells(cells)
    target = tuple(float(value) for value in full_target)
    if len(target) != len(parts["full_upstream_effect"]):
        raise MediationScoreError("target and cell vectors must have equal length")
    parts["metrics"] = {
        name: vector_metrics(values, target)
        for name, values in parts.items()
        if isinstance(values, list)
    }
    return parts


def deterministic_head_ranking(reports, component):
    ranked = []
    for head, report in reports.items():
        try:
            score = float(report["metrics"][component]["signed_projection"])
        except (KeyError, TypeError, ValueError) as error:
            raise MediationScoreError(f"missing score for {head}:{component}") from error
        if not math.isfinite(score):
            raise MediationScoreError("head score is nonfinite")
        ranked.append((str(head), score))
    return [head for head, _score in sorted(ranked, key=lambda item: (-item[1], item[0]))]


def singleton_module_composition(singletons, module):
    if not singletons:
        raise MediationScoreError("at least one singleton is required")
    try:
        vectors = [tuple(float(value) for value in report["head_reset_loss"])
                   for report in singletons.values()]
        target = tuple(float(value) for value in module["head_reset_loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise MediationScoreError("missing reset-loss vector") from error
    if not target or any(len(vector) != len(target) for vector in vectors):
        raise MediationScoreError("reset-loss vectors must be nonempty and equal length")
    summed = [sum(vector[row] for vector in vectors) for row in range(len(target))]
    metrics = vector_metrics(summed, target)
    metrics["relative_l2_error"] = math.sqrt(metrics["relative_squared_error"])
    return {"summed_singleton_reset_loss": summed,
            "module_reset_loss": list(target), "metrics": metrics}
