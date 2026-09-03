#!/usr/bin/env python3
"""Pure-CPU scalar-gauge algebra for rung 531's factor-sharing screen."""

from __future__ import annotations

import math
from typing import Mapping

import torch


ASSIGNMENTS = ("direct", "swapped")


def _flat64(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float64, device="cpu").reshape(-1)
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("factor arrays must be nonempty and finite")
    return value


def fit_scalar(source: torch.Tensor, target: torch.Tensor) -> float:
    """Least-squares scalar taking ``source`` to ``target``."""
    source64, target64 = _flat64(source), _flat64(target)
    if source64.shape != target64.shape:
        raise ValueError("source and target shapes differ")
    denominator = float(torch.dot(source64, source64))
    if denominator <= 0:
        raise ValueError("cannot fit a scalar from an all-zero source")
    return float(torch.dot(source64, target64)) / denominator


def prediction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction64, target64 = _flat64(prediction), _flat64(target)
    if prediction64.shape != target64.shape:
        raise ValueError("prediction and target shapes differ")
    target2 = float(torch.dot(target64, target64))
    prediction2 = float(torch.dot(prediction64, prediction64))
    if target2 <= 0 or prediction2 <= 0:
        raise ValueError("cosine and relative error require nonzero arrays")
    cross = float(torch.dot(prediction64, target64))
    error2 = float(torch.dot(prediction64 - target64, prediction64 - target64))
    return {
        "cosine": cross / math.sqrt(prediction2 * target2),
        "relative_rmse": math.sqrt(max(error2, 0.0) / target2),
    }


def _ordered_sources(
    source_first: torch.Tensor,
    source_second: torch.Tensor,
    assignment: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if assignment == "direct":
        return source_first, source_second
    if assignment == "swapped":
        return source_second, source_first
    raise ValueError(f"unknown assignment: {assignment}")


def fit_assignment(
    source_first: torch.Tensor,
    source_second: torch.Tensor,
    target_first: torch.Tensor,
    target_second: torch.Tensor,
    assignment: str,
) -> dict[str, object]:
    """Fit one scalar for each target branch under a direct or swapped assignment."""
    for value in (source_first, source_second, target_first, target_second):
        _flat64(value)
    first_source, second_source = _ordered_sources(source_first, source_second, assignment)
    alpha = fit_scalar(first_source, target_first)
    beta = fit_scalar(second_source, target_second)
    predicted_first = _flat64(first_source) * alpha
    predicted_second = _flat64(second_source) * beta
    target_first64, target_second64 = _flat64(target_first), _flat64(target_second)
    first = prediction_metrics(predicted_first, target_first64)
    second = prediction_metrics(predicted_second, target_second64)
    branch_objective = first["relative_rmse"] ** 2 + second["relative_rmse"] ** 2
    predicted_product = predicted_first * predicted_second
    target_product = target_first64 * target_second64
    product = prediction_metrics(predicted_product, target_product)
    source_product = _flat64(source_first) * _flat64(source_second)
    product_scale = fit_scalar(source_product, target_product)
    scalar_product_prediction = source_product * product_scale
    return {
        "assignment": assignment,
        "target_first_scale": alpha,
        "target_second_scale": beta,
        "branch_scale_product": alpha * beta,
        "independent_product_scale": product_scale,
        "scale_product_relative_difference": abs(alpha * beta - product_scale)
        / max(abs(product_scale), 1e-30),
        "first": first,
        "second": second,
        "product": product,
        "scalar_product_baseline": prediction_metrics(scalar_product_prediction, target_product),
        "branch_objective": branch_objective,
    }


def choose_assignment(
    source_first: torch.Tensor,
    source_second: torch.Tensor,
    target_first: torch.Tensor,
    target_second: torch.Tensor,
) -> dict[str, object]:
    """Discovery-only choice; exact ties resolve to direct as preregistered."""
    reports: Mapping[str, dict[str, object]] = {
        assignment: fit_assignment(
            source_first, source_second, target_first, target_second, assignment
        )
        for assignment in ASSIGNMENTS
    }
    selected = min(
        ASSIGNMENTS,
        key=lambda assignment: (float(reports[assignment]["branch_objective"]),
                                ASSIGNMENTS.index(assignment)),
    )
    return {"selected": selected, "reports": dict(reports)}


def evaluate_frozen_assignment(
    source_first: torch.Tensor,
    source_second: torch.Tensor,
    target_first: torch.Tensor,
    target_second: torch.Tensor,
    *,
    assignment: str,
    target_first_scale: float,
    target_second_scale: float,
) -> dict[str, object]:
    """Apply discovery-fitted assignment and scalars without held-out refitting."""
    first_source, second_source = _ordered_sources(source_first, source_second, assignment)
    predicted_first = _flat64(first_source) * float(target_first_scale)
    predicted_second = _flat64(second_source) * float(target_second_scale)
    target_first64, target_second64 = _flat64(target_first), _flat64(target_second)
    source_product = _flat64(source_first) * _flat64(source_second)
    target_product = target_first64 * target_second64
    return {
        "assignment": assignment,
        "first": prediction_metrics(predicted_first, target_first64),
        "second": prediction_metrics(predicted_second, target_second64),
        "product": prediction_metrics(predicted_first * predicted_second, target_product),
        "branch_scale_product": float(target_first_scale) * float(target_second_scale),
        "source_product": source_product,
        "target_product": target_product,
    }
