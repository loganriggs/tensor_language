"""CPU mathematics for trajectory-complete bilinear-gate response selection.

The production collector is deliberately absent.  This module defines and tests the
only response object it may publish, together with deterministic paired selectors.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch


def trajectory_complete_response(
    products: torch.Tensor, write_gradients: torch.Tensor, down: torch.Tensor,
) -> torch.Tensor:
    """Return d(score)/d(global gate scale) with the token-position sum intact.

    ``products`` is [context, position, gate], ``write_gradients`` is
    [probe, context, position, output], and ``down`` is [output, gate].
    """
    if (
        not torch.is_tensor(products) or not torch.is_tensor(write_gradients)
        or not torch.is_tensor(down) or products.ndim != 3
        or write_gradients.ndim != 4 or down.ndim != 2
        or products.shape[:2] != write_gradients.shape[1:3]
        or products.shape[2] != down.shape[1]
        or write_gradients.shape[3] != down.shape[0]
        or not products.is_floating_point() or not write_gradients.is_floating_point()
        or not down.is_floating_point()
        or not bool(torch.isfinite(products).all())
        or not bool(torch.isfinite(write_gradients).all())
        or not bool(torch.isfinite(down).all())
    ):
        raise ValueError("trajectory-complete gate-response inputs are malformed")
    return torch.einsum(
        "ctn,pcto,on->pcn", products.double(), write_gradients.double(), down.double(),
    ).contiguous()


def context_balance(response: torch.Tensor) -> torch.Tensor:
    """Give each context equal Frobenius weight without mixing contexts or probes."""
    _validate_response(response)
    norms = response.double().square().sum(dim=(1, 2), keepdim=True).sqrt()
    if bool((norms <= 0).any()):
        raise ValueError("every context must have positive response energy")
    return (response.double() / norms).contiguous()


def _validate_response(response: torch.Tensor) -> None:
    if (
        not torch.is_tensor(response) or response.ndim != 3
        or min(response.shape) <= 0 or not response.is_floating_point()
        or not bool(torch.isfinite(response).all())
    ):
        raise ValueError("response must be finite [context, probe, gate]")


def _matrix(response: torch.Tensor) -> torch.Tensor:
    _validate_response(response)
    return response.double().reshape(-1, response.shape[-1])


def ridge_leverage_scores(response: torch.Tensor, target_rank: int) -> torch.Tensor:
    """Column ridge leverage with lambda = rank-k tail energy / k.

    This is a response-span importance score, not a finite-pruning certificate.
    """
    matrix = _matrix(response)
    limit = min(matrix.shape)
    if type(target_rank) is not int or not 0 < target_rank < limit:
        raise ValueError("target_rank must lie strictly inside the matrix dimensions")
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    tail = singular[target_rank:].square().sum()
    ridge = tail / target_rank
    tolerance = torch.finfo(singular.dtype).eps * max(matrix.shape) * singular[0]
    if singular[target_rank] <= tolerance:
        weights = (singular > tolerance).to(singular.dtype)
    else:
        weights = singular.square() / (singular.square() + ridge)
    scores = (weights[:, None] * vh.square()).sum(dim=0)
    return scores.contiguous()


def column_energy_scores(response: torch.Tensor) -> torch.Tensor:
    return _matrix(response).square().sum(dim=0).contiguous()


def select_top(scores: torch.Tensor, count: int) -> tuple[int, ...]:
    if (
        not torch.is_tensor(scores) or scores.ndim != 1 or not scores.is_floating_point()
        or not bool(torch.isfinite(scores).all()) or type(count) is not int
        or not 0 < count <= scores.numel()
    ):
        raise ValueError("selection scores or count are malformed")
    values = scores.detach().cpu().double().tolist()
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index))[:count])


def hash_random_selection(gates: int, count: int, seed: int) -> tuple[int, ...]:
    if min(gates, count) <= 0 or count > gates or min(seed, gates) < 0:
        raise ValueError("random-control constants are malformed")
    return tuple(sorted(
        range(gates), key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest(),
    )[:count])


def projection_capture(response: torch.Tensor, selected: Sequence[int]) -> float:
    """Fraction of response-matrix Frobenius energy in selected columns' span."""
    matrix = _matrix(response)
    indices = tuple(selected)
    if (
        not indices or len(indices) != len(set(indices))
        or any(type(index) is not int or not 0 <= index < matrix.shape[1] for index in indices)
    ):
        raise ValueError("selected gate indices are malformed")
    columns = matrix[:, list(indices)]
    u, singular, _ = torch.linalg.svd(columns, full_matrices=False)
    tolerance = torch.finfo(singular.dtype).eps * max(columns.shape) * singular[0]
    support = int((singular > tolerance).sum())
    if support == 0:
        return 0.0
    basis = u[:, :support]
    captured = (basis.T @ matrix).square().sum()
    total = matrix.square().sum()
    return float((captured / total).clamp(0, 1))


def all_on_transfer_relative_error(
    fit_response: torch.Tensor, eval_response: torch.Tensor, selected: Sequence[int],
) -> float:
    """Fit selected gates to the all-on tangent target, then transfer coefficients.

    This evaluates the sparse all-on question; it is deliberately distinct from
    column-span capture.  The same coefficients are used on the evaluation half.
    """
    fit, evaluate = _matrix(fit_response), _matrix(eval_response)
    if fit.shape != evaluate.shape:
        raise ValueError("all-on response halves must have identical shape")
    indices = tuple(selected)
    if (
        not indices or len(indices) != len(set(indices))
        or any(type(index) is not int or not 0 <= index < fit.shape[1] for index in indices)
    ):
        raise ValueError("selected gate indices are malformed")
    fit_target, eval_target = fit.sum(dim=1), evaluate.sum(dim=1)
    denominator = torch.linalg.vector_norm(eval_target)
    if float(denominator) <= 0:
        raise ValueError("evaluation all-on target must have positive energy")
    coefficients = torch.linalg.lstsq(fit[:, list(indices)], fit_target).solution
    residual = evaluate[:, list(indices)] @ coefficients - eval_target
    return float(torch.linalg.vector_norm(residual) / denominator)


def paired_selector_report(
    first: torch.Tensor, second: torch.Tensor, *, budgets: Sequence[int],
    target_rank: int, random_seed: int,
) -> dict[str, Any]:
    """Outcome-blind deterministic comparison on two independent response halves."""
    _validate_response(first)
    _validate_response(second)
    if first.shape != second.shape:
        raise ValueError("paired response halves must have identical shape")
    counts = tuple(budgets)
    if (
        not counts or len(counts) != len(set(counts))
        or any(type(count) is not int or not 0 < count <= first.shape[-1] for count in counts)
    ):
        raise ValueError("gate budgets are malformed")
    first_balanced, second_balanced = context_balance(first), context_balance(second)
    score_banks = {
        "ridge": (
            ridge_leverage_scores(first_balanced, target_rank),
            ridge_leverage_scores(second_balanced, target_rank),
        ),
        "energy": (
            column_energy_scores(first_balanced), column_energy_scores(second_balanced),
        ),
    }
    rows: dict[str, Any] = {}
    for count in counts:
        controls = {
            name: (select_top(a, count), select_top(b, count))
            for name, (a, b) in score_banks.items()
        }
        random = hash_random_selection(first.shape[-1], count, random_seed)
        controls["hash_random"] = (random, random)
        row = {}
        for name, (selected_a, selected_b) in controls.items():
            intersection = len(set(selected_a) & set(selected_b))
            union = len(set(selected_a) | set(selected_b))
            row[name] = {
                "first_selected": list(selected_a),
                "second_selected": list(selected_b),
                "jaccard": intersection / union,
                "first_to_second_capture": projection_capture(second_balanced, selected_a),
                "second_to_first_capture": projection_capture(first_balanced, selected_b),
                "first_to_second_all_on_relative_error": all_on_transfer_relative_error(
                    first_balanced, second_balanced, selected_a,
                ),
                "second_to_first_all_on_relative_error": all_on_transfer_relative_error(
                    second_balanced, first_balanced, selected_b,
                ),
            }
        rows[str(count)] = row
    return {
        "status": "paired_gate_response_selector_complete",
        "shape": list(first.shape),
        "target_rank": target_rank,
        "budgets": list(counts),
        "context_balanced": True,
        "rows": rows,
        "claim_boundary": (
            "response-span selection only; no native hard-retention, refitted-Down, "
            "finite-removal, CE, causal-equivalence, or arithmetic-rank claim"
        ),
    }
