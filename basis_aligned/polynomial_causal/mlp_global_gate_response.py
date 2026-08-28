"""CPU mathematics for trajectory-complete bilinear-gate response selection.

The production collector is deliberately absent.  This module defines and tests the
only response object it may publish, together with deterministic paired selectors.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch


SVD_RELATIVE_CUTOFF = 1e-10
TIKHONOV_RELATIVE_RIDGE = 1e-6
MAXIMUM_RETAINED_CONDITION = 1e6
MAXIMUM_NORMALIZED_COEFFICIENT_NORM = 10.0


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
        "ctn,pcto,on->cpn", products.double(), write_gradients.double(), down.double(),
    ).contiguous()


def context_balance(response: torch.Tensor) -> torch.Tensor:
    """Give each context equal Frobenius weight without mixing contexts or probes."""
    _validate_response(response)
    norms = response.double().square().sum(dim=(1, 2), keepdim=True).sqrt()
    if bool((norms <= 0).any()):
        raise ValueError("every context must have positive response energy")
    return (response.double() / norms).contiguous()


def canonicalize_factor_product_gates(
    products: torch.Tensor, down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fix the scalar/sign gauge of native product and Down factors.

    For each gate, its product trace is normalized to unit RMS on the fit rows and the
    inverse scale is absorbed into its Down column.  The joint sign is oriented so the
    first maximum-magnitude Down entry is positive.  Consequently the returned pair is
    unchanged under ``h_n -> a_n h_n, d_n -> d_n/a_n`` for every nonzero ``a_n``.
    """
    if (
        not torch.is_tensor(products) or not torch.is_tensor(down)
        or products.ndim != 3 or down.ndim != 2
        or products.shape[2] != down.shape[1] or min(products.shape) <= 0
        or min(down.shape) <= 0 or not products.is_floating_point()
        or not down.is_floating_point() or not bool(torch.isfinite(products).all())
        or not bool(torch.isfinite(down).all())
    ):
        raise ValueError("factor-product canonicalization inputs are malformed")
    product64, down64 = products.double(), down.double()
    rms = product64.square().mean(dim=(0, 1)).sqrt()
    if bool((rms <= 0).any()):
        raise ValueError("every factor-product gate must have positive fit-row RMS")
    scaled_down = down64 * rms[None, :]
    pivots = scaled_down.abs().argmax(dim=0)
    pivot_values = scaled_down.gather(0, pivots[None, :]).squeeze(0)
    if bool((pivot_values == 0).any()):
        raise ValueError("every factor-product Down column must be nonzero")
    orientation = torch.where(
        pivot_values < 0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values),
    )
    canonical_products = product64 / rms[None, None, :] * orientation[None, None, :]
    canonical_down = scaled_down * orientation[None, :]
    return canonical_products.contiguous(), canonical_down.contiguous(), {
        "scale_rule": "fit-row product RMS absorbed into Down",
        "sign_rule": "first maximum-absolute Down coordinate is positive",
        "pivot_indices": pivots.detach().cpu().tolist(),
    }


def canonical_factor_product_derangement(
    products: torch.Tensor, down: torch.Tensor, seed: int,
) -> tuple[int, ...]:
    """Return the frozen +1 content-order derangement of canonical physical gates."""
    if type(seed) is not int or seed < 0:
        raise ValueError("factor-product derangement seed is malformed")
    canonical_h, canonical_d, _ = canonicalize_factor_product_gates(products, down)
    gates = canonical_h.shape[2]
    if gates <= 1:
        raise ValueError("factor-product derangement requires at least two gates")
    records: list[tuple[bytes, bytes, bytes, int]] = []
    prefix = f"{seed}:".encode()
    for index in range(gates):
        h_bytes = canonical_h[:, :, index].contiguous().numpy().tobytes(order="C")
        d_bytes = canonical_d[:, index].contiguous().numpy().tobytes(order="C")
        digest = hashlib.sha256(prefix + h_bytes + d_bytes).digest()
        records.append((digest, h_bytes, d_bytes, index))
    order = tuple(record[3] for record in sorted(records))
    permutation = [-1] * gates
    for location, source in enumerate(order):
        permutation[source] = order[(location + 1) % gates]
    if sorted(permutation) != list(range(gates)) or any(
        source == target for source, target in enumerate(permutation)
    ):
        raise RuntimeError("canonical factor-product pairing is not a derangement")
    return tuple(permutation)


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
    """In-half support-span diagnostic; not a cross-half promotion metric."""
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


def cross_fit_css_relative_error(
    fit_response: torch.Tensor, eval_response: torch.Tensor, selected: Sequence[int],
) -> float:
    """Fit a column interpolant on one half and transfer it unchanged to another."""
    fit, evaluate = _matrix(fit_response), _matrix(eval_response)
    if fit.shape != evaluate.shape:
        raise ValueError("cross-fit CSS response halves must have identical shape")
    indices = tuple(selected)
    if (
        not indices or len(indices) != len(set(indices))
        or any(type(index) is not int or not 0 <= index < fit.shape[1] for index in indices)
    ):
        raise ValueError("selected gate indices are malformed")
    denominator = torch.linalg.matrix_norm(evaluate)
    if float(denominator) <= 0:
        raise ValueError("evaluation response must have positive energy")
    interpolant, _ = regularized_svd_solution(fit[:, list(indices)], fit)
    normalized = torch.linalg.matrix_norm(interpolant) / fit.shape[1] ** 0.5
    if float(normalized) > MAXIMUM_NORMALIZED_COEFFICIENT_NORM:
        raise ValueError("CSS interpolant coefficient norm exceeds the frozen gate")
    residual = evaluate[:, list(indices)] @ interpolant - evaluate
    return float(torch.linalg.matrix_norm(residual) / denominator)


def regularized_svd_solution(
    design: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Frozen float64 Tikhonov/SVD solver used by every candidate and control."""
    if (
        not torch.is_tensor(design) or not torch.is_tensor(target)
        or design.ndim != 2 or target.ndim not in (1, 2)
        or design.shape[0] != target.shape[0] or min(design.shape) <= 0
        or not design.is_floating_point() or not target.is_floating_point()
        or not bool(torch.isfinite(design).all()) or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("regularized solver inputs are malformed")
    matrix, outcome = design.double(), target.double()
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 0:
        raise ValueError("regularized solver design has zero support")
    cutoff = SVD_RELATIVE_CUTOFF * singular[0]
    retained = singular > cutoff
    support = int(retained.sum())
    if support == 0:
        raise ValueError("regularized solver retained no singular directions")
    condition = float(singular[0] / singular[retained][-1])
    if condition > MAXIMUM_RETAINED_CONDITION:
        raise ValueError("regularized solver condition number exceeds the frozen gate")
    ridge = TIKHONOV_RELATIVE_RIDGE * singular[0].square()
    weights = torch.where(
        retained, singular / (singular.square() + ridge), torch.zeros_like(singular),
    )
    solution = (vh.T * weights) @ (u.T @ outcome)
    if not bool(torch.isfinite(solution).all()):
        raise ValueError("regularized solver produced nonfinite coefficients")
    return solution.contiguous(), {
        "numerical_rank": support,
        "retained_condition_number": condition,
        "largest_singular_value": float(singular[0]),
        "relative_singular_cutoff": SVD_RELATIVE_CUTOFF,
        "relative_tikhonov_ridge": TIKHONOV_RELATIVE_RIDGE,
    }


def fit_all_on_coefficients(
    response: torch.Tensor, selected: Sequence[int],
) -> torch.Tensor:
    """Fit the selected gates to ``E @ 1`` and return the frozen coefficients."""
    matrix = _matrix(response)
    indices = tuple(selected)
    if (
        not indices or len(indices) != len(set(indices))
        or any(type(index) is not int or not 0 <= index < matrix.shape[1] for index in indices)
    ):
        raise ValueError("selected gate indices are malformed")
    solution, _ = regularized_svd_solution(
        matrix[:, list(indices)], matrix.sum(dim=1),
    )
    normalized = torch.linalg.vector_norm(solution) / len(indices) ** 0.5
    if float(normalized) > MAXIMUM_NORMALIZED_COEFFICIENT_NORM:
        raise ValueError("all-on coefficient norm exceeds the frozen gate")
    return solution.contiguous()


def candidate_path_scale(
    gates: int, selected: Sequence[int], coefficients: torch.Tensor, epsilon: float,
) -> torch.Tensor:
    """Scale on the path from the native all-on MLP to a sparse candidate.

    At epsilon zero every gate is on.  At epsilon one, omitted gates are zero and
    selected gates have their fitted candidate coefficients.
    """
    indices = tuple(selected)
    if (
        type(gates) is not int or gates <= 0 or not indices
        or len(indices) != len(set(indices))
        or any(type(index) is not int or not 0 <= index < gates for index in indices)
        or not torch.is_tensor(coefficients) or coefficients.ndim != 1
        or coefficients.numel() != len(indices) or not coefficients.is_floating_point()
        or not bool(torch.isfinite(coefficients).all())
        or not isinstance(epsilon, float) or not 0.0 <= epsilon <= 1.0
    ):
        raise ValueError("sparse-candidate path inputs are malformed")
    candidate = torch.zeros(gates, dtype=coefficients.dtype, device=coefficients.device)
    candidate[list(indices)] = coefficients
    return (torch.ones_like(candidate) + epsilon * (candidate - 1.0)).contiguous()


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
    coefficients = fit_all_on_coefficients(fit_response, indices)
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
                "first_to_second_cross_fit_css_relative_error": cross_fit_css_relative_error(
                    first_balanced, second_balanced, selected_a,
                ),
                "second_to_first_cross_fit_css_relative_error": cross_fit_css_relative_error(
                    second_balanced, first_balanced, selected_b,
                ),
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
            "tangent response selection only; in-half span capture is diagnostic and "
            "cross-fit CSS/all-on errors freeze fit-half coefficients; no native "
            "hard-retention, refitted-Down, "
            "finite-removal, CE, causal-equivalence, or arithmetic-rank claim"
        ),
    }
