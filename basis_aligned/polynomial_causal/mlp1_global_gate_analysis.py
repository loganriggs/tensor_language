"""Deterministic fit-only analysis for the frozen MLP1 physical-gate assay.

Raw response tensors enter this module in memory and never enter its JSON summary.  A
single fit/first support and coefficient bundle is constructed for every registered
selector and budget, then transferred unchanged to fit/second and both validation
cells.  The returned tensor bundle is intended for create-only ``torch.save`` by the
production collector.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch

import mlp_global_gate_response as gate


CELL_NAMES = (
    "fit_first", "fit_second", "validation_first", "validation_second",
)
VALIDATION_CELLS = ("validation_first", "validation_second")
ARM_NAMES = (
    "primary", "response_energy", "activation_down",
    "factor_product_derangement", "hash_random",
)
CONTROL_NAMES = ARM_NAMES[1:]
METRIC_NAMES = ("css", "all_on")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def raw_tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "raw_sha256": raw_tensor_sha256(tensor),
    }


def _validate_cells(cells: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
    if set(cells) != set(CELL_NAMES):
        raise ValueError("global-gate response cell ledger is incomplete")
    shapes = {tuple(value.shape) for value in cells.values() if torch.is_tensor(value)}
    if len(shapes) != 1:
        raise ValueError("global-gate response cells must have one common shape")
    shape = next(iter(shapes), ())
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("global-gate response cells must be [document,probe,gate]")
    for value in cells.values():
        if (
            not value.is_floating_point() or value.dtype != torch.float64
            or value.device.type != "cpu" or value.requires_grad
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("global-gate response cells must be finite CPU float64")
    return shape


def _ridge_scores_by_rank(
    response: torch.Tensor, target_ranks: Sequence[int],
) -> dict[int, torch.Tensor]:
    matrix = response.reshape(-1, response.shape[-1]).double()
    limit = min(matrix.shape)
    ranks = tuple(target_ranks)
    if (
        not ranks or len(set(ranks)) != len(ranks)
        or any(type(rank) is not int or not 0 < rank < limit for rank in ranks)
    ):
        raise ValueError("ridge target ranks are malformed")
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    result = {}
    for rank in ranks:
        tail = singular[rank:].square().sum()
        ridge = tail / rank
        if bool(ridge > 0):
            weights = singular.square() / (singular.square() + ridge)
        else:
            tolerance = torch.finfo(singular.dtype).eps * max(matrix.shape) * singular[0]
            weights = (singular > tolerance).to(singular.dtype)
        result[rank] = (weights[:, None] * vh.square()).sum(dim=0).contiguous()
    return result


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or not values.is_floating_point() or not bool(
        torch.isfinite(values).all()
    ):
        raise ValueError("rank-correlation scores are malformed")
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < values.numel():
        stop = start + 1
        while stop < values.numel() and bool(sorted_values[stop] == sorted_values[start]):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_rank_correlation(first: torch.Tensor, second: torch.Tensor) -> float | None:
    if first.shape != second.shape:
        raise ValueError("Spearman score vectors must have identical shape")
    left, right = _average_ranks(first), _average_ranks(second)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0:
        return None
    return float((torch.dot(left, right) / denominator).clamp(-1, 1))


def support_jaccard(first: Sequence[int], second: Sequence[int]) -> float:
    left, right = set(first), set(second)
    if not left or not right:
        raise ValueError("support Jaccard requires nonempty supports")
    return len(left & right) / len(left | right)


def _hash_rank_scores(gates: int, seed: int) -> torch.Tensor:
    order = sorted(
        range(gates), key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest(),
    )
    scores = torch.empty(gates, dtype=torch.float64)
    for rank, index in enumerate(order):
        scores[index] = gates - rank
    return scores


def _fit_bundle(
    fit_response: torch.Tensor, support: Sequence[int], scores: torch.Tensor,
    *, selector_currency: str, target_rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = fit_response.reshape(-1, fit_response.shape[-1]).double()
    indices = tuple(support)
    design = matrix[:, list(indices)]
    css, css_solver = gate.regularized_svd_solution(design, matrix)
    css_norm = float(torch.linalg.matrix_norm(css) / matrix.shape[1] ** 0.5)
    beta, beta_solver = gate.regularized_svd_solution(design, matrix.sum(dim=1))
    beta_norm = float(torch.linalg.vector_norm(beta) / len(indices) ** 0.5)
    if css_norm > gate.MAXIMUM_NORMALIZED_COEFFICIENT_NORM or beta_norm > (
        gate.MAXIMUM_NORMALIZED_COEFFICIENT_NORM
    ):
        raise ValueError("support coefficients exceed the frozen norm gate")
    support_tensor = torch.tensor(indices, dtype=torch.int64)
    tensor_bundle = {
        "support": support_tensor,
        "selection_scores": scores.detach().cpu().double().contiguous(),
        "css_coefficients": css.detach().cpu().double().contiguous(),
        "all_on_coefficients": beta.detach().cpu().double().contiguous(),
    }
    summary = {
        "status": "fit_complete",
        "support": list(indices),
        "selector_currency": selector_currency,
        "target_rank": target_rank,
        "tensors": {name: tensor_descriptor(value) for name, value in tensor_bundle.items()},
        "css_solver": {**css_solver, "normalized_coefficient_norm": css_norm},
        "all_on_solver": {**beta_solver, "normalized_coefficient_norm": beta_norm},
    }
    return tensor_bundle, summary


def _per_document_losses(
    response: torch.Tensor, support: Sequence[int], css: torch.Tensor,
    beta: torch.Tensor,
) -> dict[str, dict[str, list[float]] | dict[str, float]]:
    selected = response[:, :, list(support)]
    css_residual = torch.einsum("dpk,kg->dpg", selected, css) - response
    css_numerator = css_residual.square().sum(dim=(1, 2))
    css_denominator = response.square().sum(dim=(1, 2))
    target = response.sum(dim=2)
    all_on_residual = torch.einsum("dpk,k->dp", selected, beta) - target
    all_on_numerator = all_on_residual.square().sum(dim=1)
    all_on_denominator = target.square().sum(dim=1)
    if bool((css_denominator <= 0).any()) or bool((all_on_denominator <= 0).any()):
        raise ValueError("every document must have positive CSS and all-on energy")

    def record(numerator: torch.Tensor, denominator: torch.Tensor) -> dict[str, Any]:
        ratios = numerator / denominator.clamp_min(1e-30)
        return {
            "numerator": numerator.tolist(),
            "denominator": denominator.tolist(),
            "ratio": ratios.tolist(),
            "pooled_loss": float(numerator.sum() / denominator.sum().clamp_min(1e-30)),
        }

    return {
        "css": record(css_numerator, css_denominator),
        "all_on": record(all_on_numerator, all_on_denominator),
    }


def _comparison_records(
    losses: Mapping[str, Any], budgets: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records = []
    public = {}
    for budget in budgets:
        key = str(budget)
        if any(losses[key][arm].get("status") != "fit_complete" for arm in ARM_NAMES):
            continue
        for control in CONTROL_NAMES:
            for metric in METRIC_NAMES:
                for cell in VALIDATION_CELLS:
                    primary = losses[key]["primary"]["cells"][cell][metric]
                    baseline = losses[key][control]["cells"][cell][metric]
                    primary_loss = float(primary["pooled_loss"])
                    control_loss = float(baseline["pooled_loss"])
                    improvement = (
                        (control_loss - primary_loss) / max(control_loss, 1e-30)
                    )
                    primary_ratio = torch.tensor(primary["ratio"], dtype=torch.float64)
                    control_ratio = torch.tensor(baseline["ratio"], dtype=torch.float64)
                    comparison_id = f"K{budget}:{control}:{metric}:{cell}"
                    worst_harm = float((primary_ratio - control_ratio).max())
                    row = {
                        "id": comparison_id,
                        "budget": budget,
                        "control": control,
                        "metric": metric,
                        "cell": cell,
                        "observed_improvement": improvement,
                        "worst_per_document_primary_minus_control_loss": worst_harm,
                        "primary_numerator": torch.tensor(
                            primary["numerator"], dtype=torch.float64,
                        ),
                        "primary_denominator": torch.tensor(
                            primary["denominator"], dtype=torch.float64,
                        ),
                        "control_numerator": torch.tensor(
                            baseline["numerator"], dtype=torch.float64,
                        ),
                        "control_denominator": torch.tensor(
                            baseline["denominator"], dtype=torch.float64,
                        ),
                    }
                    records.append(row)
                    public[comparison_id] = {
                        key: value for key, value in row.items()
                        if not torch.is_tensor(value)
                    }
    return records, public


def _simultaneous_bootstrap(
    records: Sequence[Mapping[str, Any]], *, documents: int,
    repetitions: int, seed: int, confidence: float,
) -> dict[str, Any]:
    if (
        documents <= 1 or repetitions <= 0 or not 0 < confidence < 1
        or not records
    ):
        raise ValueError("simultaneous bootstrap inputs are malformed")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        documents, (repetitions, documents), generator=generator,
    )
    maximum_errors = torch.full((repetitions,), -torch.inf, dtype=torch.float64)
    for record in records:
        primary_numerator = record["primary_numerator"][indices].sum(dim=1)
        primary_denominator = record["primary_denominator"][indices].sum(dim=1)
        control_numerator = record["control_numerator"][indices].sum(dim=1)
        control_denominator = record["control_denominator"][indices].sum(dim=1)
        primary_loss = primary_numerator / primary_denominator.clamp_min(1e-30)
        control_loss = control_numerator / control_denominator.clamp_min(1e-30)
        bootstrap_improvement = (
            (control_loss - primary_loss) / control_loss.clamp_min(1e-30)
        )
        error = float(record["observed_improvement"]) - bootstrap_improvement
        maximum_errors = torch.maximum(maximum_errors, error)
    rank = math.ceil(confidence * repetitions)
    critical = float(torch.kthvalue(maximum_errors, rank).values)
    return {
        "status": "complete",
        "repetitions": repetitions,
        "seed": seed,
        "documents_per_draw": documents,
        "shared_document_indices_across_every_comparison": True,
        "comparisons_evaluated": len(records),
        "confidence": confidence,
        "critical_order_statistic_one_indexed": rank,
        "interpolation": "none",
        "critical_max_error": critical,
        "maximum_error_draws_raw_sha256": raw_tensor_sha256(maximum_errors),
        "simultaneous_lcb": {
            str(record["id"]): float(record["observed_improvement"]) - critical
            for record in records
        },
    }


def analyze_global_gate_responses(
    cells: Mapping[str, torch.Tensor], *, deranged_fit_first: torch.Tensor,
    activation_rms: torch.Tensor, down: torch.Tensor, plan: Mapping[str, Any],
    fit_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build fit/first bundles and return aggregate result plus tensor bundle."""
    documents, probes, gates = _validate_cells(cells)
    if (
        tuple(deranged_fit_first.shape) != (documents, probes, gates)
        or deranged_fit_first.dtype != torch.float64
        or deranged_fit_first.device.type != "cpu"
        or not bool(torch.isfinite(deranged_fit_first).all())
        or tuple(activation_rms.shape) != (gates,)
        or activation_rms.dtype != torch.float64 or activation_rms.device.type != "cpu"
        or tuple(down.shape)[1:] != (gates,) or down.ndim != 2
        or down.dtype != torch.float64 or down.device.type != "cpu"
        or not bool(torch.isfinite(activation_rms).all())
        or not bool(torch.isfinite(down).all()) or bool((activation_rms <= 0).any())
    ):
        raise ValueError("global-gate control inputs are malformed")
    budgets = tuple(int(value) for value in plan["selectors"]["budgets"])
    target_map = {
        int(key): int(value)
        for key, value in plan["selectors"]["target_rank_by_budget"].items()
    }
    if (
        set(budgets) != set(target_map) or max(budgets) > gates
        or documents != int(plan["cohorts"]["fit"]["contexts"])
        or probes != int(plan["operator"]["probes_per_half"])
    ):
        raise ValueError("global-gate plan dimensions changed")

    balanced = {name: gate.context_balance(value) for name, value in cells.items()}
    balanced_deranged = gate.context_balance(deranged_fit_first)
    ranks = tuple(target_map[budget] for budget in budgets)
    primary_fit = _ridge_scores_by_rank(balanced["fit_first"], ranks)
    primary_second = _ridge_scores_by_rank(balanced["fit_second"], ranks)
    deranged_scores = _ridge_scores_by_rank(balanced_deranged, ranks)
    raw_fit = _ridge_scores_by_rank(cells["fit_first"], ranks)
    response_energy = gate.column_energy_scores(balanced["fit_first"])
    activation_scores = activation_rms * torch.linalg.vector_norm(down, dim=0)
    random_scores = _hash_rank_scores(gates, int(plan["selectors"]["random_control_seed"]))

    bundles: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    losses: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    raw_diagnostics: dict[str, Any] = {}
    for budget in budgets:
        key, target_rank = str(budget), target_map[budget]
        score_vectors = {
            "primary": primary_fit[target_rank],
            "response_energy": response_energy,
            "activation_down": activation_scores,
            "factor_product_derangement": deranged_scores[target_rank],
            "hash_random": random_scores,
        }
        supports = {
            arm: (
                gate.hash_random_selection(
                    gates, budget, int(plan["selectors"]["random_control_seed"]),
                ) if arm == "hash_random" else gate.select_top(scores, budget)
            )
            for arm, scores in score_vectors.items()
        }
        second_support = gate.select_top(primary_second[target_rank], budget)
        stability[key] = {
            "support_jaccard": support_jaccard(supports["primary"], second_support),
            "fit_second_diagnostic_support": list(second_support),
            "score_rank_spearman": spearman_rank_correlation(
                primary_fit[target_rank], primary_second[target_rank],
            ),
            "fit_second_support_is_nonpromotive": True,
        }
        raw_diagnostics[key] = {
            "primary_support": list(gate.select_top(raw_fit[target_rank], budget)),
            "response_energy_support": list(gate.select_top(
                gate.column_energy_scores(cells["fit_first"]), budget,
            )),
            "currency": "raw response; nonpromotive",
        }
        bundles[key], summaries[key], losses[key] = {}, {}, {}
        for arm in ARM_NAMES:
            try:
                tensor_bundle, summary = _fit_bundle(
                    balanced["fit_first"], supports[arm], score_vectors[arm],
                    selector_currency=(
                        "context-balanced fit/first " + arm.replace("_", " ")
                    ), target_rank=target_rank,
                )
                cell_losses = {
                    cell: _per_document_losses(
                        balanced[cell], summary["support"],
                        tensor_bundle["css_coefficients"],
                        tensor_bundle["all_on_coefficients"],
                    )
                    for cell in CELL_NAMES[1:]
                }
                summary["cells"] = cell_losses
                bundles[key][arm] = tensor_bundle
                summaries[key][arm] = {
                    name: value for name, value in summary.items() if name != "cells"
                }
                losses[key][arm] = {**summary, "cells": cell_losses}
            except ValueError as error:
                failure = {
                    "status": "solver_rejected", "reason": str(error),
                    "support": list(supports[arm]), "target_rank": target_rank,
                }
                bundles[key][arm] = None
                summaries[key][arm] = failure
                losses[key][arm] = failure

    tensor_bundle = {
        "status": "fit_first_support_and_coefficients_frozen",
        "plan_fingerprint": plan["plan_fingerprint"],
        "budgets": bundles,
    }
    if fit_only:
        return {
            "status": "fit_bundle_complete_no_validation_opened",
            "plan_fingerprint": plan["plan_fingerprint"],
            "dimensions": {
                "fit_documents": documents, "probes": probes, "gates": gates,
            },
            "bundle_summaries": summaries,
            "stability": stability,
            "raw_nonpromotive_diagnostics": raw_diagnostics,
            "validation_metrics_computed": False,
        }, tensor_bundle

    records, comparisons = _comparison_records(losses, budgets)
    bootstrap_plan = plan["metrics"]["bootstrap"]
    planned_comparisons = len(budgets) * len(CONTROL_NAMES) * len(
        METRIC_NAMES
    ) * len(VALIDATION_CELLS)
    bootstrap = (
        _simultaneous_bootstrap(
            records, documents=documents,
            repetitions=int(bootstrap_plan["repetitions"]),
            seed=int(bootstrap_plan["seed"]),
            confidence=float(bootstrap_plan["simultaneous_confidence"]),
        ) if len(records) == planned_comparisons else {
            "status": "not_computed_incomplete_registered_family",
            "planned_comparisons": planned_comparisons,
            "comparisons_evaluated": len(records),
            "simultaneous_lcb": {},
        }
    )
    if bootstrap["status"] == "complete":
        for comparison_id, value in comparisons.items():
            value["simultaneous_lcb"] = bootstrap["simultaneous_lcb"][comparison_id]

    decision_plan = plan["decision"]
    every_observed_positive = len(records) == len(budgets) * len(CONTROL_NAMES) * 2 * 2 and all(
        float(record["observed_improvement"]) > 0 for record in records
    )
    budget_decisions = {}
    promoted = []
    for budget in budgets:
        key = str(budget)
        complete = all(losses[key][arm].get("status") == "fit_complete" for arm in ARM_NAMES)
        ids = [
            f"K{budget}:{control}:{metric}:{cell}"
            for control in CONTROL_NAMES for metric in METRIC_NAMES
            for cell in VALIDATION_CELLS
        ]
        lcb_pass = bootstrap["status"] == "complete" and complete and all(
            comparisons[name]["simultaneous_lcb"] > float(
                decision_plan["relative_improvement_lcb_over_every_control_minimum"]
            ) for name in ids
        )
        harm_pass = complete and all(
            comparisons[name]["worst_per_document_primary_minus_control_loss"] <= float(
                decision_plan["maximum_per_document_primary_minus_each_control_loss"]
            ) for name in ids
        )
        jaccard_pass = stability[key]["support_jaccard"] >= float(
            decision_plan["support_jaccard_minimum"]
        )
        passed = complete and lcb_pass and harm_pass and jaccard_pass and every_observed_positive
        budget_decisions[key] = {
            "all_arms_solver_complete": complete,
            "support_jaccard_pass": jaccard_pass,
            "every_simultaneous_lcb_pass": lcb_pass,
            "every_per_document_harm_pass": harm_pass,
            "all_budgets_every_observed_improvement_positive": every_observed_positive,
            "full_numeric_pass": passed,
        }
        if passed:
            promoted.append(budget)

    promoted_budget = min(promoted) if promoted else None
    result = {
        "status": "promoted" if promoted_budget is not None else "no_admitted_support",
        "scope": (
            "fit-frozen tangent response/all-on physical-gate support; no finite "
            "removal, CE, semantic, tensor-rank, or OOD claim"
        ),
        "plan_fingerprint": plan["plan_fingerprint"],
        "dimensions": {"documents_per_cell": documents, "probes": probes, "gates": gates},
        "bundle_summaries": summaries,
        "per_document_loss_ledgers": losses,
        "stability": stability,
        "raw_nonpromotive_diagnostics": raw_diagnostics,
        "comparisons": comparisons,
        "bootstrap": bootstrap,
        "decisions": {
            "budgets": budget_decisions,
            "promoted_budget": promoted_budget,
            "consequence_stage_authorized": False,
        },
        "publication": {
            "raw_logits_published": False,
            "raw_targets_published": False,
            "raw_vjps_published": False,
            "raw_responses_published": False,
            "per_document_sufficient_statistics_published": True,
        },
    }
    return result, tensor_bundle


def build_fit_gate_bundle(
    fit_first: torch.Tensor, fit_second: torch.Tensor, *,
    deranged_fit_first: torch.Tensor, activation_rms: torch.Tensor,
    down: torch.Tensor, plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the only outcome bundle before any validation response is available."""
    cells = {
        "fit_first": fit_first,
        "fit_second": fit_second,
        # These aliases satisfy the common shape validator but are never evaluated in
        # fit-only mode. They contain no validation outcome.
        "validation_first": fit_second,
        "validation_second": fit_second,
    }
    return analyze_global_gate_responses(
        cells, deranged_fit_first=deranged_fit_first,
        activation_rms=activation_rms, down=down, plan=plan, fit_only=True,
    )


def tensor_tree_equal(first: Any, second: Any) -> bool:
    """Exact recursive equality used to prove validation did not alter the bundle."""
    if torch.is_tensor(first) or torch.is_tensor(second):
        return bool(torch.is_tensor(first) and torch.is_tensor(second) and (
            first.dtype == second.dtype and first.shape == second.shape
            and torch.equal(first, second)
        ))
    if isinstance(first, dict) or isinstance(second, dict):
        return bool(isinstance(first, dict) and isinstance(second, dict) and (
            set(first) == set(second)
        ) and all(tensor_tree_equal(first[key], second[key]) for key in first))
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return bool(type(first) is type(second) and len(first) == len(second) and all(
            tensor_tree_equal(left, right) for left, right in zip(first, second, strict=True)
        ))
    return first == second


def validate_fit_gate_bundle(
    fit_summary: Mapping[str, Any], tensor_bundle: Mapping[str, Any],
    plan: Mapping[str, Any], *, replay_inputs: Mapping[str, torch.Tensor] | None = None,
) -> None:
    budgets = tuple(int(value) for value in plan["selectors"]["budgets"])
    gates = int(fit_summary.get("dimensions", {}).get("gates", -1))
    if (
        fit_summary.get("status") != "fit_bundle_complete_no_validation_opened"
        or fit_summary.get("validation_metrics_computed") is not False
        or fit_summary.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or tensor_bundle.get("status") != "fit_first_support_and_coefficients_frozen"
        or tensor_bundle.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or set(fit_summary.get("bundle_summaries", {})) != {str(k) for k in budgets}
        or set(tensor_bundle.get("budgets", {})) != {str(k) for k in budgets}
        or gates <= 0
    ):
        raise RuntimeError("fit gate bundle top-level schema changed")
    for budget in budgets:
        key = str(budget)
        summaries = fit_summary["bundle_summaries"][key]
        tensors = tensor_bundle["budgets"][key]
        if set(summaries) != set(ARM_NAMES) or set(tensors) != set(ARM_NAMES):
            raise RuntimeError("fit gate bundle arm schema changed")
        for arm in ARM_NAMES:
            summary, values = summaries[arm], tensors[arm]
            if summary.get("status") == "solver_rejected":
                if values is not None:
                    raise RuntimeError("solver-rejected gate arm retained tensors")
                continue
            if summary.get("status") != "fit_complete" or not isinstance(values, dict) or set(
                values
            ) != {"support", "selection_scores", "css_coefficients", "all_on_coefficients"}:
                raise RuntimeError("fit gate arm schema changed")
            support = values["support"]
            if (
                support.dtype != torch.int64 or tuple(support.shape) != (budget,)
                or len(set(support.tolist())) != budget
                or int(support.min()) < 0 or int(support.max()) >= gates
                or summary.get("support") != support.tolist()
                or gate.select_top(values["selection_scores"], budget) != tuple(
                    support.tolist()
                )
                or values["selection_scores"].dtype != torch.float64
                or tuple(values["selection_scores"].shape) != (gates,)
                or values["css_coefficients"].dtype != torch.float64
                or tuple(values["css_coefficients"].shape) != (budget, gates)
                or values["all_on_coefficients"].dtype != torch.float64
                or tuple(values["all_on_coefficients"].shape) != (budget,)
                or any(value.device.type != "cpu" or value.requires_grad or not bool(
                    torch.isfinite(value).all()
                ) for value in values.values())
                or summary.get("tensors") != {
                    name: tensor_descriptor(value) for name, value in values.items()
                }
            ):
                raise RuntimeError("fit gate arm tensor identity changed")
    if replay_inputs is not None:
        required = {
            "fit_first", "fit_second", "deranged_fit_first", "activation_rms", "down",
        }
        if set(replay_inputs) != required:
            raise RuntimeError("fit gate replay inputs changed")
        replay_summary, replay_bundle = build_fit_gate_bundle(
            replay_inputs["fit_first"], replay_inputs["fit_second"],
            deranged_fit_first=replay_inputs["deranged_fit_first"],
            activation_rms=replay_inputs["activation_rms"], down=replay_inputs["down"],
            plan=plan,
        )
        if replay_summary != dict(fit_summary) or not tensor_tree_equal(
            replay_bundle, tensor_bundle,
        ):
            raise RuntimeError("fit gate bundle does not replay frozen fit responses")


def validate_gate_analysis_result(
    result: Mapping[str, Any], plan: Mapping[str, Any], *,
    replay_inputs: Mapping[str, Any] | None = None,
) -> None:
    budgets = tuple(int(value) for value in plan["selectors"]["budgets"])
    documents = int(result.get("dimensions", {}).get("documents_per_cell", -1))
    expected_ids = {
        f"K{budget}:{control}:{metric}:{cell}"
        for budget in budgets for control in CONTROL_NAMES
        for metric in METRIC_NAMES for cell in VALIDATION_CELLS
    }
    comparisons = result.get("comparisons", {})
    if (
        result.get("status") not in {"promoted", "no_admitted_support"}
        or result.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or result.get("decisions", {}).get("consequence_stage_authorized") is not False
        or any(result.get("publication", {}).get(name) is not False for name in (
            "raw_logits_published", "raw_targets_published", "raw_vjps_published",
            "raw_responses_published",
        ))
        or not set(comparisons) <= expected_ids or documents <= 1
        or set(result.get("decisions", {}).get("budgets", {})) != {str(k) for k in budgets}
    ):
        raise RuntimeError("global-gate scientific result top-level schema changed")
    ledgers = result.get("per_document_loss_ledgers", {})
    if set(ledgers) != {str(k) for k in budgets}:
        raise RuntimeError("global-gate per-document budget ledger changed")
    for budget in budgets:
        key = str(budget)
        if set(ledgers[key]) != set(ARM_NAMES):
            raise RuntimeError("global-gate per-document arm ledger changed")
        for arm in ARM_NAMES:
            arm_value = ledgers[key][arm]
            if arm_value.get("status") != "fit_complete":
                continue
            if set(arm_value.get("cells", {})) != set(CELL_NAMES[1:]):
                raise RuntimeError("global-gate evaluation cell ledger changed")
            for cell in CELL_NAMES[1:]:
                for metric in METRIC_NAMES:
                    record = arm_value["cells"][cell][metric]
                    if any(len(record[name]) != documents for name in (
                        "numerator", "denominator", "ratio",
                    )) or any(float(value) <= 0 for value in record["denominator"]):
                        raise RuntimeError("global-gate document sufficient statistics changed")
                    pooled = sum(record["numerator"]) / max(sum(record["denominator"]), 1e-30)
                    if abs(pooled - float(record["pooled_loss"])) > 1e-12 * max(1.0, abs(pooled)):
                        raise RuntimeError("global-gate pooled loss does not replay documents")
    bootstrap = result.get("bootstrap", {})
    planned_comparisons = len(expected_ids)
    replay_records = []
    replay_comparisons = {}
    for comparison_id, published in comparisons.items():
        prefix, control, metric, cell = comparison_id.split(":")
        key = prefix.removeprefix("K")
        primary = ledgers[key]["primary"]["cells"][cell][metric]
        baseline = ledgers[key][control]["cells"][cell][metric]
        primary_loss = float(primary["pooled_loss"])
        control_loss = float(baseline["pooled_loss"])
        improvement = (control_loss - primary_loss) / max(control_loss, 1e-30)
        harm = float((
            torch.tensor(primary["ratio"], dtype=torch.float64)
            - torch.tensor(baseline["ratio"], dtype=torch.float64)
        ).max())
        expected_public = {
            "id": comparison_id, "budget": int(key), "control": control,
            "metric": metric, "cell": cell, "observed_improvement": improvement,
            "worst_per_document_primary_minus_control_loss": harm,
        }
        if any(published.get(name) != value for name, value in expected_public.items()):
            raise RuntimeError("global-gate comparison does not replay loss ledgers")
        replay_comparisons[comparison_id] = expected_public
        replay_records.append({
            **expected_public,
            "primary_numerator": torch.tensor(primary["numerator"], dtype=torch.float64),
            "primary_denominator": torch.tensor(primary["denominator"], dtype=torch.float64),
            "control_numerator": torch.tensor(baseline["numerator"], dtype=torch.float64),
            "control_denominator": torch.tensor(baseline["denominator"], dtype=torch.float64),
        })
    if len(comparisons) == planned_comparisons:
        expected_bootstrap = plan["metrics"]["bootstrap"]
        replay_bootstrap = _simultaneous_bootstrap(
            replay_records, documents=documents,
            repetitions=int(expected_bootstrap["repetitions"]),
            seed=int(expected_bootstrap["seed"]),
            confidence=float(expected_bootstrap["simultaneous_confidence"]),
        )
        if (
            bootstrap.get("status") != "complete"
            or bootstrap.get("repetitions") != expected_bootstrap["repetitions"]
            or bootstrap.get("seed") != expected_bootstrap["seed"]
            or bootstrap.get("critical_order_statistic_one_indexed") != math.ceil(
                expected_bootstrap["simultaneous_confidence"]
                * expected_bootstrap["repetitions"]
            )
            or bootstrap.get("comparisons_evaluated") != len(comparisons)
            or set(bootstrap.get("simultaneous_lcb", {})) != set(comparisons)
            or bootstrap != replay_bootstrap
            or any(
                comparisons[name].get("simultaneous_lcb")
                != replay_bootstrap["simultaneous_lcb"][name]
                for name in comparisons
            )
        ):
            raise RuntimeError("global-gate simultaneous bootstrap schema changed")
    elif (
        bootstrap.get("status") != "not_computed_incomplete_registered_family"
        or bootstrap.get("planned_comparisons") != planned_comparisons
        or bootstrap.get("comparisons_evaluated") != len(comparisons)
        or bootstrap.get("simultaneous_lcb") != {}
        or any("simultaneous_lcb" in value for value in comparisons.values())
    ):
        raise RuntimeError("incomplete global-gate family reported a registered LCB")
    promoted = result["decisions"].get("promoted_budget")
    passing = [
        budget for budget in budgets
        if result["decisions"]["budgets"][str(budget)].get("full_numeric_pass") is True
    ]
    if promoted != (min(passing) if passing else None) or (
        (promoted is None) != (result["status"] == "no_admitted_support")
    ):
        raise RuntimeError("global-gate promotion decision does not replay budget gates")
    decision_plan = plan["decision"]
    every_positive = len(comparisons) == planned_comparisons and all(
        float(value["observed_improvement"]) > 0 for value in comparisons.values()
    )
    for budget in budgets:
        key = str(budget)
        complete = all(ledgers[key][arm].get("status") == "fit_complete" for arm in ARM_NAMES)
        ids = [name for name in expected_ids if name.startswith(f"K{budget}:")]
        expected_decision = {
            "all_arms_solver_complete": complete,
            "support_jaccard_pass": result["stability"][key]["support_jaccard"] >= float(
                decision_plan["support_jaccard_minimum"]
            ),
            "every_simultaneous_lcb_pass": bootstrap.get("status") == "complete" and complete
            and all(comparisons[name]["simultaneous_lcb"] > float(
                decision_plan["relative_improvement_lcb_over_every_control_minimum"]
            ) for name in ids),
            "every_per_document_harm_pass": complete and all(
                comparisons[name]["worst_per_document_primary_minus_control_loss"] <= float(
                    decision_plan["maximum_per_document_primary_minus_each_control_loss"]
                ) for name in ids
            ),
            "all_budgets_every_observed_improvement_positive": every_positive,
        }
        expected_decision["full_numeric_pass"] = all(expected_decision.values())
        if result["decisions"]["budgets"][key] != expected_decision:
            raise RuntimeError("global-gate budget decision does not replay metrics")
    if replay_inputs is not None:
        required = {"cells", "deranged_fit_first", "activation_rms", "down", "bundle"}
        if set(replay_inputs) != required:
            raise RuntimeError("global-gate scientific replay inputs changed")
        replay_result, replay_bundle = analyze_global_gate_responses(
            replay_inputs["cells"],
            deranged_fit_first=replay_inputs["deranged_fit_first"],
            activation_rms=replay_inputs["activation_rms"], down=replay_inputs["down"],
            plan=plan,
        )
        if replay_result != dict(result) or not tensor_tree_equal(
            replay_bundle, replay_inputs["bundle"],
        ):
            raise RuntimeError("global-gate result does not replay frozen response tensors")
