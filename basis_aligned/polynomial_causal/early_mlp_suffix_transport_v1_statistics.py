"""Pure CPU statistics for early-MLP suffix transport v1.

The collector must emit raw source-document/occurrence sufficient statistics.
This module alone pools nonlinear response metrics, constructs the shared cluster
bootstrap, computes literal percentile intervals, and evaluates intervention gates.
It never imports a model/runtime and never selects a program or amplitude.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

import early_mlp_suffix_transport_v1 as contract


BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260832
DENOMINATOR_FLOOR = 1e-12


def _vector(name: str, value: Any, length: int | None = None) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 1 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector")
    result = value.detach().cpu().double().contiguous()
    if length is not None and len(result) != length:
        raise ValueError(f"{name} length changed")
    return result


def document_bootstrap_weights(
    records: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> torch.Tensor:
    """Return [draw,row] multiplicities from a source-document cluster bootstrap."""

    if not records or draws <= 0:
        raise ValueError("bootstrap requires records and positive draws")
    document_to_index: dict[str, int] = {}
    row_documents = []
    for record in records:
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("bootstrap provenance lacks document_id")
        if document_id not in document_to_index:
            document_to_index[document_id] = len(document_to_index)
        row_documents.append(document_to_index[document_id])
    n_documents = len(document_to_index)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randint(
        0, n_documents, (draws, n_documents), generator=generator,
    )
    document_weights = torch.zeros(draws, n_documents, dtype=torch.float64)
    document_weights.scatter_add_(
        1, samples, torch.ones_like(samples, dtype=torch.float64),
    )
    row_index = torch.tensor(row_documents, dtype=torch.long)
    weights = document_weights.index_select(1, row_index)
    if not torch.equal(weights.sum(dim=1), torch.full(
        (draws,), len(records), dtype=torch.float64,
    )):
        # This equality holds only when every document has one row. Cluster resampling
        # legitimately changes total row support for variable-size documents.
        expected_document_draws = document_weights.sum(dim=1)
        if not torch.equal(
            expected_document_draws,
            torch.full((draws,), n_documents, dtype=torch.float64),
        ):
            raise RuntimeError("bootstrap document multiplicities do not sum correctly")
    return weights.contiguous()


def linear_interval(samples: torch.Tensor, *, alpha: float = 0.05) -> tuple[float, float]:
    values = _vector("bootstrap samples", samples)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    quantiles = torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64)
    result = torch.quantile(values, quantiles, interpolation="linear")
    return float(result[0]), float(result[1])


def pooled_mean_draws(
    row_sum: torch.Tensor, row_count: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    sums = _vector("row_sum", row_sum)
    counts = _vector("row_count", row_count, len(sums))
    if weights.ndim != 2 or weights.shape[1] != len(sums) or not (
        torch.isfinite(weights).all()
    ) or bool((weights < 0).any()):
        raise ValueError("bootstrap weights are invalid")
    numerator = weights.double() @ sums
    denominator = weights.double() @ counts
    if bool((denominator <= DENOMINATOR_FLOOR).any()):
        raise ValueError("pooled mean denominator is too small")
    return (numerator / denominator).contiguous()


def paired_pooled_mean_difference(
    left_sum: torch.Tensor,
    left_count: torch.Tensor,
    right_sum: torch.Tensor,
    right_count: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return pooled_mean_draws(left_sum, left_count, weights) - pooled_mean_draws(
        right_sum, right_count, weights,
    )


RESPONSE_KEYS = ("error_sum", "teacher_sum", "student_sum", "dot_sum")
RESPONSE_SCHEMA = (*RESPONSE_KEYS, "unit_identity")


def _unit_identity(statistics: Mapping[str, Any]) -> str:
    identity = statistics.get("unit_identity")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("response unit identity is malformed")
    try:
        int(identity, 16)
    except ValueError as error:
        raise ValueError("response unit identity is malformed") from error
    return identity


def validate_response_sufficient_statistics(
    statistics: Mapping[str, Any], *, length: int | None = None,
) -> dict[str, torch.Tensor]:
    if set(statistics) != set(RESPONSE_SCHEMA):
        raise ValueError("response sufficient-statistic schema changed")
    _unit_identity(statistics)
    output: dict[str, torch.Tensor] = {}
    for key in RESPONSE_KEYS:
        output[key] = _vector(key, statistics[key], length)
        length = len(output[key]) if length is None else length
    if bool((output["error_sum"] < 0).any()) or bool(
        (output["teacher_sum"] < 0).any()
    ) or bool((output["student_sum"] < 0).any()):
        raise ValueError("response square sums must be nonnegative")
    reconstructed_error = (
        output["student_sum"] + output["teacher_sum"] - 2 * output["dot_sum"]
    )
    scale = torch.maximum(
        torch.ones_like(reconstructed_error),
        output["student_sum"] + output["teacher_sum"] + 2 * output["dot_sum"].abs(),
    )
    if bool((output["error_sum"] - reconstructed_error).abs().gt(1e-9 * scale).any()):
        raise ValueError("response error/dot sufficient statistics are inconsistent")
    cauchy = torch.sqrt(output["student_sum"] * output["teacher_sum"])
    if bool((output["dot_sum"].abs() > cauchy + 1e-9 * scale).any()):
        raise ValueError("response dot sum violates Cauchy bound")
    return output


def pooled_response_draws(
    statistics: Mapping[str, Any], weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    values = validate_response_sufficient_statistics(statistics)
    if weights.ndim != 2 or weights.shape[1] != len(values["error_sum"]) or not (
        torch.isfinite(weights).all()
    ) or bool((weights < 0).any()):
        raise ValueError("response bootstrap weights are invalid")
    weights = weights.detach().cpu().double()
    pooled = {key: weights @ value for key, value in values.items()}
    if bool((pooled["teacher_sum"] <= DENOMINATOR_FLOOR).any()):
        raise ValueError("teacher-response denominator is too small")
    cosine_denominator = torch.sqrt(pooled["student_sum"]) * torch.sqrt(
        pooled["teacher_sum"]
    )
    if bool((cosine_denominator <= DENOMINATOR_FLOOR).any()):
        raise ValueError("response cosine denominator is too small")
    nre = torch.sqrt(pooled["error_sum"] / pooled["teacher_sum"])
    return {
        **pooled,
        "nre": nre.contiguous(),
        "r2": (1 - nre.square()).contiguous(),
        "cosine": (pooled["dot_sum"] / cosine_denominator).contiguous(),
    }


def pooled_response_point(statistics: Mapping[str, Any]) -> dict[str, float]:
    values = validate_response_sufficient_statistics(statistics)
    weights = torch.ones(1, len(values["error_sum"]), dtype=torch.float64)
    draws = pooled_response_draws(statistics, weights)
    return {key: float(value[0]) for key, value in draws.items()}


def pooled_ratio_draws(
    row_numerator: torch.Tensor, row_denominator: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    numerator = _vector("row_numerator", row_numerator)
    denominator = _vector("row_denominator", row_denominator, len(numerator))
    if weights.ndim != 2 or weights.shape[1] != len(numerator) or not (
        torch.isfinite(weights).all()
    ) or bool((weights < 0).any()):
        raise ValueError("ratio bootstrap weights are invalid")
    pooled_numerator = weights.double() @ numerator
    pooled_denominator = weights.double() @ denominator
    if bool((pooled_denominator <= DENOMINATOR_FLOOR).any()):
        raise ValueError("pooled ratio denominator is too small")
    return (pooled_numerator / pooled_denominator).contiguous()


def response_family_summary(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    weights: torch.Tensor,
) -> dict[str, Any]:
    """Return one aligned response family's estimates without a promotive decision."""

    if _unit_identity(baseline) != _unit_identity(candidate):
        raise ValueError("response families use different ordered intervention units")
    base_draws = pooled_response_draws(baseline, weights)
    candidate_draws = pooled_response_draws(candidate, weights)
    base_point = pooled_response_point(baseline)
    candidate_point = pooled_response_point(candidate)
    improvement_draws = base_draws["nre"] - candidate_draws["nre"]
    improvement_point = base_point["nre"] - candidate_point["nre"]
    improvement_interval = linear_interval(improvement_draws)
    nre_interval = linear_interval(candidate_draws["nre"])
    r2_interval = linear_interval(candidate_draws["r2"])
    return {
        "unit_identity": _unit_identity(baseline),
        "baseline_point": base_point,
        "candidate_point": candidate_point,
        "nre_improvement_point": improvement_point,
        "nre_improvement_interval95": improvement_interval,
        "candidate_nre_interval95": nre_interval,
        "candidate_r2_interval95": r2_interval,
    }


TRANSPORT_OBSERVATIONAL_GATES = (
    "lt_beats_ll_ce_n",
    "lt_beats_ll_ce_e",
    "lt_beats_ll_teacher_kl_n",
    "copy_bound",
    "frequency_bounds",
    "common_integrity",
)


def transport_route_decision(
    *,
    code_baseline: Mapping[str, Any],
    code_candidate: Mapping[str, Any],
    logit_baseline: Mapping[str, Any],
    logit_candidate: Mapping[str, Any],
    logit_nulls: Sequence[Mapping[str, Any]],
    weights: torch.Tensor,
    calibration_passed: bool,
    observational_gates: Mapping[str, bool],
) -> dict[str, Any]:
    """Conjoin the complete registered transport route from aligned raw responses."""

    if type(calibration_passed) is not bool:
        raise ValueError("calibration_passed must be a literal boolean")
    if len(logit_nulls) != 20:
        raise ValueError("transport requires exactly 20 null response records")
    if set(observational_gates) != set(TRANSPORT_OBSERVATIONAL_GATES) or any(
        type(value) is not bool for value in observational_gates.values()
    ):
        raise ValueError("transport observational gate schema changed")
    identities = {
        _unit_identity(record)
        for record in (
            code_baseline, code_candidate, logit_baseline, logit_candidate,
            *logit_nulls,
        )
    }
    if len(identities) != 1:
        raise ValueError("transport arms do not share ordered intervention units")

    code = response_family_summary(code_baseline, code_candidate, weights)
    logit = response_family_summary(logit_baseline, logit_candidate, weights)
    null_summaries = [
        response_family_summary(logit_baseline, null, weights) for null in logit_nulls
    ]
    null_improvements = torch.tensor(
        [summary["nre_improvement_point"] for summary in null_summaries],
        dtype=torch.float64,
    )
    finite_null_rank = contract.finite_null_rank(
        logit["nre_improvement_point"], null_improvements,
    )
    gates = {
        "calibration": bool(calibration_passed),
        "code_nre_improvement_point_positive": code["nre_improvement_point"] > 0,
        "code_nre_improvement_lcb_positive": code["nre_improvement_interval95"][0] > 0,
        "logit_nre_improvement_point_positive": logit["nre_improvement_point"] > 0,
        "logit_nre_improvement_lcb_positive": logit["nre_improvement_interval95"][0] > 0,
        "logit_nre_point_le_half": logit["candidate_point"]["nre"] <= 0.5,
        "logit_nre_ucb_le_half": logit["candidate_nre_interval95"][1] <= 0.5,
        "logit_r2_point_ge_three_quarters": logit["candidate_point"]["r2"] >= 0.75,
        "logit_r2_lcb_ge_three_quarters": logit["candidate_r2_interval95"][0] >= 0.75,
        "finite_null_rank_one": finite_null_rank == 1,
        **dict(observational_gates),
    }
    return {
        "unit_identity": identities.pop(),
        "code_response": code,
        "logit_response": logit,
        "null_logit_nre_improvements": null_improvements,
        "finite_null_rank": finite_null_rank,
        "gates": gates,
        "passes": all(gates.values()),
    }
