from __future__ import annotations

import pytest
import torch

import early_mlp_suffix_transport_v1_statistics as statistics


def test_document_bootstrap_is_clustered_reproducible_and_variable_support() -> None:
    records = [
        {"document_id": "a"}, {"document_id": "a"},
        {"document_id": "b"}, {"document_id": "c"},
    ]
    first = statistics.document_bootstrap_weights(records, draws=64, seed=7)
    second = statistics.document_bootstrap_weights(records, draws=64, seed=7)
    assert torch.equal(first, second)
    assert tuple(first.shape) == (64, 4)
    assert torch.equal(first[:, 0], first[:, 1])
    # Three documents are sampled per draw, while row count varies because a has 2 rows.
    assert set(first.sum(dim=1).tolist()).issubset({3.0, 4.0, 5.0, 6.0})


def test_linear_interval_uses_literal_float64_quantiles() -> None:
    samples = torch.arange(2000, dtype=torch.float64)
    observed = statistics.linear_interval(samples)
    expected = torch.quantile(
        samples, torch.tensor([0.025, 0.975], dtype=torch.float64),
        interpolation="linear",
    )
    assert observed == (float(expected[0]), float(expected[1]))
    bonferroni = statistics.linear_interval(samples, alpha=0.025)
    expected_b = torch.quantile(
        samples, torch.tensor([0.0125, 0.9875], dtype=torch.float64),
        interpolation="linear",
    )
    assert bonferroni == (float(expected_b[0]), float(expected_b[1]))


def test_pooled_mean_recomputes_ratio_of_sums_inside_each_draw() -> None:
    row_sum = torch.tensor([1.0, 100.0], dtype=torch.float64)
    row_count = torch.tensor([1.0, 100.0], dtype=torch.float64)
    weights = torch.tensor([[1.0, 1.0], [2.0, 0.0]], dtype=torch.float64)
    draws = statistics.pooled_mean_draws(row_sum, row_count, weights)
    assert torch.allclose(draws, torch.tensor([101 / 101, 1.0], dtype=torch.float64))


def _response(scale: float, n_rows: int = 4) -> dict[str, torch.Tensor]:
    # Teacher unit norm per row, collinear student with the specified scale.
    teacher = torch.ones(n_rows, dtype=torch.float64)
    student = torch.full((n_rows,), scale * scale, dtype=torch.float64)
    error = torch.full((n_rows,), (scale - 1) ** 2, dtype=torch.float64)
    dot = torch.full((n_rows,), scale, dtype=torch.float64)
    return {
        "unit_identity": "a" * 64,
        "error_sum": error,
        "teacher_sum": teacher,
        "student_sum": student,
        "dot_sum": dot,
    }


def _output_kl(ratio: float, n_rows: int = 4) -> dict[str, torch.Tensor]:
    return {
        "unit_identity": "a" * 64,
        "numerator_sum": torch.full((n_rows,), ratio, dtype=torch.float64),
        "denominator_sum": torch.ones(n_rows, dtype=torch.float64),
    }


def test_response_metrics_pool_sums_before_nonlinearity() -> None:
    response = _response(0.5)
    point = statistics.pooled_response_point(response)
    assert abs(point["nre"] - 0.5) < 1e-14
    assert abs(point["r2"] - 0.75) < 1e-14
    assert abs(point["cosine"] - 1.0) < 1e-14
    weights = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 3.0, 0.0]])
    draws = statistics.pooled_response_draws(response, weights)
    assert torch.allclose(draws["nre"], torch.full((2,), 0.5, dtype=torch.float64))


def test_response_denominators_fail_closed() -> None:
    response = {
        "unit_identity": "a" * 64,
        "error_sum": torch.ones(4, dtype=torch.float64),
        "teacher_sum": torch.zeros(4, dtype=torch.float64),
        "student_sum": torch.ones(4, dtype=torch.float64),
        "dot_sum": torch.zeros(4, dtype=torch.float64),
    }
    with pytest.raises(ValueError, match="teacher-response denominator"):
        statistics.pooled_response_point(response)


def test_pooled_output_ratio_recomputes_under_weights() -> None:
    numerator = torch.tensor([1.0, 3.0])
    denominator = torch.tensor([2.0, 9.0])
    weights = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
    ratio = statistics.pooled_ratio_draws(numerator, denominator, weights)
    assert torch.allclose(ratio, torch.tensor([4 / 11, 1 / 3], dtype=torch.float64))


def test_transport_decision_conjoins_both_modalities_nulls_and_observational_gates() -> None:
    baseline = _response(0.25)
    candidate = _response(0.75)
    weights = torch.ones(2000, 4, dtype=torch.float64)
    nulls = [_response(0.30 + 0.005 * index) for index in range(20)]
    observational = {
        key: True for key in statistics.TRANSPORT_OBSERVATIONAL_GATES
    }
    summary = statistics.transport_route_decision(
        code_baseline=baseline,
        code_candidate=candidate,
        logit_baseline=baseline,
        logit_candidate=candidate,
        logit_nulls=nulls,
        output_kl_baseline=_output_kl(0.8),
        output_kl_candidate=_output_kl(0.4),
        output_kl_nulls=[_output_kl(0.9 + index / 100) for index in range(20)],
        weights=weights,
        calibration_passed=True,
        observational_gates=observational,
    )
    assert summary["logit_response"]["candidate_point"]["nre"] == 0.25
    assert summary["logit_response"]["candidate_point"]["r2"] == 0.9375
    assert summary["finite_null_rank"] == 1
    assert summary["output_kl_response"]["candidate"]["point"] == 0.4
    assert summary["passes"]

    failed = statistics.transport_route_decision(
        code_baseline=baseline,
        code_candidate=candidate,
        logit_baseline=baseline,
        logit_candidate=candidate,
        logit_nulls=nulls,
        output_kl_baseline=_output_kl(0.8),
        output_kl_candidate=_output_kl(0.4),
        output_kl_nulls=[_output_kl(0.9 + index / 100) for index in range(20)],
        weights=weights,
        calibration_passed=False,
        observational_gates=observational,
    )
    assert not failed["passes"]
    assert not failed["gates"]["calibration"]
    with pytest.raises(ValueError, match="exactly 20"):
        statistics.transport_route_decision(
            code_baseline=baseline,
            code_candidate=candidate,
            logit_baseline=baseline,
            logit_candidate=candidate,
            logit_nulls=nulls[:19],
            output_kl_baseline=_output_kl(0.8),
            output_kl_candidate=_output_kl(0.4),
            output_kl_nulls=[_output_kl(0.9 + index / 100) for index in range(19)],
            weights=weights,
            calibration_passed=True,
            observational_gates=observational,
        )
    with pytest.raises(ValueError, match="literal boolean"):
        statistics.transport_route_decision(
            code_baseline=baseline,
            code_candidate=candidate,
            logit_baseline=baseline,
            logit_candidate=candidate,
            logit_nulls=nulls,
            output_kl_baseline=_output_kl(0.8),
            output_kl_candidate=_output_kl(0.4),
            output_kl_nulls=[_output_kl(0.9 + index / 100) for index in range(20)],
            weights=weights,
            calibration_passed="false",  # type: ignore[arg-type]
            observational_gates=observational,
        )


def test_response_family_summary_is_nonpromotive_and_checks_unit_alignment() -> None:
    baseline = _response(0.25)
    candidate = _response(0.75)
    weights = torch.ones(3, 4, dtype=torch.float64)
    summary = statistics.response_family_summary(baseline, candidate, weights)
    assert "passes" not in summary and "gates" not in summary
    candidate["unit_identity"] = "b" * 64
    with pytest.raises(ValueError, match="different ordered"):
        statistics.response_family_summary(baseline, candidate, weights)


def test_response_statistics_enforce_error_identity_and_cauchy_bound() -> None:
    response = _response(0.5)
    response["error_sum"][0] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        statistics.pooled_response_point(response)
    response = _response(0.5)
    response["dot_sum"][0] = 10
    response["error_sum"][0] = (
        response["student_sum"][0] + response["teacher_sum"][0]
        - 2 * response["dot_sum"][0]
    )
    with pytest.raises(ValueError):
        statistics.pooled_response_point(response)


def test_output_kl_statistics_fail_closed_and_pool_before_ratio() -> None:
    output_kl = _output_kl(0.5)
    output_kl["numerator_sum"] = torch.tensor([1.0, 3.0, 5.0, 7.0])
    output_kl["denominator_sum"] = torch.tensor([2.0, 9.0, 10.0, 14.0])
    weights = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0]])
    draws = statistics.pooled_output_kl_draws(output_kl, weights)
    assert torch.allclose(draws, torch.tensor([4 / 11, 17 / 34], dtype=torch.float64))
    output_kl["denominator_sum"].zero_()
    with pytest.raises(ValueError, match="pooled ratio denominator"):
        statistics.output_kl_summary(output_kl, torch.ones(2, 4))
