from __future__ import annotations

import copy

import pytest
import torch

import mlp1_global_gate_analysis as analysis


def tiny_plan() -> dict:
    return {
        "plan_fingerprint": "tiny-plan",
        "operator": {"probes_per_half": 3},
        "cohorts": {"fit": {"contexts": 3}},
        "selectors": {
            "budgets": [1, 2],
            "target_rank_by_budget": {"1": 1, "2": 2},
            "random_control_seed": 97,
        },
        "metrics": {"bootstrap": {
            "repetitions": 200,
            "seed": 101,
            "simultaneous_confidence": 0.95,
        }},
        "decision": {
            "support_jaccard_minimum": 0.0,
            "relative_improvement_lcb_over_every_control_minimum": -1e9,
            "maximum_per_document_primary_minus_each_control_loss": 1e9,
        },
    }


def synthetic_inputs():
    generator = torch.Generator().manual_seed(811)
    base = torch.randn(3, 3, 4, generator=generator, dtype=torch.float64)
    cells = {
        "fit_first": base,
        "fit_second": base + 0.01 * torch.randn(
            base.shape, generator=generator, dtype=torch.float64,
        ),
        "validation_first": base + 0.02 * torch.randn(
            base.shape, generator=generator, dtype=torch.float64,
        ),
        "validation_second": base + 0.03 * torch.randn(
            base.shape, generator=generator, dtype=torch.float64,
        ),
    }
    deranged = torch.roll(base, 1, dims=2)
    rms = torch.tensor([1.0, 1.2, 0.9, 1.1], dtype=torch.float64)
    down = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    return cells, deranged, rms, down


def test_analysis_is_deterministic_and_fit_bundle_is_one_way() -> None:
    cells, deranged, rms, down = synthetic_inputs()
    first_result, first_bundle = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    second_result, second_bundle = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    assert first_result == second_result
    for budget in ("1", "2"):
        for arm in analysis.ARM_NAMES:
            left, right = first_bundle["budgets"][budget][arm], second_bundle["budgets"][budget][arm]
            if left is None:
                assert right is None
            else:
                for key in left:
                    assert torch.equal(left[key], right[key])
        assert first_result["stability"][budget]["fit_second_support_is_nonpromotive"]
    assert first_result["bootstrap"]["critical_order_statistic_one_indexed"] == 190
    assert first_result["publication"]["raw_responses_published"] is False
    assert len(first_result["per_document_loss_ledgers"]["1"]["primary"]["cells"][
        "validation_first"
    ]["css"]["numerator"]) == 3


def test_validation_changes_metrics_but_never_the_frozen_bundle() -> None:
    cells, deranged, rms, down = synthetic_inputs()
    original_result, original_bundle = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    changed = copy.deepcopy(cells)
    changed["validation_first"] = changed["validation_first"] * 3.0 + 0.7
    changed_result, changed_bundle = analysis.analyze_global_gate_responses(
        changed, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    for budget in ("1", "2"):
        for arm in analysis.ARM_NAMES:
            left, right = original_bundle["budgets"][budget][arm], changed_bundle["budgets"][budget][arm]
            if left is None:
                assert right is None
            else:
                for key in left:
                    assert torch.equal(left[key], right[key])
    assert original_result["comparisons"] != changed_result["comparisons"]


def test_fit_bundle_can_be_frozen_before_any_validation_response_exists() -> None:
    cells, deranged, rms, down = synthetic_inputs()
    fit_summary, frozen = analysis.build_fit_gate_bundle(
        cells["fit_first"], cells["fit_second"], deranged_fit_first=deranged,
        activation_rms=rms, down=down, plan=tiny_plan(),
    )
    result, rebuilt = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    assert fit_summary["status"] == "fit_bundle_complete_no_validation_opened"
    assert fit_summary["validation_metrics_computed"] is False
    assert analysis.tensor_tree_equal(frozen, rebuilt)
    assert result["bootstrap"]["status"] == "complete"
    analysis.validate_fit_gate_bundle(fit_summary, frozen, tiny_plan())
    analysis.validate_gate_analysis_result(result, tiny_plan())

    frozen["budgets"]["1"]["primary"]["support"][0] = 99
    with pytest.raises(RuntimeError, match="tensor identity"):
        analysis.validate_fit_gate_bundle(fit_summary, frozen, tiny_plan())


def test_fit_bundle_semantic_replay_rejects_self_consistent_tensor_corruption() -> None:
    cells, deranged, rms, down = synthetic_inputs()
    fit_summary, frozen = analysis.build_fit_gate_bundle(
        cells["fit_first"], cells["fit_second"], deranged_fit_first=deranged,
        activation_rms=rms, down=down, plan=tiny_plan(),
    )
    corrupted = copy.deepcopy(frozen)
    values = corrupted["budgets"]["1"]["primary"]
    values["css_coefficients"].zero_()
    fit_summary = copy.deepcopy(fit_summary)
    fit_summary["bundle_summaries"]["1"]["primary"]["tensors"][
        "css_coefficients"
    ] = analysis.tensor_descriptor(values["css_coefficients"])
    with pytest.raises(RuntimeError, match="does not replay frozen fit responses"):
        analysis.validate_fit_gate_bundle(
            fit_summary, corrupted, tiny_plan(), replay_inputs={
                "fit_first": cells["fit_first"], "fit_second": cells["fit_second"],
                "deranged_fit_first": deranged, "activation_rms": rms, "down": down,
            },
        )


def test_result_validator_recomputes_comparisons_bootstrap_and_decisions() -> None:
    cells, deranged, rms, down = synthetic_inputs()
    result, bundle = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    corrupted = copy.deepcopy(result)
    name = next(iter(corrupted["comparisons"]))
    corrupted["comparisons"][name]["observed_improvement"] = 12345.0
    corrupted["comparisons"][name]["simultaneous_lcb"] = 12345.0
    corrupted["bootstrap"]["simultaneous_lcb"][name] = 12345.0
    with pytest.raises(RuntimeError, match="does not replay loss ledgers"):
        analysis.validate_gate_analysis_result(
            corrupted, tiny_plan(), replay_inputs={
                "cells": cells, "deranged_fit_first": deranged,
                "activation_rms": rms, "down": down, "bundle": bundle,
            },
        )


def test_solver_rejection_never_bootstraps_a_reduced_registered_family(
    monkeypatch,
) -> None:
    cells, deranged, rms, down = synthetic_inputs()
    original = analysis._fit_bundle

    def reject_rank_one(*args, target_rank, **kwargs):
        if target_rank == 1:
            raise ValueError("synthetic registered-arm rejection")
        return original(*args, target_rank=target_rank, **kwargs)

    monkeypatch.setattr(analysis, "_fit_bundle", reject_rank_one)
    result, _ = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    assert len(result["comparisons"]) == 16
    assert result["bootstrap"] == {
        "status": "not_computed_incomplete_registered_family",
        "planned_comparisons": 32,
        "comparisons_evaluated": 16,
        "simultaneous_lcb": {},
    }
    assert all(
        "simultaneous_lcb" not in value for value in result["comparisons"].values()
    )
    assert result["decisions"]["promoted_budget"] is None
    analysis.validate_gate_analysis_result(result, tiny_plan())


def test_controls_are_separate_and_no_oracle_union_or_routing_is_published() -> None:
    cells, deranged, rms, down = synthetic_inputs()
    result, bundle = analysis.analyze_global_gate_responses(
        cells, deranged_fit_first=deranged, activation_rms=rms, down=down,
        plan=tiny_plan(),
    )
    assert set(bundle["budgets"]["1"]) == set(analysis.ARM_NAMES)
    ids = set(result["comparisons"])
    assert ids == {
        f"K{budget}:{control}:{metric}:{cell}"
        for budget in (1, 2) for control in analysis.CONTROL_NAMES
        for metric in analysis.METRIC_NAMES for cell in analysis.VALIDATION_CELLS
    }
    serialized = str(result).lower()
    assert "best_control" not in serialized
    assert "union" not in serialized
    assert "route" not in serialized


def test_spearman_uses_average_ranks_and_handles_constant_scores() -> None:
    first = torch.tensor([1.0, 1.0, 3.0, 4.0])
    second = torch.tensor([2.0, 2.0, 4.0, 5.0])
    assert analysis.spearman_rank_correlation(first, second) == 1.0
    assert analysis.spearman_rank_correlation(torch.ones(4), second) is None


def test_multirank_ridge_scores_use_registered_positive_tail_and_scale_invariance() -> None:
    response = torch.diag(torch.tensor(
        [1.0, 1e-16, 0.0, 0.0], dtype=torch.float64,
    )).reshape(1, 4, 4)
    scores = analysis._ridge_scores_by_rank(response, (1,))[1]
    assert torch.allclose(
        scores, torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=torch.float64),
        atol=1e-14, rtol=1e-14,
    )
    assert torch.allclose(
        analysis._ridge_scores_by_rank(response * 1e6, (1,))[1], scores,
        atol=1e-14, rtol=1e-14,
    )


def test_basic_max_error_bootstrap_uses_one_shared_nearest_rank_statistic() -> None:
    record = {
        "id": "one",
        "observed_improvement": 0.25,
        "primary_numerator": torch.tensor([1.0, 3.0, 2.0]),
        "primary_denominator": torch.tensor([4.0, 5.0, 6.0]),
        "control_numerator": torch.tensor([2.0, 4.0, 4.0]),
        "control_denominator": torch.tensor([4.0, 5.0, 6.0]),
    }
    observed = analysis._simultaneous_bootstrap(
        [record], documents=3, repetitions=20, seed=7, confidence=0.8,
    )
    generator = torch.Generator().manual_seed(7)
    indices = torch.randint(3, (20, 3), generator=generator)
    primary = record["primary_numerator"][indices].sum(1) / record[
        "primary_denominator"
    ][indices].sum(1)
    control = record["control_numerator"][indices].sum(1) / record[
        "control_denominator"
    ][indices].sum(1)
    errors = 0.25 - (control - primary) / control
    expected = float(torch.kthvalue(errors, 16).values)
    assert observed["critical_order_statistic_one_indexed"] == 16
    assert observed["critical_max_error"] == expected
    assert observed["simultaneous_lcb"]["one"] == 0.25 - expected
