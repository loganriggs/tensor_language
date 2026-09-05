from __future__ import annotations

import pytest

import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as validation


def test_price_is_derived_from_four_method_lattice() -> None:
    assert validation.derive_price() == {
        "physical_model_forwards": 9, "example_evaluations": 2144,
        "causal_installations": 2048, "backwards": 0, "parameter_updates": 0,
        "maximum_patch_chunk_rows": 256, "patch_chunks": 8,
    }


def test_stats_and_pass_thresholds() -> None:
    items = [
        {"actual": value, "predicted": value * 0.9}
        for value in (-2.0, -1.0, 1.0, 2.0)
    ]
    stats = validation._stats(items, "actual", "predicted")
    assert stats["cosine"] == pytest.approx(1.0)
    assert stats["relative_l2_error"] == pytest.approx(0.1)
    assert stats["sign_agreement"] == 1.0
    assert validation._passes(stats, 0.9, 0.45, 0.85)


def test_plan_binds_prediction_before_causal_execution() -> None:
    plan = validation.compile_plan()
    assert plan["sealed_prediction_sha256"] == validation.PREDICTION_SHA256
    assert plan["prototype_construction_reads_target_exact_displacement"] is False
    assert plan["prototype_construction_reads_target_causal_outcome"] is False
