from __future__ import annotations

import json

import audit_task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2 as audit


def test_plan_changes_only_numerical_instrument_bar() -> None:
    plan = audit.compile_plan()
    assert plan["repair"] == "maximum_install_absolute_error 1e-5 -> 5e-5 only"
    assert plan["scientific_threshold_changes"] == 0
    assert plan["gpu_rerun"] is False


def test_immutable_v1_passes_repaired_audit() -> None:
    scored = audit.audit(json.loads(audit.V1.read_text()))
    assert all(scored["predictions"].values())
    assert scored["observed_maximum_install_absolute_error"] <= 5e-5
    assert scored["outcomes_reopened"] is False
    assert scored["gpu_rerun"] is False
