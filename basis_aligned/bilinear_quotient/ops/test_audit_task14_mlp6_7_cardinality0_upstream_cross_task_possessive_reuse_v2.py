from __future__ import annotations

import json

import audit_task14_mlp6_7_cardinality0_upstream_cross_task_possessive_reuse_v2 as audit


def test_plan_changes_only_native_capability_gate() -> None:
    plan = audit.compile_plan()
    assert plan["minimum_cell_native_accuracy"] == 0.75
    assert plan["reuse_effect_threshold_changes"] == 0
    assert plan["gpu_rerun"] is False


def test_immutable_v1_becomes_valid_reuse_null() -> None:
    scored = audit.audit(json.loads(audit.V1.read_text()))
    assert scored["predictions"]["pred_b_repaired_native_capability_and_instrument"] is True
    assert scored["predictions"]["pred_c_correct_write_moves_possessive_margin_unchanged"] is False
    assert scored["predictions"]["pred_d_each_direction_construction_unchanged"] is False
    assert scored["predictions"]["pred_e_direction_assignment_unchanged"] is False
    assert all(item["native_base_answer_accuracy"] >= 0.9375 for item in scored["native_capability_by_direction_construction"].values())
