#!/usr/bin/env python3
"""Focused CPU tests for the Task 14 reusable fast-screen adapter."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

import circuit_fast_screen_candidate_task14_agreement as adapter
import circuit_fast_screen_spec as screen
import circuit_prior_art


ROOT = Path(__file__).resolve().parent.parent


def test_immutable_sources_and_authority_shape() -> None:
    sources = adapter.load_sources()
    assert set(sources) == set(adapter.EXPECTED_SOURCE_SHA256)
    rows = adapter.build_rows(adapter.TASK_ID)
    assert len(rows) == 128
    assert len({row["group_id"] for row in rows}) == 32
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    assert all(
        [row["transform_id"] for row in group] == ["A1", "A2", "P", "C"]
        for group in by_group.values()
    )
    assert all(
        row["base_semantic_position"] == row["base_prediction_position"]
        and row["donor_semantic_position"] == row["donor_prediction_position"]
        for row in rows
    )
    assert adapter.validate_rows(rows) == adapter.authority_sha256()
    assert adapter.authority_sha256() == \
        "9b8ede7d17b0358467438b7f8fda7703bba1c93c9c594d55454404c1bb6e21cc"


def test_adapter_interface_rejects_unknown_task_and_mutated_frozen_row() -> None:
    with pytest.raises(KeyError):
        adapter.build_rows("unknown.task")
    rows = adapter.build_rows(adapter.TASK_ID)
    rows[0] = dict(rows[0], base_text="The altered prompt")
    with pytest.raises(adapter.Task14AdapterError, match="deterministic"):
        adapter.validate_rows(rows)


def test_a1_a2_targets_and_both_same_answer_controls_keep_frozen_meanings() -> None:
    rows = adapter.adapted_rows()
    for row in rows:
        family = row["transform_id"]
        if family in {"A1", "A2"}:
            assert row["answer_changes"] is True
            assert row["base_answer_id"] != row["donor_answer_id"]
            assert row["expected_effect"] == "toward_donor"
        elif family == "P":
            assert row["answer_changes"] is False
            assert row["base_answer_id"] == row["donor_answer_id"]
            assert row["expected_effect"] == "invariant"
        else:
            assert family == "C"
            assert row["answer_changes"] is False
            assert row["base_answer_id"] == row["donor_answer_id"]
            assert row["base_subject_number"] == row["donor_subject_number"] == "plural"
            assert row["expected_effect"] == "registered_active"


def test_capability_reproduces_249_of_256_and_every_ordered_cell_bar() -> None:
    report = adapter.compatibility_report()
    assert report.capability_correct == 249
    assert report.capability_expected == 256
    assert report.capability_errors == 7
    assert len(report.capability_cells) == 8
    assert all(cell.row_count == 16 for cell in report.capability_cells)
    assert report.all_numerical_cell_bars_pass is True
    cells = {cell.cell_id: cell for cell in report.capability_cells}
    assert cells["C/fit_coord_near/attractor_singular_to_plural"].base_accuracy == 0.875
    assert cells["C/fit_coord_near/attractor_singular_to_plural"].donor_accuracy == 0.9375
    assert cells["C/fit_coord_near/attractor_plural_to_singular"].base_accuracy == 0.875
    assert cells["C/fit_coord_near/attractor_plural_to_singular"].donor_accuracy == 0.875
    assert all(
        cell.base_accuracy == cell.donor_accuracy == 1.0
        for cell in report.capability_cells if cell.family != "C"
    )


def test_seven_errors_pass_current_per_cell_bars_and_c_is_compatible() -> None:
    report = adapter.compatibility_report()
    assert report.generic_semantics_compatible is True
    assert report.c_control_semantics == "same_answer_active_negative_control"
    assert report.v2_answer_changing_c_donors_are_target_related is True
    # All seven errors occur in the two C cells.  Their 87.5--93.75% native
    # accuracies still exceed the registered 75% C threshold.
    assert report.all_numerical_cell_bars_pass is True
    assert adapter.build_rows() == adapter.adapted_rows()


def test_full_state_screen_scope_is_exactly_19_plus_36() -> None:
    report = adapter.compatibility_report()
    assert report.residual_sites == 19
    assert report.module_sites == 36
    assert report.total_sites == 55


def test_compiles_exact_55_site_screen_with_same_answer_c_role() -> None:
    rows = adapter.build_rows()
    spec = adapter.build_spec(rows)
    compiled = screen.compile_screen(spec, rows)
    assert compiled["score_contract"]["family_roles"] == {
        "A1": "answer_changing_target",
        "A2": "answer_changing_target",
        "P": "same_answer_invariance_control",
        "C": "same_answer_active_negative_control",
    }
    assert len(compiled["call_manifest"]) == 228
    assert compiled["price"] == {
        "phase": "FIT", "forward_calls": 228, "example_evaluations": 7296,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 58368,
    }
    assert compiled["max_price"] == {
        "phase": "FIT", "forward_calls": 264, "example_evaluations": 8448,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 67584,
    }
    ceiling_calls = [
        call for call in compiled["call_manifest"] if call.get("stage") == "ceiling"
    ]
    assert len(ceiling_calls) == 220
    assert {call["site"]["site_id"] for call in ceiling_calls} == set(
        screen.CEILING_SITE_IDS
    )
    assert all(
        binding["recipient_position"] == len(row["base_ids"]) - 1
        and binding["donor_position"] == len(row["donor_ids"]) - 1
        for call in ceiling_calls
        for binding, row in zip(
            call["semantic_bindings"],
            [next(item for item in rows if item["row_id"] == row_id)
             for row_id in call["row_ids"]],
        )
    )
    screen.validate_compiled_screen(spec, rows, compiled)


def test_cpu_dryrun_keeps_head_stage_unopened() -> None:
    rows = adapter.build_rows()
    receipt = screen.compile_dryrun(adapter.build_spec(rows), rows)
    assert receipt["authority_sha256"] == adapter.authority_sha256()
    assert receipt["active_price"]["forward_calls"] == 228
    assert receipt["max_price"]["forward_calls"] == 264
    assert receipt["head_stage"] == "pending"
    assert receipt["model_loaded"] is False
    assert receipt["gpu_accessed"] is False
    assert receipt["queue_touched"] is False


def test_audit_digest_is_frozen() -> None:
    expected = "e28a33ce6860eaaf1fc9e60570546770f67b9413d6a21dd1b12cac07c3e05b1a"
    assert adapter.audit_sha256() == expected
    assert adapter.canonical_sha256(asdict(adapter.compatibility_report())) == expected


def test_prior_art_receipt_is_current_and_declares_extension() -> None:
    receipt = json.loads(
        (ROOT / "circuits/fast_screen_task14_agreement_prior_art.json").read_text()
    )
    assert receipt["candidate_id"] == adapter.TASK_ID
    assert receipt["relation"] == "extension"
    assert circuit_prior_art.validate_source_files(receipt, ROOT)
