#!/usr/bin/env python3
# BQLANE: cpu

from collections import Counter
import json
from pathlib import Path

import pytest

import circuit_fast_screen_candidate_task14_head11_3_cross_circuit_collateral as candidate


def test_exact_core_panel_membership_balance_and_uniqueness() -> None:
    rows = candidate.build_rows()
    assert len(rows) == 32
    assert len({tuple(row["ids"]) for row in rows}) == 32
    assert Counter(row["behavior"] for row in rows) == {
        "numbered_list": 16, "bracket_pending_opener": 16,
    }
    assert Counter(row["answer_id"] for row in rows) == {
        2091: 8, 2682: 8, 8: 8, 1: 8,
    }
    numbered = [row for row in rows if row["behavior"] == "numbered_list"]
    bracket = [row for row in rows if row["behavior"] == "bracket_pending_opener"]
    assert [row["source_row_id"] for row in numbered] == list(candidate.NUMBERED_ROW_IDS)
    assert {row["endpoint"] for row in numbered} == {"base"}
    assert Counter(row["source_group_id"] for row in bracket) == {
        group_id: 2 for group_id in candidate.BRACKET_GROUP_IDS
    }
    assert all(row["semantic_position"] == len(row["ids"]) - 1 for row in rows)


def test_answers_and_foils_keep_frozen_orientation() -> None:
    rows = candidate.build_rows()
    for row in rows:
        if row["behavior"] == "numbered_list":
            assert {row["answer_id"], row["foil_id"]} == {2091, 2682}
        else:
            assert {row["answer_id"], row["foil_id"]} == {1, 8}
    bracket = [row for row in rows if row["behavior"] == "bracket_pending_opener"]
    assert Counter((row["endpoint"], row["answer_id"], row["foil_id"])
                   for row in bracket) == {
        ("base", 8, 1): 8,
        ("donor", 1, 8): 8,
    }


def test_plan_has_three_one_batch_conditions_and_exact_price() -> None:
    plan = candidate.compile_plan()
    assert plan["conditions"] == [
        "native_capture", "zero_head11_3", "native_head_replay",
    ]
    assert [call["condition"] for call in plan["calls"]] == plan["conditions"]
    assert len(plan["ordered_row_ids"]) == 32
    assert all(call["row_set"] == "ordered_row_ids" and
               (call["row_start"], call["row_stop"]) == (0, 32)
               for call in plan["calls"])
    assert plan["price"] == {
        "forward_calls": 3,
        "example_evaluations": 96,
        "backward_calls": 0,
        "model_updates": 0,
        "evidence_values": 192,
        "evidence_dtype": "float32",
        "raw_numeric_evidence_bytes": 768,
        "evidence_formula": "32 rows * 3 conditions * 2 logits * 4 bytes",
    }


def test_mutation_is_rejected_against_exact_regeneration() -> None:
    rows = candidate.build_rows()
    rows[0] = dict(rows[0], answer_id=rows[0]["foil_id"])
    with pytest.raises(candidate.CrossCircuitCollateralAuthorityError,
                       match="exact frozen regeneration"):
        candidate.validate_rows(rows)


def test_every_declared_source_hash_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = dict(candidate.EXPECTED_SOURCE_SHA256)
    changed["increment_builder"] = "0" * 64
    monkeypatch.setattr(candidate, "EXPECTED_SOURCE_SHA256", changed)
    with pytest.raises(candidate.CrossCircuitCollateralAuthorityError,
                       match="frozen source changed: increment_builder"):
        candidate.build_rows()


def test_checked_in_dryrun_is_exact_compiled_plan() -> None:
    path = Path(__file__).with_name("task14_head11_3_cross_circuit_collateral_dryrun.json")
    assert json.loads(path.read_text()) == candidate.compile_plan()
