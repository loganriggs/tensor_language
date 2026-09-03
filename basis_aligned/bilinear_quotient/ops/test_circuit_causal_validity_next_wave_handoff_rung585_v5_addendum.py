"""CPU-only checks for the prospective phase-specific panel contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


OPS = Path(__file__).resolve().parent
V4 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v4_addendum.json"
V5 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_distinct(rows, *, split, level, count):
    selected = sorted(
        (row for row in rows if row["split"] == split and row["level"] == level),
        key=lambda row: row["id"],
    )
    if len(selected) < count:
        raise ValueError("requested panel cell lacks support in opened split")
    result = selected[:count]
    if len({row["id"] for row in result}) != count:
        raise ValueError("panel IDs are not distinct")
    return result


def test_v5_binds_v4_and_enumerates_exact_rules():
    payload = json.loads(V5.read_text(encoding="utf-8"))
    assert payload["schema"] == "circuit_causal_validity_next_wave_handoff_v5_addendum"
    assert payload["v4_contract_sha256"] == sha256(V4)
    assert [item["lesson"] for item in payload["accepted_lessons"]] == [23]
    assert set(payload["forbidden_fallbacks"]) == {
        "borrow_rows_from_unopened_split",
        "sample_with_replacement_without_registration",
        "silently_reduce_requested_cell_size",
        "validate_only_global_not_phase_specific_support",
    }
    assert len(payload["required_test_ids"]) == 4
    assert len(payload["planted_negative_fixture_ids"]) == 4


def test_global_support_cannot_replace_missing_fit_support():
    rows = [
        {"id": "fit-a", "split": "FIT", "level": "short"},
        {"id": "select-b", "split": "SELECT", "level": "long"},
        {"id": "select-c", "split": "SELECT", "level": "long"},
    ]
    assert {row["level"] for row in rows} == {"short", "long"}
    try:
        select_distinct(rows, split="FIT", level="long", count=1)
    except ValueError as error:
        assert "lacks support in opened split" in str(error)
    else:
        raise AssertionError("globally present SELECT-only level satisfied FIT panel")


def test_duplicate_ids_do_not_satisfy_requested_cell_size():
    rows = [
        {"id": "same", "split": "FIT", "level": "x"},
        {"id": "same", "split": "FIT", "level": "x"},
    ]
    try:
        select_distinct(rows, split="FIT", level="x", count=2)
    except ValueError as error:
        assert "not distinct" in str(error)
    else:
        raise AssertionError("duplicate row IDs satisfied distinct panel request")
