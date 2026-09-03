#!/usr/bin/env python3
"""Outcome-blind semantic source-position audit for R560."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
ROWS = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
OUT = ROOT / "pending_opener_source_positions_rung560_audit.json"
EXPECTED_ROWS_SHA256 = "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9"
VARIABLE_OPENERS = {"pending_parenthesis": 357, "pending_square": 685, "pending_quote": 366}
CLOSER_OPENERS = {8: 357, 60: 685, 1: 366}
SPLITS = ("FIT", "SELECT")
TARGETS = {"direct_three_value_type_substitution", "completed_then_reopened_three_value_order"}
CONTROLS = {
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_position(ids: list[int], answer_id: int) -> int:
    opener = CLOSER_OPENERS[answer_id]
    positions = [index for index, token in enumerate(ids) if token == opener]
    if not positions:
        raise AssertionError(f"no opener {opener} for closer {answer_id}")
    return positions[-1]


def main() -> None:
    if sha256(ROWS) != EXPECTED_ROWS_SHA256:
        raise RuntimeError("R545 rows changed")
    document = json.loads(ROWS.read_text())
    rows = [row for row in document["rows"] if row["split"] in SPLITS]
    if len(rows) != 540 or len({row["row_id"] for row in rows}) != 540:
        raise RuntimeError("FIT/SELECT row identity changed")
    records = []
    metadata_mismatches = 0
    counts = Counter()
    for row in rows:
        family = row["family_id"]
        assert family in TARGETS | CONTROLS
        base_position = source_position(row["base_ids"], row["base_answer_id"])
        donor_position = source_position(row["donor_ids"], row["donor_answer_id"])
        assert base_position < len(row["base_ids"]) - 1
        assert donor_position < len(row["donor_ids"]) - 1
        assert base_position > 0 and donor_position > 0
        assert row["base_ids"][base_position] == CLOSER_OPENERS[row["base_answer_id"]]
        assert row["donor_ids"][donor_position] == CLOSER_OPENERS[row["donor_answer_id"]]
        assert row["base_ids"][base_position - 1] not in CLOSER_OPENERS.values()
        assert row["donor_ids"][donor_position - 1] not in CLOSER_OPENERS.values()
        metadata_mismatches += int(
            VARIABLE_OPENERS[row["proposed_variable_base"]] != CLOSER_OPENERS[row["base_answer_id"]]
        )
        metadata_mismatches += int(
            VARIABLE_OPENERS[row["proposed_variable_donor"]] != CLOSER_OPENERS[row["donor_answer_id"]]
        )
        if family in TARGETS:
            assert row["answer_changes"] is True
        else:
            assert row["answer_changes"] is False
            assert row["base_answer_id"] == row["donor_answer_id"]
        if family == "pending_type_preserved_distance_extension":
            assert len(row["base_ids"]) != len(row["donor_ids"])
        counts[(row["split"], family)] += 1
        records.append({
            "row_id": row["row_id"],
            "group_id": row["group_id"],
            "split": row["split"],
            "family_id": family,
            "base_source_position": base_position,
            "donor_source_position": donor_position,
            "base_wrong_source_position": base_position - 1,
            "donor_wrong_source_position": donor_position - 1,
        })
    assert all(counts[("FIT", family)] == 72 for family in TARGETS | CONTROLS)
    assert all(counts[("SELECT", family)] == 36 for family in TARGETS | CONTROLS)
    payload = {
        "rung": 560,
        "stage": "pending_opener_semantic_source_position_audit",
        "all_checks_pass": True,
        "row_count": len(records),
        "source_definition": {str(closer): {"opener_token_id": opener, "rule": "final occurrence"}
                              for closer, opener in CLOSER_OPENERS.items()},
        "inconsistent_proposed_variable_endpoint_labels": metadata_mismatches,
        "counts": {f"{split}:{family}": count for (split, family), count in sorted(counts.items())},
        "unequal_length_distance_rows": sum(
            row["family_id"] == "pending_type_preserved_distance_extension" for row in records
        ),
        "records": records,
        "input_sha256": {str(ROWS): EXPECTED_ROWS_SHA256},
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
