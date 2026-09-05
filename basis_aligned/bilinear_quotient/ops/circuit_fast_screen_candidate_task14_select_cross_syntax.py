#!/usr/bin/env python3
# BQLANE: cpu
"""Held-out SELECT cross-syntax rows for the preselected Task14 carrier sites.

The FIT screen selected attention 11 and head 11.3.  This module pairs the
already-frozen SELECT A1/A2 examples across syntax without reading any SELECT
model outcomes.  SELECT has a disjoint noun pool and disjoint prompt templates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import circuit_battery_task14 as task14


ROOT = Path(__file__).resolve().parent.parent
TASK_ID = "subject_verb.number_agreement"
PHASE = "SELECT"
PARTITION = "HELD_OUT"
VALIDATION_SCOPE = "unseen_nouns_and_prompt_templates_after_fit_site_selection"
SCHEMA = "task14_select_cross_syntax_authority_v1"
SITE_IDS = ("attn:11", "attn:11:head:03")
BATCH_SIZE = 32
MIN_NATIVE_CELL_ACCURACY = 0.85
MIN_CELL_DIRECTION_FRACTION = 0.75
MIN_CELL_MEAN_RECOVERY = 0.40
MIN_DONOR_DENOMINATOR = 1.0e-6

EXPECTED_SOURCE_SHA256 = {
    "task14_generator_file": "33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94",
    "task14_full_authority": "1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1",
    "task14_select_records": "d6d8a7e7cae24ac3e25e3bef11bde4b4b235e950a23c2842978e7fd2a91803b6",
    "fit_full_state_result": "3c87e3973e1a7627f504ce26dfdaa3d7c48536f27a522e36c9e85741f09555c1",
    "fit_cross_syntax_result": "ab1969b363f9a5578c29419bd53050186c0d06c08f01a53858e4e7f7e77d21d7",
}


class SelectCrossSyntaxAuthorityError(ValueError):
    pass


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_rows() -> list[dict]:
    path = Path(task14.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != \
            EXPECTED_SOURCE_SHA256["task14_generator_file"]:
        raise SelectCrossSyntaxAuthorityError("Task14 generator file changed")
    authority, authority_sha = task14.build_authority()
    if authority_sha != EXPECTED_SOURCE_SHA256["task14_full_authority"]:
        raise SelectCrossSyntaxAuthorityError("Task14 full authority changed")
    rows, rows_sha = task14.split_rows(authority, PHASE)
    if rows_sha != EXPECTED_SOURCE_SHA256["task14_select_records"]:
        raise SelectCrossSyntaxAuthorityError("Task14 SELECT records changed")
    full_path = ROOT / "circuits/fast_screens/task14_subject_verb_agreement_full_state_v2_result.json"
    cross_path = ROOT / "circuits/fast_screens/task14_subject_verb_agreement_cross_syntax_v1_result.json"
    if hashlib.sha256(full_path.read_bytes()).hexdigest() != \
            EXPECTED_SOURCE_SHA256["fit_full_state_result"]:
        raise SelectCrossSyntaxAuthorityError("FIT site-selection result changed")
    if hashlib.sha256(cross_path.read_bytes()).hexdigest() != \
            EXPECTED_SOURCE_SHA256["fit_cross_syntax_result"]:
        raise SelectCrossSyntaxAuthorityError("FIT cross-syntax result changed")
    full = json.loads(full_path.read_text())
    head = [item for item in full.get("run", {}).get("site_results", [])
            if item.get("site", {}).get("site_id") == "attn:11:head:03"]
    cross = json.loads(cross_path.read_text())
    cross_head = [item for item in cross.get("site_results", [])
                  if item.get("site_id") == "attn:11:head:03"]
    if len(head) != 1 or head[0].get("terminal") != "screen" \
            or len(cross_head) != 1 or cross_head[0].get("passed") is not True \
            or cross.get("validation_scope") != "new_cross_syntax_relations_not_unseen_text":
        raise SelectCrossSyntaxAuthorityError("FIT evidence no longer preselects head 11.3")
    return rows


def _endpoint(row: Mapping[str, object], side: str) -> dict:
    answer_id = int(row[f"{side}_answer_id"])
    if answer_id not in {318, 389}:
        raise SelectCrossSyntaxAuthorityError("answer left the frozen is/are vocabulary")
    return {
        "ids": list(row[f"{side}_ids"]),
        "answer_id": answer_id,
        "foil_id": 389 if answer_id == 318 else 318,
        "position": int(row[f"{side}_prediction_position"]),
        "text": str(row[f"{side}_text"]),
        "subject_number": str(row[f"{side}_subject_number"]),
        "attractor_plural": bool(row[f"{side}_attractor_plural"]),
        "family": str(row["transform_id"]),
        "group_id": str(row["group_id"]),
        "group_number": int(row["group_number"]),
        "head_pair": list(row["head_pair"]),
        "source_row_id": str(row["row_id"]),
        "side": side,
        "endpoint_id": f"{row['row_id']}:{side}",
    }


def _build_rows_unvalidated() -> list[dict]:
    by_group: dict[str, dict[str, dict]] = {}
    for row in _source_rows():
        by_group.setdefault(str(row["group_id"]), {})[str(row["transform_id"])] = row
    output = []
    for group_id, panel in sorted(
        by_group.items(), key=lambda item: int(item[1]["A1"]["group_number"]),
    ):
        if set(panel) != {"A1", "A2", "P", "C"}:
            raise SelectCrossSyntaxAuthorityError("SELECT panel is incomplete")
        for target_family, donor_family in (("A1", "A2"), ("A2", "A1")):
            target = _endpoint(panel[target_family], "base")
            donor = _endpoint(panel[donor_family], "donor")
            target_syntax = "pp" if target_family == "A1" else "relative"
            donor_syntax = "pp" if donor_family == "A1" else "relative"
            cell_id = (
                f"{target_syntax}_{target['subject_number']}_to_"
                f"{donor_syntax}_{donor['subject_number']}"
            )
            identity = [SCHEMA, group_id, target_family, donor_family]
            output.append({
                "schema": SCHEMA, "task_id": TASK_ID, "split": PHASE,
                "partition": PARTITION, "validation_scope": VALIDATION_SCOPE,
                "row_id": canonical_sha256(identity), "group_id": group_id,
                "cell_id": cell_id,
                "target_family": target_family, "donor_family": donor_family,
                "target_endpoint_id": target["endpoint_id"],
                "donor_endpoint_id": donor["endpoint_id"],
                "base_ids": target["ids"], "donor_ids": donor["ids"],
                "base_text": target["text"], "donor_text": donor["text"],
                "base_answer_id": target["answer_id"], "base_foil_id": target["foil_id"],
                "donor_answer_id": donor["answer_id"], "donor_foil_id": donor["foil_id"],
                "base_semantic_position": target["position"],
                "donor_semantic_position": donor["position"],
                "base_subject_number": target["subject_number"],
                "donor_subject_number": donor["subject_number"],
                "expected_effect": "toward_opposite_number_cross_syntax_donor",
            })
    return output


def build_rows() -> list[dict]:
    output = _build_rows_unvalidated()
    validate_rows(output)
    return output


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if len(materialized) != 64 or len({row.get("row_id") for row in materialized}) != 64:
        raise SelectCrossSyntaxAuthorityError("SELECT authority must contain 64 unique rows")
    cells: dict[str, int] = {}
    for row in materialized:
        if row.get("schema") != SCHEMA or row.get("task_id") != TASK_ID \
                or row.get("split") != PHASE or row.get("partition") != PARTITION \
                or row.get("validation_scope") != VALIDATION_SCOPE:
            raise SelectCrossSyntaxAuthorityError("SELECT row identity changed")
        if {row.get("target_family"), row.get("donor_family")} != {"A1", "A2"}:
            raise SelectCrossSyntaxAuthorityError("row is not cross-syntax")
        if row.get("base_subject_number") == row.get("donor_subject_number"):
            raise SelectCrossSyntaxAuthorityError("row does not reverse subject number")
        if row.get("base_answer_id") != row.get("donor_foil_id") \
                or row.get("base_foil_id") != row.get("donor_answer_id"):
            raise SelectCrossSyntaxAuthorityError("answer orientation is not reversed")
        if row.get("base_semantic_position") != len(row.get("base_ids", ())) - 1 \
                or row.get("donor_semantic_position") != len(row.get("donor_ids", ())) - 1:
            raise SelectCrossSyntaxAuthorityError("semantic position is not final")
        cells[str(row["cell_id"])] = cells.get(str(row["cell_id"]), 0) + 1
    expected = {
        "pp_singular_to_relative_plural", "pp_plural_to_relative_singular",
        "relative_singular_to_pp_plural", "relative_plural_to_pp_singular",
    }
    if set(cells) != expected or set(cells.values()) != {16}:
        raise SelectCrossSyntaxAuthorityError(f"direction-cell balance changed: {cells}")
    if canonical_sha256(materialized) != canonical_sha256(_build_rows_unvalidated()):
        raise SelectCrossSyntaxAuthorityError("rows differ from exact regenerated SELECT authority")
    return canonical_sha256(materialized)


def authority_sha256() -> str:
    return validate_rows(build_rows())


def compile_plan(rows: Sequence[Mapping[str, object]] | None = None) -> dict:
    materialized = build_rows() if rows is None else [dict(row) for row in rows]
    digest = validate_rows(materialized)
    calls = []
    for side in ("base", "donor"):
        for start in range(0, len(materialized), BATCH_SIZE):
            calls.append({"kind": "native", "side": side,
                          "row_ids": [row["row_id"] for row in materialized[start:start+BATCH_SIZE]]})
    for site_id in SITE_IDS:
        for start in range(0, len(materialized), BATCH_SIZE):
            calls.append({"kind": "exact_single_position_interchange", "site_id": site_id,
                          "row_ids": [row["row_id"] for row in materialized[start:start+BATCH_SIZE]]})
    plan = {
        "schema": "task14_targeted_cross_syntax_plan_v1", "phase": PHASE,
        "partition": PARTITION, "validation_scope": VALIDATION_SCOPE,
        "authority_sha256": digest, "source_sha256": EXPECTED_SOURCE_SHA256,
        "site_ids": list(SITE_IDS), "row_count": 64, "batch_size": BATCH_SIZE,
        "calls": calls,
        "price": {"forward_calls": 8, "example_evaluations": 256,
                  "backward_calls": 0, "model_updates": 0,
                  "raw_numeric_evidence_bytes": 2048},
        "score": {"minimum_native_cell_accuracy": MIN_NATIVE_CELL_ACCURACY,
                  "minimum_cell_direction_fraction": MIN_CELL_DIRECTION_FRACTION,
                  "minimum_cell_mean_recovery": MIN_CELL_MEAN_RECOVERY},
        "correction": "FIT selected the two sites; SELECT changes both nouns and prompt templates.",
    }
    plan["compiled_sha256"] = canonical_sha256(plan)
    return plan


if __name__ == "__main__":
    print(json.dumps(compile_plan(), sort_keys=True))
