#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen held-out Task14 rows for literal attention/head zero removal."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import circuit_battery_task14 as task14


ROOT = Path(__file__).resolve().parent.parent
TASK_ID = "subject_verb.number_agreement"
PHASE = "SELECT"
PARTITION = "HELD_OUT"
SCHEMA = "task14_select_head11_3_zero_removal_authority_v1"
HEAD_SITE_ID = "attn:11:head:03"
ATTENTION_SITE_ID = "attn:11"
BATCH_SIZE = 32
MIN_NATIVE_ACCURACY = {"A1": 0.85, "A2": 0.85, "P": 0.85, "C": 0.75}
MIN_TARGET_DAMAGE = 0.25
MIN_TARGET_DIRECTION = 0.65
MAX_P_ABSOLUTE_DAMAGE = 0.20
MAX_C_ABSOLUTE_DAMAGE = 0.35
MAX_REPLAY_LOGIT_ERROR = 1.0e-4
MIN_TARGET_SCALE = 1.0e-6

EXPECTED_SOURCE_SHA256 = {
    "task14_generator_file": "33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94",
    "task14_full_authority": "1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1",
    "task14_select_records": "d6d8a7e7cae24ac3e25e3bef11bde4b4b235e950a23c2842978e7fd2a91803b6",
    "matched_select_result": "dbe254e8a62d2180d6d1901e6b917d56d9dd8cd8bf37cec27c4579c9d9427eb2",
    "cross_noun_select_result": "c77c68f8122c844ef8241daf48b4e1c2d30fd06102d7672550dd350bc47d1292",
}


class SelectRemovalAuthorityError(ValueError):
    pass


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _source_rows() -> list[dict]:
    if hashlib.sha256(Path(task14.__file__).read_bytes()).hexdigest() != \
            EXPECTED_SOURCE_SHA256["task14_generator_file"]:
        raise SelectRemovalAuthorityError("Task14 generator file changed")
    authority, authority_sha = task14.build_authority()
    if authority_sha != EXPECTED_SOURCE_SHA256["task14_full_authority"]:
        raise SelectRemovalAuthorityError("Task14 full authority changed")
    rows, rows_sha = task14.split_rows(authority, PHASE)
    if rows_sha != EXPECTED_SOURCE_SHA256["task14_select_records"]:
        raise SelectRemovalAuthorityError("Task14 SELECT records changed")
    result_paths = {
        "matched_select_result": ROOT / (
            "circuits/fast_screens/task14_subject_verb_agreement_select_cross_syntax_v1_result.json"
        ),
        "cross_noun_select_result": ROOT / (
            "circuits/fast_screens/"
            "task14_subject_verb_agreement_select_cross_noun_managed_replication_v3_result.json"
        ),
    }
    for name, path in result_paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256[name]:
            raise SelectRemovalAuthorityError(f"preselected SELECT result changed: {name}")
    return rows


def _build_rows_unvalidated() -> list[dict]:
    output = []
    for source in _source_rows():
        answer = int(source["base_answer_id"])
        if answer not in {318, 389}:
            raise SelectRemovalAuthorityError("answer left the frozen is/are vocabulary")
        family = str(source["transform_id"])
        cell_id = "/".join((
            family, str(source["base_template_id"]),
            str(source["base_subject_number"]),
            "attractor_plural" if source["base_attractor_plural"] else "attractor_singular",
        ))
        output.append({
            "schema": SCHEMA, "task_id": TASK_ID, "phase": PHASE,
            "partition": PARTITION,
            "row_id": canonical_sha256([SCHEMA, source["row_id"], "base"]),
            "source_row_id": str(source["row_id"]),
            "group_id": str(source["group_id"]),
            "family": family, "cell_id": cell_id,
            "ids": list(source["base_ids"]), "text": str(source["base_text"]),
            "answer_id": answer, "foil_id": 389 if answer == 318 else 318,
            "semantic_position": int(source["base_prediction_position"]),
            "subject_number": str(source["base_subject_number"]),
            "attractor_plural": bool(source["base_attractor_plural"]),
        })
    return output


def build_rows() -> list[dict]:
    rows = _build_rows_unvalidated()
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if len(materialized) != 128 or len({row.get("row_id") for row in materialized}) != 128:
        raise SelectRemovalAuthorityError("removal authority must contain 128 unique rows")
    families = Counter(str(row.get("family")) for row in materialized)
    if families != Counter({"A1": 32, "A2": 32, "P": 32, "C": 32}):
        raise SelectRemovalAuthorityError(f"family census changed: {families}")
    cell_counts = Counter(str(row.get("cell_id")) for row in materialized)
    if len(cell_counts) != 14 or set(cell_counts.values()) != {8, 16}:
        raise SelectRemovalAuthorityError(f"semantic-cell balance changed: {cell_counts}")
    for row in materialized:
        if row.get("schema") != SCHEMA or row.get("task_id") != TASK_ID \
                or row.get("phase") != PHASE or row.get("partition") != PARTITION:
            raise SelectRemovalAuthorityError("row identity changed")
        if row.get("family") not in MIN_NATIVE_ACCURACY:
            raise SelectRemovalAuthorityError("unknown family")
        if row.get("answer_id") == row.get("foil_id") \
                or row.get("semantic_position") != len(row.get("ids", ())) - 1:
            raise SelectRemovalAuthorityError("endpoint or semantic position changed")
    if canonical_sha256(materialized) != canonical_sha256(_build_rows_unvalidated()):
        raise SelectRemovalAuthorityError("rows differ from exact regenerated SELECT authority")
    return canonical_sha256(materialized)


def compile_plan(rows: Sequence[Mapping[str, object]] | None = None) -> dict:
    materialized = build_rows() if rows is None else [dict(row) for row in rows]
    authority_sha = validate_rows(materialized)
    chunks = [materialized[start:start+BATCH_SIZE]
              for start in range(0, len(materialized), BATCH_SIZE)]
    calls = []
    for condition in ("native_capture", "zero_head11_3", "zero_attention11", "native_head_replay"):
        calls.extend({"condition": condition, "row_ids": [row["row_id"] for row in chunk]}
                     for chunk in chunks)
    plan = {
        "schema": "task14_select_head11_3_zero_removal_plan_v1",
        "phase": PHASE, "partition": PARTITION,
        "authority_sha256": authority_sha,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "conditions": ["native_capture", "zero_head11_3", "zero_attention11", "native_head_replay"],
        "site_ids": [HEAD_SITE_ID, ATTENTION_SITE_ID],
        "row_count": len(materialized), "batch_size": BATCH_SIZE, "calls": calls,
        "bars": {
            "minimum_native_accuracy": MIN_NATIVE_ACCURACY,
            "minimum_target_damage": MIN_TARGET_DAMAGE,
            "minimum_target_direction_fraction": MIN_TARGET_DIRECTION,
            "maximum_P_absolute_damage": MAX_P_ABSOLUTE_DAMAGE,
            "maximum_C_absolute_damage": MAX_C_ABSOLUTE_DAMAGE,
            "maximum_replay_logit_error": MAX_REPLAY_LOGIT_ERROR,
        },
        "price": {"forward_calls": 16, "example_evaluations": 512,
                  "backward_calls": 0, "model_updates": 0,
                  "raw_numeric_evidence_bytes": 4096},
        "null": (
            "valid replay and live whole-attention removal, but head 11.3 either fails "
            "A1/A2 necessity or exceeds a P/C collateral ceiling"
        ),
    }
    plan["compiled_sha256"] = canonical_sha256(plan)
    return plan


if __name__ == "__main__":
    print(json.dumps(compile_plan(), sort_keys=True))
