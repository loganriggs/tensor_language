#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen SELECT authority for the exact numbered-list T3/T7 removal factorial."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import numbered_list_cached_value_weight_removal_rung576 as r576


ROOT = Path(__file__).resolve().parent.parent
POLY = ROOT.parent / "polynomial_causal"
TASK_ID = "numbered_list.index_successor.cached_term_head_source_factorial"
PHASE = "SELECT"
PARTITION = "HELD_OUT_WITHIN_AUTHORITY"
SCHEMA = "numbered_list_cached_term_head_source_factorial_authority_v1"
FAMILIES = (
    "list_two_line_state_shift", "list_three_line_state_shift",
    "list_repeated_index_control", "sequence_word_copy_control",
)
TARGET_FAMILIES = FAMILIES[:2]
CONTROL_FAMILIES = FAMILIES[2:]
ENDPOINTS = ("base", "donor")
CONDITIONS = ("native_direct", "native_factor_replay", "zero_T3", "zero_T7", "zero_T3_T7")
HEADS_BY_CONDITION = {"zero_T3": (3,), "zero_T7": (7,), "zero_T3_T7": (3, 7)}
BATCH_SIZE = 32
BOOTSTRAPS = 2000
SEED = 57637
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
EXPECTED_SOURCE_SHA256 = {
    "r567_rows": "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    "r575_positions": "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b",
    "r573_result": "052930b8b9086e8b7606e3d05929f521f468c04427be8d1182720f1772ee43ec",
    "r576_result": "a6041c28cefc4f695f6e649210884774ed576bae80c14c031473d6b8c8ff2f73",
    "r576_implementation": "91db3a2a9210aef915ce2e4f0a62253274e0b5470cbfaa05a95d50a3c0cf985a",
    "prior_art": "bd1eebc1d19052ce95042569fb6deca2bf29eb8c79f197f5c7cc60c40c8ad6a0",
    "preregistration": "0abb5d488451cc530c4966f41a781c8dbe931644aecca51f36f36a7c8da85ae7",
}
SOURCE_PATHS = {
    "r567_rows": r576.ROWS, "r575_positions": r576.POSITIONS,
    "r573_result": r576.R573_RESULT,
    "r576_result": ROOT / "numbered_list_cached_value_weight_removal_rung576_results.json",
    "r576_implementation": Path(r576.__file__),
    "prior_art": ROOT / "circuits/numbered_list_cached_term_head_source_factorial_prior_art.json",
    "preregistration": POLY / "NUMBERED_LIST_CACHED_TERM_HEAD_SOURCE_FACTORIAL_PREREGISTRATION.md",
}
FIT_SCALES = {"margin_damage": 2.181361138820648, "logit_rms": 0.659287303686142,
              "term_norm": 3411.7650146484375}
MIN_POSITIVE_FRACTION = .75
MIN_ANSWER_PRESERVED = .75
MAX_CONTROL_CE = .1
MAX_CONTROL_SCALE_FRACTION = .25
MIN_TERM_SCALE_FRACTION = .10
MAX_NATIVE_REPLAY_RSE = 1e-12
MAX_JOINT_TERM_RSE = 1e-10


class CandidateError(ValueError):
    pass


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _verify_sources() -> None:
    for name, path in SOURCE_PATHS.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256[name]:
            raise CandidateError(f"frozen source changed: {name}")


def _build_rows_unvalidated() -> list[dict]:
    _verify_sources()
    rows, positions = r576.load_authority()
    output = []
    for source in rows:
        if source["split"] != PHASE or source["family_id"] not in FAMILIES:
            continue
        for endpoint in ENDPOINTS:
            mapping = positions[source["row_id"]]["endpoints"][endpoint]
            answer_id = int(source[f"{endpoint}_answer_id"])
            output.append({
                "schema": SCHEMA, "task_id": TASK_ID, "phase": PHASE, "partition": PARTITION,
                "row_id": canonical_sha256([SCHEMA, source["row_id"], endpoint]),
                "source_row_id": source["row_id"], "group_id": source["group_id"],
                "family": source["family_id"], "endpoint": endpoint,
                "ids": list(source[f"{endpoint}_ids"]), "text": source[f"{endpoint}_text"],
                "answer": source[f"{endpoint}_answer"], "answer_id": answer_id,
                "query_position": int(mapping["query_position"]),
                "source_position": int(mapping["source_position"]),
            })
    return output


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if len(materialized) != 128 or len({row.get("row_id") for row in materialized}) != 128:
        raise CandidateError("authority must contain 128 unique endpoint rows")
    if Counter(row.get("family") for row in materialized) != Counter({family: 32 for family in FAMILIES}):
        raise CandidateError("family counts changed")
    if Counter((row.get("family"), row.get("endpoint")) for row in materialized) != Counter(
            {(family, endpoint): 16 for family in FAMILIES for endpoint in ENDPOINTS}):
        raise CandidateError("family/endpoint counts changed")
    if len({row.get("group_id") for row in materialized}) != 32:
        raise CandidateError("group count changed")
    for row in materialized:
        if row.get("schema") != SCHEMA or row.get("phase") != PHASE or row.get("partition") != PARTITION:
            raise CandidateError("row identity changed")
        ids = row.get("ids")
        if not isinstance(ids, list) or row.get("query_position") != len(ids) - 1:
            raise CandidateError("final query position changed")
        source = row.get("source_position")
        if type(source) is not int or not 0 <= source < int(row["query_position"]):
            raise CandidateError("cached-value source position changed")
        if type(row.get("answer_id")) is not int or not str(row.get("answer", "")):
            raise CandidateError("answer orientation changed")
    regenerated = _build_rows_unvalidated()
    if canonical_sha256(materialized) != canonical_sha256(regenerated):
        raise CandidateError("rows differ from regenerated frozen membership")
    return canonical_sha256(materialized)


def build_rows() -> list[dict]:
    rows = _build_rows_unvalidated()
    validate_rows(rows)
    return rows


def _chunks(rows: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
    ordered = sorted(rows, key=lambda row: (len(row["ids"]), str(row["row_id"])))
    output = []
    while ordered:
        length = len(ordered[0]["ids"])
        same = [row for row in ordered if len(row["ids"]) == length]
        ordered = [row for row in ordered if len(row["ids"]) != length]
        output.extend(same[start:start+BATCH_SIZE] for start in range(0, len(same), BATCH_SIZE))
    return output


def compile_plan(rows: Sequence[Mapping[str, object]] | None = None) -> dict:
    materialized = build_rows() if rows is None else [dict(row) for row in rows]
    authority_sha = validate_rows(materialized)
    chunks = _chunks(materialized)
    calls = [{"condition": condition, "row_ids": [row["row_id"] for row in chunk]}
             for condition in CONDITIONS for chunk in chunks]
    evaluations = sum(len(call["row_ids"]) for call in calls)
    plan = {
        "schema": "numbered_list_cached_term_head_source_factorial_plan_v1",
        "task_id": TASK_ID, "phase": PHASE, "partition": PARTITION,
        "authority_sha256": authority_sha, "source_sha256": EXPECTED_SOURCE_SHA256,
        "conditions": list(CONDITIONS), "heads_by_condition": HEADS_BY_CONDITION,
        "row_count": len(materialized), "group_count": 32, "batch_size": BATCH_SIZE,
        "calls": calls, "fit_scales": FIT_SCALES,
        "bars": {"minimum_positive_fraction": MIN_POSITIVE_FRACTION,
                 "minimum_answer_preserved": MIN_ANSWER_PRESERVED,
                 "maximum_control_ce": MAX_CONTROL_CE,
                 "maximum_control_scale_fraction": MAX_CONTROL_SCALE_FRACTION,
                 "minimum_term_scale_fraction": MIN_TERM_SCALE_FRACTION,
                 "maximum_native_replay_rse": MAX_NATIVE_REPLAY_RSE,
                 "maximum_joint_term_rse": MAX_JOINT_TERM_RSE},
        "opened_splits": [PHASE], "closed_splits": ["FINAL_TEST", "OOD"],
        "execution_policy": {"compile_mode": "cpu_only", "science_execution": "managed_queue_only",
                             "enqueue_after_preregistration": True, "create_only": True},
        "price": {"forward_calls": len(calls), "example_evaluations": evaluations,
                  "backward_calls": 0, "model_updates": 0,
                  "raw_numeric_evidence_bytes": len(materialized) * 13 * 8},
    }
    plan["compiled_sha256"] = canonical_sha256(plan)
    return plan


if __name__ == "__main__":
    print(json.dumps(compile_plan(), sort_keys=True))
