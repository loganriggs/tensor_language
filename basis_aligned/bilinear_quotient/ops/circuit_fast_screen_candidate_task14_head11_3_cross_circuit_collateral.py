#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen SELECT collateral rows for Task14 head 11.3 literal removal.

This module only compiles already-frozen, native-capable rows from two
unrelated behaviors.  It never loads a model and has no queue or result path.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "task14_head11_3_cross_circuit_collateral_authority_v1"
PLAN_SCHEMA = "task14_head11_3_cross_circuit_collateral_plan_v1"
TASK_ID = "subject_verb.number_agreement"
PHASE = "SELECT"
PARTITION = "HELD_OUT_CROSS_CIRCUIT"
SITE_ID = "attn:11:head:03"
BATCH_SIZE = 32
CONDITIONS = ("native_capture", "zero_head11_3", "native_head_replay")
BEHAVIORS = ("numbered_list", "bracket_pending_opener")
MIN_NATIVE_MARGIN = 0.0
MIN_NATIVE_HEAD_NORM = 1.0e-8
MAX_REPLAY_LOGIT_ERROR = 1.0e-4
MAX_MEDIAN_EFFECT = 0.10
MAX_ROW_EFFECT = 0.25
MIN_ROWS_UNDER_EFFECT = 14
MAX_ANSWER_FLIPS = 1

NUMBERED_FAMILY = "list_two_line_state_shift"
NUMBERED_ROW_IDS = (
    "66cda9f85c7f422df63bc516f35caba62a8cbfcda784a55b42d1e2a5d22339cc",
    "ddb6732886ed0298f5c8e1f9b5f1260fc6a48b3cbcacef82d5f3def4304fcd72",
    "a4e2fa1d1a32a8f3405cbe060d5923a8812f5b9c3319bc31d96cdc2f341821eb",
    "cbae5aedc57648c1bc02333ba6970260956f83a63ccd551fee3987b1981a5ead",
    "65070a8c8facb2799fff723902a1ddfd174ce1b4979f821c87d3aaf85d76ca66",
    "d17fd0395382870ec647bdd0b47f343788e59bca883523a62d0e78ecd2354d26",
    "34f4bb34cfca5a280150825f228a204a70c8bf4a5268c74966e94313dceb2c58",
    "9fa97c8130c2482ead1e6fdb170c250278c094572a597f6045f1cac72fc20f26",
    "c31bf11f5f0b3b49c6b2eac92aa68caafb56ace9864297937c1f38a8079ad49d",
    "21c564dc74750f0a074f138acd6dd9f446a5f452b62264635b961caf3991f11f",
    "04e50d9c0811663bc87bb580c62e6cfcca4da62392b5a6ae91fda2e4faa7714e",
    "3f2d0e089caac4477e5016c240a4b94078310596e7e00fb3f2d8ac9be95db93e",
    "ccd3e36f11e19c6d1c2348004b0a3d77e7a977c43b2f3d551cd9059029476799",
    "a044dae6b8dfda2be4d9c8db09241a696d880a72f1d8ad2750df0c2c32aeb592",
    "6275008bd8820f819a8c44b9ad3640a0c40653d44db1ac040500dab2fd2db548",
    "bc3d090851b8307ca69db9d85dfd417a1e5be5c3cdbcc70e102aa73b4b4dc3fd",
)
BRACKET_FAMILY = "opener_type_substitution"
BRACKET_GROUP_IDS = tuple(f"select-{index:03d}" for index in range(48, 56))

SOURCE_PATHS = {
    "increment_builder": ROOT / "ops/increment_two_hypothesis_rows_rung567.py",
    "increment_rows": ROOT / "increment_two_hypothesis_rows_rung567.json",
    "increment_rows_receipt": ROOT / "increment_two_hypothesis_rows_rung567_receipt.json",
    "numeric_native_receipt": ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json",
    "bracket_builder": ROOT / "ops/pending_opener_multifamily_rows_rung537.py",
    "bracket_rows": ROOT / "pending_opener_multifamily_rows_rung537.json",
    "bracket_rows_receipt": ROOT / "pending_opener_multifamily_rows_rung537_receipt.json",
    "bracket_native_receipt": ROOT / "pending_opener_capability_rung537_results.json",
}
EXPECTED_SOURCE_SHA256 = {
    "increment_builder": "222538abe629cdfb38722e238f0cf3263a1b928715072f7d78433dc46e576da3",
    "increment_rows": "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    "increment_rows_receipt": "02b2c37cc23434138accd63e920f417cda10f1c86a4c08c174537149ec2b1072",
    "numeric_native_receipt": "7cc56f22def334673e0035fad7c6a7d1fc58ab8edd3a99744bebd9fb4e6af7e7",
    "bracket_builder": "2de03d2741e01f1ef58ccd12f38f82ca060129a4bfc9f9382e6d00a8424649a5",
    "bracket_rows": "c62cdf3929231e06de6883d74f3ab2c86bd524e02474bb2259267d6976e9e7d9",
    "bracket_rows_receipt": "d50528aa355ba89ab43edd43491c672a6aed88bd8a805ffda936afbfa4cc4816",
    "bracket_native_receipt": "9e76d2c7dba8ea1cfeaf640f9d80508bda3c5df1b151af20f407b535c0dbcb0c",
}


class CrossCircuitCollateralAuthorityError(ValueError):
    """Raised when any frozen authority or panel invariant changes."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_sources() -> tuple[dict, dict]:
    for name, path in SOURCE_PATHS.items():
        if _file_sha256(path) != EXPECTED_SOURCE_SHA256[name]:
            raise CrossCircuitCollateralAuthorityError(
                f"frozen source changed: {name}")

    increment = json.loads(SOURCE_PATHS["increment_rows"].read_text())
    bracket = json.loads(SOURCE_PATHS["bracket_rows"].read_text())
    numeric_native = json.loads(SOURCE_PATHS["numeric_native_receipt"].read_text())
    bracket_native = json.loads(SOURCE_PATHS["bracket_native_receipt"].read_text())

    numbered_summary = numeric_native["hypothesis_results"][
        "numbered_list_index_successor"][PHASE]["state_shifts"][NUMBERED_FAMILY]["base"]
    if (numbered_summary["n_groups"], numbered_summary["correct_fraction"],
            numbered_summary["passed"]) != (16, 1.0, True):
        raise CrossCircuitCollateralAuthorityError(
            "numbered-list SELECT native capability changed")
    bracket_summary = bracket_native["summaries"][PHASE][BRACKET_FAMILY]
    if (bracket_summary["n"], bracket_summary["both_endpoints_correct_fraction"],
            bracket_summary["passed"]) != (16, 1.0, True):
        raise CrossCircuitCollateralAuthorityError(
            "bracket SELECT native capability changed")
    if numeric_native["evaluated_splits"] != {
            "numbered_list_index_successor": ["FIT", "SELECT"],
            "numeric_sequence_continuation": ["FIT", "SELECT"],
    } or numeric_native["forbidden_splits_opened"] != []:
        raise CrossCircuitCollateralAuthorityError("numeric split boundary changed")
    if bracket_native["evaluated_splits"] != ["FIT", "SELECT"] \
            or bracket_native["forbidden_splits_opened"] != []:
        raise CrossCircuitCollateralAuthorityError("bracket split boundary changed")
    return increment, bracket


def _panel_row(source: Mapping[str, object], behavior: str, endpoint: str) -> dict:
    ids = list(source[f"{endpoint}_ids"])
    answer_id = int(source[f"{endpoint}_answer_id"])
    other = "donor" if endpoint == "base" else "base"
    foil_id = int(source[f"{other}_answer_id"])
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "phase": PHASE,
        "partition": PARTITION,
        "site_id": SITE_ID,
        "behavior": behavior,
        "family_id": str(source["family_id"]),
        "endpoint": endpoint,
        "row_id": canonical_sha256([SCHEMA, behavior, source["row_id"], endpoint]),
        "source_row_id": str(source["row_id"]),
        "source_group_id": str(source["group_id"]),
        "ids": ids,
        "text": str(source[f"{endpoint}_text"]),
        "answer_id": answer_id,
        "foil_id": foil_id,
        "semantic_position": len(ids) - 1,
    }


def _build_rows_unvalidated() -> list[dict]:
    increment, bracket = _load_sources()
    numbered_by_id = {
        row["row_id"]: row for row in increment["rows"]
        if row.get("split") == PHASE and row.get("family_id") == NUMBERED_FAMILY
    }
    if tuple(numbered_by_id) != NUMBERED_ROW_IDS:
        raise CrossCircuitCollateralAuthorityError(
            "numbered-list SELECT row membership or order changed")

    bracket_by_group = {
        row["group_id"]: row for row in bracket["rows"]
        if row.get("split") == PHASE and row.get("family_id") == BRACKET_FAMILY
    }
    if tuple(bracket_by_group) != tuple(f"select-{index:03d}" for index in range(48, 64)):
        raise CrossCircuitCollateralAuthorityError(
            "bracket SELECT row membership or order changed")

    output = [
        _panel_row(numbered_by_id[row_id], "numbered_list", "base")
        for row_id in NUMBERED_ROW_IDS
    ]
    for group_id in BRACKET_GROUP_IDS:
        source = bracket_by_group[group_id]
        output.append(_panel_row(source, "bracket_pending_opener", "base"))
        output.append(_panel_row(source, "bracket_pending_opener", "donor"))
    return output


def build_rows() -> list[dict]:
    rows = _build_rows_unvalidated()
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    expected = _build_rows_unvalidated()
    if materialized != expected:
        raise CrossCircuitCollateralAuthorityError(
            "rows differ from exact frozen regeneration")
    if len(materialized) != 32 or len({row["row_id"] for row in materialized}) != 32:
        raise CrossCircuitCollateralAuthorityError("panel must contain 32 unique row IDs")
    if len({tuple(row["ids"]) for row in materialized}) != 32:
        raise CrossCircuitCollateralAuthorityError("panel input sequences are not unique")
    if Counter(row["behavior"] for row in materialized) != Counter({
            "numbered_list": 16, "bracket_pending_opener": 16}):
        raise CrossCircuitCollateralAuthorityError("behavior census changed")
    if Counter(row["endpoint"] for row in materialized) != Counter({
            "base": 24, "donor": 8}):
        raise CrossCircuitCollateralAuthorityError("endpoint census changed")
    groups = {(row["behavior"], row["source_group_id"]) for row in materialized}
    if len(groups) != 24:
        raise CrossCircuitCollateralAuthorityError("source group count changed")
    if Counter(row["answer_id"] for row in materialized) != Counter({
            2091: 8, 2682: 8, 8: 8, 1: 8}):
        raise CrossCircuitCollateralAuthorityError("answer orientation changed")
    for row in materialized:
        if row["phase"] != PHASE or row["site_id"] != SITE_ID:
            raise CrossCircuitCollateralAuthorityError("phase or site changed")
        if row["answer_id"] == row["foil_id"]:
            raise CrossCircuitCollateralAuthorityError("answer equals foil")
        if row["semantic_position"] != len(row["ids"]) - 1:
            raise CrossCircuitCollateralAuthorityError("semantic position is not final")
    return canonical_sha256(materialized)


def compile_plan(rows: Sequence[Mapping[str, object]] | None = None) -> dict:
    materialized = build_rows() if rows is None else [dict(row) for row in rows]
    authority_sha = validate_rows(materialized)
    row_ids = [row["row_id"] for row in materialized]
    calls = [{
        "condition": condition,
        "batch_index": 0,
        "row_set": "ordered_row_ids",
        "row_start": 0,
        "row_stop": 32,
    } for condition in CONDITIONS]
    plan = {
        "schema": PLAN_SCHEMA,
        "task_id": TASK_ID,
        "phase": PHASE,
        "partition": PARTITION,
        "authority_sha256": authority_sha,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "site": {
            "site_id": SITE_ID,
            "layer": 11,
            "head": 3,
            "slice_width": 128,
            "position_scope": "final_prediction_position_only",
            "location": "attention_pre_output_projection_head_slice",
        },
        "conditions": list(CONDITIONS),
        "row_count": 32,
        "source_group_count": 24,
        "batch_size": BATCH_SIZE,
        "ordered_row_ids": row_ids,
        "calls": calls,
        "scoring": {
            "margin": "answer_logit_minus_foil_logit",
            "native_gate": "all_32_margins_strictly_positive",
            "minimum_native_head_norm": MIN_NATIVE_HEAD_NORM,
            "maximum_native_head_replay_logit_error": MAX_REPLAY_LOGIT_ERROR,
            "collateral_scale": "within_behavior_median_native_margin",
            "maximum_median_absolute_scaled_change": MAX_MEDIAN_EFFECT,
            "maximum_rows_above_0_25_scaled_change_per_behavior": 16 - MIN_ROWS_UNDER_EFFECT,
            "maximum_answer_flips_per_behavior": MAX_ANSWER_FLIPS,
            "pool_behaviors": False,
        },
        "price": {
            "forward_calls": 3,
            "example_evaluations": 96,
            "backward_calls": 0,
            "model_updates": 0,
            "evidence_values": 192,
            "evidence_dtype": "float32",
            "raw_numeric_evidence_bytes": 768,
            "evidence_formula": "32 rows * 3 conditions * 2 logits * 4 bytes",
        },
        "execution_policy": {
            "compile_mode": "cpu_only",
            "science_execution": "managed_queue_only",
            "enqueue_after_preregistration": True,
            "create_only": True,
        },
    }
    plan["compiled_sha256"] = canonical_sha256(plan)
    return plan


if __name__ == "__main__":
    print(json.dumps(compile_plan(), sort_keys=True, indent=2))
