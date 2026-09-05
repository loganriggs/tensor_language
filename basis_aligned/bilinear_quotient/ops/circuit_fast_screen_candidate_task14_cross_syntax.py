#!/usr/bin/env python3
"""Frozen Task 14 literal cross-syntax interchange authority (CPU only).

This adapter selects the 64 ``cross_syntax`` relations in the pre-existing
Task 14 v2 donor manifest's FIT-internal VALIDATION partition.  Each relation
patches a PP prompt from an opposite-number relative-clause prompt, or vice
versa.  It changes no prompt, token, endpoint, donor, or partition assignment.

Important correction: the earlier Task 14 fast screen swapped A1 rows only
with A1 donors and A2 rows only with A2 donors.  A common site passing both
families was evidence across two constructions, but it was not literal
cross-syntax donor interchange.  This authority performs that missing test.

The endpoint texts themselves appeared in the earlier full-state screen.
Therefore ``VALIDATION`` here names the frozen v2 within-FIT partition and does
not mean unseen text relative to the earlier site selection.  The new unit of
generalization is the cross-syntax target/donor relation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
TASK_ID = "subject_verb.number_agreement"
SCHEMA = "task14_cross_syntax_interchange_authority_v1"
PHASE = "FIT"
PARTITION = "VALIDATION"
VALIDATION_SCOPE = "new_cross_syntax_relations_not_unseen_text"
SITE_IDS = ("attn:11", "attn:11:head:03")
BATCH_SIZE = 32
ANSWER_TOKEN_IDS = {" is": 318, " are": 389}

AUTHORITY_PATH = ROOT / "ops/circuit_battery_task14_agreement_fit_authority.json"
PARTITION_PATH = ROOT / "ops/circuit_battery_task14_fit_localization_partition_v2.json"
DONORS_PATH = ROOT / "ops/circuit_battery_task14_fit_localization_donors_v2.json"
V2_RESULT_PATH = (
    ROOT / "circuits/fast_screens/"
    "task14_subject_verb_agreement_full_state_v2_result.json"
)

SOURCE_PATHS = {
    "authority": AUTHORITY_PATH,
    "partition": PARTITION_PATH,
    "donors": DONORS_PATH,
    "v2_result": V2_RESULT_PATH,
}
EXPECTED_SOURCE_SHA256 = {
    "authority": "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f",
    "partition": "1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3",
    "donors": "ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a",
    "v2_result": "3c87e3973e1a7627f504ce26dfdaa3d7c48536f27a522e36c9e85741f09555c1",
}

MIN_NATIVE_CELL_ACCURACY = 0.85
MIN_CELL_DIRECTION_FRACTION = 0.75
MIN_CELL_MEAN_RECOVERY = 0.40
MIN_DONOR_DENOMINATOR = 1.0e-6


class CrossSyntaxAuthorityError(ValueError):
    """A frozen source or derived cross-syntax relation is inconsistent."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise CrossSyntaxAuthorityError(f"source is not a JSON object: {path}")
    return value


def load_sources() -> dict[str, dict[str, Any]]:
    """Load every immutable input after checking its exact file bytes."""
    output: dict[str, dict[str, Any]] = {}
    for name, path in SOURCE_PATHS.items():
        observed = file_sha256(path)
        expected = EXPECTED_SOURCE_SHA256[name]
        if observed != expected:
            raise CrossSyntaxAuthorityError(
                f"immutable {name} changed: expected={expected}, observed={observed}"
            )
        output[name] = _load_json(path)
    return output


def _endpoint(
    endpoint_id: str,
    authority_by_row: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, object], str]:
    try:
        row_id, side = endpoint_id.rsplit(":", 1)
    except ValueError as error:
        raise CrossSyntaxAuthorityError("endpoint ID lacks a side") from error
    if side not in {"base", "donor"} or row_id not in authority_by_row:
        raise CrossSyntaxAuthorityError(f"unknown endpoint: {endpoint_id}")
    return authority_by_row[row_id], side


def _side_value(row: Mapping[str, object], side: str, field: str) -> object:
    key = f"{side}_{field}"
    if key not in row:
        raise CrossSyntaxAuthorityError(f"endpoint lacks {key}")
    return row[key]


def _tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise CrossSyntaxAuthorityError(f"{label} is not a token sequence")
    return list(value)


def _token(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CrossSyntaxAuthorityError(f"{label} is not a token ID")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CrossSyntaxAuthorityError(f"{label} is not text")
    return value


def _validate_selected_sites(v2_result: Mapping[str, object]) -> None:
    if v2_result.get("candidate_id") != TASK_ID or v2_result.get("terminal") != "screen":
        raise CrossSyntaxAuthorityError("v2 source is not the completed Task 14 screen")
    run = v2_result.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("site_results"), list):
        raise CrossSyntaxAuthorityError("v2 source lacks site results")
    by_site = {
        str(item.get("site", {}).get("site_id")): item
        for item in run["site_results"]
        if isinstance(item, dict) and isinstance(item.get("site"), dict)
    }
    for site_id in SITE_IDS:
        item = by_site.get(site_id)
        if not isinstance(item, dict) or item.get("terminal") != "screen":
            raise CrossSyntaxAuthorityError(f"v2 did not select a live site: {site_id}")
    if not math.isclose(float(by_site["attn:11"]["target_recovery"]),
                        0.6126028456484134, rel_tol=0.0, abs_tol=1e-15):
        raise CrossSyntaxAuthorityError("v2 attention-11 score changed")
    if not math.isclose(float(by_site["attn:11:head:03"]["target_recovery"]),
                        0.603607803023223, rel_tol=0.0, abs_tol=1e-15):
        raise CrossSyntaxAuthorityError("v2 head-11.3 score changed")


def build_rows(task_id: str = TASK_ID) -> list[dict[str, Any]]:
    """Derive the exact 64 relation rows without changing source endpoints."""
    if task_id != TASK_ID:
        raise KeyError(task_id)
    sources = load_sources()
    authority = sources["authority"]
    partition = sources["partition"]
    donors = sources["donors"]
    _validate_selected_sites(sources["v2_result"])

    if authority.get("task_id") != TASK_ID or authority.get("split") != "FIT":
        raise CrossSyntaxAuthorityError("Task 14 authority identity changed")
    authority_rows = authority.get("rows")
    partition_records = partition.get("records")
    donor_records = donors.get("records")
    if not isinstance(authority_rows, list) or len(authority_rows) != 128:
        raise CrossSyntaxAuthorityError("expected 128 frozen Task 14 rows")
    if not isinstance(partition_records, list) or len(partition_records) != 32:
        raise CrossSyntaxAuthorityError("expected 32 frozen partition records")
    if not isinstance(donor_records, list) or len(donor_records) != 1088:
        raise CrossSyntaxAuthorityError("expected 1088 frozen donor records")
    by_row = {str(row["row_id"]): row for row in authority_rows}
    if len(by_row) != 128:
        raise CrossSyntaxAuthorityError("Task 14 row IDs are not unique")
    validation_groups = {
        str(record["group_id"])
        for record in partition_records
        if record.get("partition") == "VALIDATION"
    }
    if len(validation_groups) != 16:
        raise CrossSyntaxAuthorityError("expected 16 FIT-internal VALIDATION groups")

    selected = [
        record for record in donor_records
        if record.get("partition") == "VALIDATION"
        and record.get("arm") == "cross_syntax"
    ]
    if len(selected) != 64 or [record.get("ordinal") for record in selected] != list(
        range(832, 896)
    ):
        raise CrossSyntaxAuthorityError("cross-syntax record census/order changed")

    output: list[dict[str, Any]] = []
    for relation in selected:
        if relation.get("family") not in {"A1", "A2"} \
                or relation.get("matching") != "cross_syntax_1" \
                or relation.get("expected_relation") != "opposite_subject_toward_donor" \
                or relation.get("source_contract") != "v1_original_704" \
                or relation.get("q_only") is not False:
            raise CrossSyntaxAuthorityError("cross-syntax relation semantics changed")
        target, target_side = _endpoint(str(relation["target_endpoint_id"]), by_row)
        donor, donor_side = _endpoint(str(relation["donor_endpoint_id"]), by_row)
        target_family = str(target["transform_id"])
        donor_family = str(donor["transform_id"])
        if target_family != relation["family"] or {target_family, donor_family} != {
            "A1", "A2"
        }:
            raise CrossSyntaxAuthorityError("relation no longer maps PP to relative syntax")
        if target["group_id"] not in validation_groups \
                or donor["group_id"] not in validation_groups:
            raise CrossSyntaxAuthorityError("cross-syntax endpoint escaped VALIDATION")

        target_number = _text(
            _side_value(target, target_side, "subject_number"), "target number"
        )
        donor_number = _text(
            _side_value(donor, donor_side, "subject_number"), "donor number"
        )
        if target_number not in {"singular", "plural"} \
                or donor_number not in {"singular", "plural"} \
                or target_number == donor_number:
            raise CrossSyntaxAuthorityError("cross-syntax donor does not flip subject number")

        target_ids = _tokens(_side_value(target, target_side, "ids"), "target IDs")
        donor_ids = _tokens(_side_value(donor, donor_side, "ids"), "donor IDs")
        target_position = _token(
            _side_value(target, target_side, "prediction_position"), "target position"
        )
        donor_position = _token(
            _side_value(donor, donor_side, "prediction_position"), "donor position"
        )
        if target_position != len(target_ids) - 1 or donor_position != len(donor_ids) - 1:
            raise CrossSyntaxAuthorityError("prediction position is no longer the final token")
        target_answer = _token(
            _side_value(target, target_side, "answer_id"), "target answer"
        )
        target_foil_text = _text(
            _side_value(target, target_side, "foil"), "target foil text"
        )
        target_foil = ANSWER_TOKEN_IDS.get(target_foil_text)
        if target_foil is None:
            raise CrossSyntaxAuthorityError("target foil left the is/are vocabulary")
        donor_answer = _token(_side_value(donor, donor_side, "answer_id"), "donor answer")
        donor_foil_text = _text(
            _side_value(donor, donor_side, "foil"), "donor foil text"
        )
        donor_foil = ANSWER_TOKEN_IDS.get(donor_foil_text)
        if donor_foil is None:
            raise CrossSyntaxAuthorityError("donor foil left the is/are vocabulary")
        if target_answer != donor_foil or target_foil != donor_answer:
            raise CrossSyntaxAuthorityError("answer/foil orientation is not exactly reversed")

        target_syntax = "pp" if target_family == "A1" else "relative"
        donor_syntax = "pp" if donor_family == "A1" else "relative"
        cell_id = (
            f"{target_syntax}_{target_number}_to_"
            f"{donor_syntax}_{donor_number}"
        )
        output.append({
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "split": "FIT",
            "partition": "VALIDATION",
            "validation_scope": "new_cross_syntax_relations_not_unseen_text",
            "row_id": str(relation["record_id"]),
            "donor_ordinal": int(relation["ordinal"]),
            "cell_id": cell_id,
            "target_family": target_family,
            "donor_family": donor_family,
            "target_endpoint_id": str(relation["target_endpoint_id"]),
            "donor_endpoint_id": str(relation["donor_endpoint_id"]),
            "base_ids": target_ids,
            "donor_ids": donor_ids,
            "base_text": _text(_side_value(target, target_side, "text"), "target text"),
            "donor_text": _text(_side_value(donor, donor_side, "text"), "donor text"),
            "base_answer_id": target_answer,
            "base_foil_id": target_foil,
            "donor_answer_id": donor_answer,
            "donor_foil_id": donor_foil,
            "base_semantic_position": target_position,
            "donor_semantic_position": donor_position,
            "base_subject_number": target_number,
            "donor_subject_number": donor_number,
            "expected_effect": "toward_opposite_number_cross_syntax_donor",
        })
    validate_rows(output)
    return output


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    """Validate the derived relation table independently of source generation."""
    materialized = [dict(row) for row in rows]
    if len(materialized) != 64:
        raise CrossSyntaxAuthorityError("cross-syntax authority must contain 64 rows")
    row_ids = [row.get("row_id") for row in materialized]
    if len(set(row_ids)) != 64:
        raise CrossSyntaxAuthorityError("cross-syntax relation IDs are not unique")
    cells: dict[str, int] = {}
    for row in materialized:
        if row.get("schema") != SCHEMA or row.get("task_id") != TASK_ID \
                or row.get("split") != "FIT" or row.get("partition") != "VALIDATION" \
                or row.get("validation_scope") != \
                "new_cross_syntax_relations_not_unseen_text":
            raise CrossSyntaxAuthorityError("derived row identity changed")
        if {row.get("target_family"), row.get("donor_family")} != {"A1", "A2"}:
            raise CrossSyntaxAuthorityError("derived row is not cross-syntax")
        if row.get("base_subject_number") == row.get("donor_subject_number"):
            raise CrossSyntaxAuthorityError("derived row does not change subject number")
        if row.get("base_answer_id") != row.get("donor_foil_id") \
                or row.get("base_foil_id") != row.get("donor_answer_id"):
            raise CrossSyntaxAuthorityError("derived answer orientation changed")
        base_ids = _tokens(row.get("base_ids"), "derived target IDs")
        donor_ids = _tokens(row.get("donor_ids"), "derived donor IDs")
        if row.get("base_semantic_position") != len(base_ids) - 1 \
                or row.get("donor_semantic_position") != len(donor_ids) - 1:
            raise CrossSyntaxAuthorityError("derived semantic position changed")
        cell_id = _text(row.get("cell_id"), "cell ID")
        cells[cell_id] = cells.get(cell_id, 0) + 1
    expected_cells = {
        "pp_singular_to_relative_plural",
        "pp_plural_to_relative_singular",
        "relative_singular_to_pp_plural",
        "relative_plural_to_pp_singular",
    }
    if set(cells) != expected_cells or set(cells.values()) != {16}:
        raise CrossSyntaxAuthorityError(f"direction-cell balance changed: {cells}")
    return canonical_sha256(materialized)


def authority_sha256() -> str:
    return validate_rows(build_rows())


def compile_plan(rows: Sequence[Mapping[str, object]] | None = None) -> dict[str, object]:
    """Compile the exact eight-call plan; this function cannot load a model."""
    materialized = build_rows() if rows is None else [dict(row) for row in rows]
    digest = validate_rows(materialized)
    calls = []
    for side in ("target", "donor"):
        for start in range(0, len(materialized), BATCH_SIZE):
            chunk = materialized[start:start + BATCH_SIZE]
            calls.append({
                "call_id": f"FIT:VALIDATION:native:{side}:{start // BATCH_SIZE}",
                "kind": "native_paired_logits",
                "side": side,
                "capture": side == "donor",
                "row_ids": [str(row["row_id"]) for row in chunk],
            })
    for site_id in SITE_IDS:
        for start in range(0, len(materialized), BATCH_SIZE):
            chunk = materialized[start:start + BATCH_SIZE]
            calls.append({
                "call_id": (
                    f"FIT:VALIDATION:cross_syntax_patch:{site_id}:"
                    f"{start // BATCH_SIZE}"
                ),
                "kind": "exact_single_position_interchange",
                "site_id": site_id,
                "row_ids": [str(row["row_id"]) for row in chunk],
            })
    if len(calls) != 8:
        raise CrossSyntaxAuthorityError("targeted call plan is not exactly eight calls")
    plan = {
        "schema": "task14_cross_syntax_interchange_plan_v1",
        "task_id": TASK_ID,
        "phase": "FIT",
        "partition": "VALIDATION",
        "validation_scope": "new_cross_syntax_relations_not_unseen_text",
        "authority_sha256": digest,
        "source_sha256": dict(EXPECTED_SOURCE_SHA256),
        "site_ids": list(SITE_IDS),
        "row_count": 64,
        "batch_size": BATCH_SIZE,
        "calls": calls,
        "price": {
            "forward_calls": 8,
            "example_evaluations": 256,
            "backward_calls": 0,
            "model_updates": 0,
            "raw_numeric_evidence_bytes": 2048,
        },
        "score": {
            "target_native_margin": "target_answer_logit - target_foil_logit",
            "donor_native_margin": "donor_answer_logit - donor_foil_logit",
            "patched_target_margin": "target_answer_logit - target_foil_logit",
            "row_recovery": (
                "(target_native_margin - patched_target_margin) / "
                "(target_native_margin + donor_native_margin)"
            ),
            "minimum_native_cell_accuracy": MIN_NATIVE_CELL_ACCURACY,
            "minimum_cell_direction_fraction": MIN_CELL_DIRECTION_FRACTION,
            "minimum_cell_mean_recovery": MIN_CELL_MEAN_RECOVERY,
        },
        "correction": (
            "The v2 fast screen used within-construction A1-to-A1 and A2-to-A2 "
            "donors. A shared passing site was not literal cross-syntax interchange."
        ),
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
    }
    plan["call_manifest_sha256"] = canonical_sha256(calls)
    plan["compiled_sha256"] = canonical_sha256(plan)
    return plan
