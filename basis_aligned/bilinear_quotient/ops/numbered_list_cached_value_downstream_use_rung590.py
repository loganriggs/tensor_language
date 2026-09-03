#!/usr/bin/env python3
"""R590 prospective contract-correct replication of frozen R582/R584 science.

This wrapper preserves the exact R584 model computations and scientific gates,
but writes a new staged evidence/result/receipt package.  It is model-free unless
explicitly invoked with ``--execute-science`` after independent review.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import ast
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, OPS, POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import numbered_list_cached_value_downstream_use_rung584 as r584  # noqa: E402
import audit_numbered_list_cached_value_downstream_use_rung588 as r588  # noqa: E402
import result_contract  # noqa: E402


SCRIPT = Path(__file__).resolve()
TEST = SCRIPT.with_name("test_numbered_list_cached_value_downstream_use_rung590.py")
NOTE = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG590_PROSPECTIVE_CONTRACT_REPLICATION.md"
ADAPTER = SCRIPT.with_name("execute_numbered_list_cached_value_downstream_use_rung590.py")
ADAPTER_TEST = SCRIPT.with_name("test_execute_numbered_list_cached_value_downstream_use_rung590.py")
DRYRUN = ROOT / "numbered_list_cached_value_downstream_use_rung590_dryrun.json"
OUT = ROOT / "numbered_list_cached_value_downstream_use_rung590_results.json"
RECEIPT = ROOT / "numbered_list_cached_value_downstream_use_rung590_receipt.json"
EVIDENCE_DIR = ROOT / "numbered_list_cached_value_downstream_use_rung590_evidence"
EVIDENCE_FILE = EVIDENCE_DIR / "primitive_evidence.json"
STAGE_PREFIX = ".numbered_list_cached_value_downstream_use_rung590_stage-"
RECOVERY_PREFIX = ".numbered_list_cached_value_downstream_use_rung590_recovery-"
STAGE_MARKER_NAME = "r590-stage-marker.json"
STAGE_MARKER_BYTES = b'{"experiment":"r590","schema":"r590-stage-v1"}\n'

ROWS = r584.ROWS
ROWS_RECEIPT = r584.RECEIPT
R582_PREREG = r584.R582_PREREG
R582_HELPER = r584.R582_HELPER
R584_NOTE = r584.R584_PREREG
R584_RUNNER = Path(r584.__file__).resolve()
R588_AUDITOR = Path(r588.__file__).resolve()
BLOCK_REVIEW = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG584_POSTEXECUTION_CONTRACT_AUDIT.md"
BLOCK_TEST = OPS / "test_numbered_list_cached_value_downstream_use_rung584_postexecution_contract_audit.py"
RESULT_CONTRACT = OPS / "result_contract.py"
HANDOFF_V1 = OPS / "circuit_causal_validity_next_wave_handoff_rung585.json"
HANDOFF_V2 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v2_addendum.json"
HANDOFF_V3 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v3_addendum.json"
HANDOFF_V4 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v4_addendum.json"
HANDOFF_V5 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"

AUTHORITY_HASHES = {
    ROWS: "84c6a78882a33c266b3875285f63ceaed746dac7810fce16b591f7b57763cf3b",
    ROWS_RECEIPT: "1511cfd7fcfe729edf4427f9f88f8552c32230e013d01a0661767713fdc29148",
    R582_PREREG: "e7832dc77cabe7a1afba61c759188a0aca73802163cef1abe013ffaff5c987b3",
    R582_HELPER: "b0d99eeeef834091cf9ddfe77b58682f0e9a7e101e143a18570808dacb57bc1c",
    R584_NOTE: "612005760bccda8f1a9f16b540b0734de3241e5da1c40246f514509733539181",
    R584_RUNNER: "50609756d97de2f13f717774f13d72b1c743f38a172375e9b08efc2b055336c7",
    R588_AUDITOR: "b4acebb23bff71c7dc11beec95ff83f5490a86971787bce5930351cfb4572115",
    BLOCK_REVIEW: "2fbefdb84822f4b727de769736f182f1b0864912c9f41f76247cc2df385cb45d",
    BLOCK_TEST: "8508b56c1c9e3d25ccd5f8b4cae0780fc263d0782682d8c57cdc22e8aaaef020",
    RESULT_CONTRACT: "af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272",
    HANDOFF_V1: "e8970f9ef2d7eb7b291a5fb288833bc252e62fabf1016a699e981c19a6be560a",
    HANDOFF_V2: "eb8ef7d00324c7f38210f0e8303951d97282fc8dbede9ee10ef8409db414709b",
    HANDOFF_V3: "bf04cda987fc281f146c1e6f054620934f1d994a5d6d3135d7456be6fe9feb8c",
    HANDOFF_V4: "349afa9ec4fe465dbf08109a63cb1a8dc2a278e53a710bf210035f57b8500da0",
    HANDOFF_V5: "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80",
    NOTE: "8b4019b2da24ee8a6acf73cf1cb35b157e3feece713ca9e90698a0801cf15ab5",
}

SITES = tuple(r584.SITES)
COMPONENTS = tuple(r584.COMPONENTS)
NULLS = tuple(r584.NULLS)
SELECTION = tuple(r584.SELECTION)
SELECTION_NAMES = tuple(f"mlp{site}_{component}" for site, component in SELECTION)
BATCH = int(r584.BATCH)
EXACT_BAR = float(r584.EXACT_BAR)
BOOTSTRAPS = int(r588.BOOTSTRAPS)
CHECKPOINT_SHA256 = r588.CHECKPOINT_SHA256

RESULT_SCHEMA = "numbered_list_cached_value_downstream_use_rung590_result_v1"
RECEIPT_SCHEMA = "numbered_list_cached_value_downstream_use_rung590_receipt_v1"
EVIDENCE_SCHEMA = "numbered_list_cached_value_downstream_use_rung590_evidence_v1"
DRYRUN_SCHEMA = "numbered_list_cached_value_downstream_use_rung590_dryrun_v1"
SHAPE_SCHEMA = "numbered_list_cached_value_downstream_use_rung590_call_shapes_v1"
SHAPE_MODE = "dynamic_batched_token_matrix_exact_common_length_v1"
SUPPORT_SCHEMA = "numbered_list_cached_value_downstream_use_rung590_phase_support_v1"
SUPPORT_CELL_FIELDS = ("condition", "representation", "source_level")
AUTHORIZED_SPLITS = ("FIT", "SELECT", "FINAL_TEST", "OOD")

class UnretainedInstrumentError(RuntimeError):
    """A live tensor check failed and cannot be serialized as a scientific null."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_load_json(path: Path) -> dict:
    value = r588.strict_loads(path.read_bytes(), str(path))
    if type(value) is not dict:
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def validate_authorities() -> dict[str, str]:
    observed = {}
    for path, expected in AUTHORITY_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen R590 authority is missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(
                f"frozen R590 authority changed: {path}; expected={expected}, observed={digest}"
            )
        observed[str(path)] = digest
    # This additionally regenerates and validates the frozen 1,440-row authority.
    observed.update(r588.verify_preoutcome_authority())
    rows, _ = r588.load_authority()
    if len(rows) != 1_440:
        raise RuntimeError("R582 authority row count changed")
    return observed


def source_hashes() -> dict[str, str]:
    observed = validate_authorities()
    for path in (SCRIPT, TEST):
        if not path.is_file():
            raise RuntimeError(f"R590 owned source is missing: {path}")
        observed[str(path)] = sha256(path)
    return observed


def build_phase_support_census(
    rows: Sequence[Mapping[str, object]], splits: Sequence[str],
) -> dict[str, object]:
    """Enumerate exact phase-local support without borrowing or replacement."""
    requested = list(splits)
    if len(requested) != len(set(requested)) \
            or any(split not in AUTHORIZED_SPLITS for split in requested):
        raise RuntimeError("phase support requested an invalid or repeated split")
    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("phase support contains duplicate/replacement row IDs")
    by_split: dict[str, object] = {}
    for split in requested:
        selected = [row for row in rows if row.get("split") == split]
        cells: dict[tuple[object, ...], list[str]] = collections.defaultdict(list)
        for row in selected:
            cells[tuple(row[field] for field in SUPPORT_CELL_FIELDS)].append(
                str(row["row_id"])
            )
        cell_records = []
        for key in sorted(cells, key=lambda item: tuple(str(value) for value in item)):
            ids = cells[key]
            cell_records.append({
                **dict(zip(SUPPORT_CELL_FIELDS, key)),
                "row_count": len(ids),
                "ordered_row_ids": ids,
                "ordered_row_ids_sha256": canonical_sha256(ids),
            })
        ordered_ids = [str(row["row_id"]) for row in selected]
        by_split[split] = {
            "row_count": len(selected),
            "cell_count": len(cell_records),
            "ordered_row_ids": ordered_ids,
            "ordered_row_ids_sha256": canonical_sha256(ordered_ids),
            "cells": cell_records,
        }
    census = {
        "schema": SUPPORT_SCHEMA,
        "cell_fields": list(SUPPORT_CELL_FIELDS),
        "splits": by_split,
    }
    r588.validate_standard_json(census)
    return census


def validate_full_phase_panel(
    panel_rows: Sequence[Mapping[str, object]],
    authority_rows: Sequence[Mapping[str, object]], split: str,
) -> dict[str, object]:
    """Require the full ordered authority panel for one opened phase."""
    if split not in AUTHORIZED_SPLITS:
        raise RuntimeError("phase panel split is unauthorized")
    observed = list(panel_rows)
    if any(row.get("split") != split for row in observed):
        raise RuntimeError("phase panel borrowed a row from another split")
    expected = [row for row in authority_rows if row.get("split") == split]
    observed_ids = [str(row["row_id"]) for row in observed]
    expected_ids = [str(row["row_id"]) for row in expected]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("phase panel sampled with replacement or duplicated a row")
    if observed_ids != expected_ids:
        raise RuntimeError("phase panel silently shrank, reordered, or changed membership")
    report = build_phase_support_census(observed, [split])
    split_report = report["splits"][split]
    expected_per_cell = 16 if split == "FIT" else 8
    if split_report["cell_count"] != 36 \
            or any(cell["row_count"] != expected_per_cell for cell in split_report["cells"]):
        raise RuntimeError("phase panel has missing or incorrectly sized support cells")
    return report


def frozen_phase_support_census(
    rows: Sequence[Mapping[str, object]], splits: Sequence[str],
) -> dict[str, object]:
    for split in splits:
        validate_full_phase_panel(
            [row for row in rows if row.get("split") == split], rows, split
        )
    return build_phase_support_census(rows, splits)


def _batch_records(rows: Sequence[dict], split: str, *, eligible_null: bool = False):
    selected = [row for row in rows if row["split"] == split]
    if eligible_null:
        selected = [row for row in selected if row["condition"] in r588.ELIGIBLE_CONDITIONS]
    return r584.chunks(selected, lambda row: len(row["ids"]))


def _call_record(
    *, call_id: str, split: str, guard: str, kind: str,
    group: Sequence[Mapping[str, object]], arm: str,
) -> dict[str, object]:
    lengths = {len(row["ids"]) for row in group}
    if len(lengths) != 1:
        raise RuntimeError(f"{call_id}: logical batch mixes token lengths")
    return {
        "call_id": call_id,
        "split": split,
        "guard": guard,
        "call_kind": kind,
        "arm": arm,
        "logical_batch_size": len(group),
        "padded_sequence_length": next(iter(lengths)),
        "row_ids": [str(row["row_id"]) for row in group],
        "shape_validation_mode": SHAPE_MODE,
        "checkpoint_validation": "facade_verified_sha256",
        "model_structure_validation": "facade_bilin18_structure",
    }


def build_forward_call_manifest(rows: Sequence[dict]) -> list[dict[str, object]]:
    """Enumerate every possible priced high-level source-model call."""
    calls: list[dict[str, object]] = []
    for split in ("FIT", "SELECT"):
        batches = _batch_records(rows, split)
        for index, group in enumerate(batches):
            for mode in ("source_present", "source_deleted"):
                calls.append(_call_record(
                    call_id=f"{split}:capture:{index}:{mode}", split=split,
                    guard="fit_always" if split == "FIT" else "selected_only",
                    kind="trajectory", group=group, arm=mode,
                ))
            if index == 0:
                calls.append(_call_record(
                    call_id=f"{split}:capture:{index}:native_smoke", split=split,
                    guard="fit_always" if split == "FIT" else "selected_only",
                    kind="native_logits_smoke", group=group, arm="native_smoke",
                ))
        if split == "FIT":
            candidates = SELECTION_NAMES
            guard = "fit_always"
        else:
            # The site is chosen on FIT; all three component names have identical shapes.
            candidates = tuple(f"selected_site_{component}" for component in COMPONENTS)
            guard = "selected_only"
        for arm in candidates:
            for index, group in enumerate(batches):
                calls.append(_call_record(
                    call_id=f"{split}:real:{arm}:{index}", split=split, guard=guard,
                    kind="component_suffix", group=group, arm=arm,
                ))
        for null_name in NULLS:
            for index, group in enumerate(_batch_records(rows, split, eligible_null=True)):
                calls.append(_call_record(
                    call_id=f"{split}:null:{null_name}:{index}", split=split,
                    guard="provisional_only" if split == "FIT" else "selected_only",
                    kind="null_component_suffix", group=group, arm=null_name,
                ))
    return calls


def source_forward_callsite_census() -> dict[str, int]:
    """Pin the high-level model calls inside the immutable R584 implementation."""
    tree = ast.parse(R584_RUNNER.read_text(encoding="utf-8"))
    wanted = {"capture_split", "evaluate_component"}
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    }
    if set(functions) != wanted:
        raise RuntimeError("R584 source-model entry functions changed")

    def count_callee(function, callee: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == callee)
                or (isinstance(node.func, ast.Name) and node.func.id == callee)
            )
            for node in ast.walk(function)
        )

    census = {
        "capture_split.trajectory": count_callee(functions["capture_split"], "trajectory"),
        "capture_split.native_logits": count_callee(functions["capture_split"], "native_logits"),
        "evaluate_component.component_forward": count_callee(
            functions["evaluate_component"], "component_forward"
        ),
    }
    expected = {
        "capture_split.trajectory": 2,
        "capture_split.native_logits": 1,
        "evaluate_component.component_forward": 1,
    }
    if census != expected:
        raise RuntimeError(f"hidden or missing R584 source-model call site: {census}")
    return census


def wrapper_science_callsite_census() -> dict[str, int]:
    """Reject a hidden model path added outside the pinned R584 entry points."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_science"
    }
    if set(functions) != {"run_science"}:
        raise RuntimeError("R590 run_science function is missing or duplicated")
    function = functions["run_science"]

    def count(attribute: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attribute
            for node in ast.walk(function)
        )

    observed = {
        "run_science.facade_load_bilin18": count("load_bilin18"),
        "run_science.capture_split": count("capture_split"),
        "run_science.evaluate_component": count("evaluate_component"),
    }
    expected = {
        "run_science.facade_load_bilin18": 1,
        "run_science.capture_split": 2,
        "run_science.evaluate_component": 4,
    }
    if observed != expected:
        raise RuntimeError(f"hidden or missing R590 scientific call site: {observed}")
    return observed


def validate_forward_call_manifest(
    manifest: Sequence[Mapping[str, object]], rows: Sequence[dict]
) -> dict[str, object]:
    expected = build_forward_call_manifest(rows)
    if list(manifest) != expected:
        raise RuntimeError("forward-call manifest differs from frozen schedule")
    if len(manifest) != 510:
        raise RuntimeError(f"forward-call manifest has {len(manifest)} calls, expected 510")
    ids = [str(item["call_id"]) for item in manifest]
    if len(ids) != len(set(ids)):
        raise RuntimeError("forward-call manifest repeats a call ID")
    authority = {str(row["row_id"]): row for row in rows}
    shape_counts = collections.Counter()
    kind_counts = collections.Counter()
    guard_counts = collections.Counter()
    for call in manifest:
        if call.get("shape_validation_mode") != SHAPE_MODE:
            raise RuntimeError(f"{call.get('call_id')}: incompatible shape validation mode")
        batch = call.get("logical_batch_size")
        length = call.get("padded_sequence_length")
        row_ids = call.get("row_ids")
        if type(batch) is not int or isinstance(batch, bool) or not 1 <= batch <= BATCH:
            raise RuntimeError(f"{call.get('call_id')}: batch size violates R582 schedule")
        if type(length) is not int or isinstance(length, bool) or length <= 0:
            raise RuntimeError(f"{call.get('call_id')}: token length is invalid")
        if type(row_ids) is not list or len(row_ids) != batch or len(set(row_ids)) != batch:
            raise RuntimeError(f"{call.get('call_id')}: row membership is invalid")
        for row_id in row_ids:
            row = authority.get(str(row_id))
            if row is None or row["split"] != call.get("split") or len(row["ids"]) != length:
                raise RuntimeError(f"{call.get('call_id')}: row shape/authority mismatch")
        if call.get("checkpoint_validation") != "facade_verified_sha256" \
                or call.get("model_structure_validation") != "facade_bilin18_structure":
            raise RuntimeError(f"{call.get('call_id')}: model/checkpoint validation missing")
        shape_counts[f"{batch}x{length}"] += 1
        kind_counts[str(call["call_kind"])] += 1
        guard_counts[str(call["guard"])] += 1
    census = source_forward_callsite_census()
    wrapper_census = wrapper_science_callsite_census()
    return {
        "schema": SHAPE_SCHEMA,
        "call_count": len(manifest),
        "manifest_sha256": canonical_sha256(list(manifest)),
        "shape_counts": dict(sorted(shape_counts.items())),
        "call_kind_counts": dict(sorted(kind_counts.items())),
        "guard_counts": dict(sorted(guard_counts.items())),
        "source_callsite_census": census,
        "wrapper_callsite_census": wrapper_census,
        "maximum_batch_size": BATCH,
        "validation_mode": SHAPE_MODE,
        "fixed_4x256_mode_rejected": True,
    }


def expected_executed_call_ids(
    manifest: Sequence[Mapping[str, object]], provisional: str | None,
    selected: str | None,
) -> list[str]:
    if selected is not None and selected != provisional:
        raise RuntimeError("selected component differs from provisional component")
    allowed_guards = {"fit_always"}
    if provisional is not None:
        allowed_guards.add("provisional_only")
    if selected is not None:
        allowed_guards.add("selected_only")
    return [str(call["call_id"]) for call in manifest if call["guard"] in allowed_guards]


def hard_abort_unretained_exactness(
    captures: Sequence[Mapping[str, object]], exactness: Mapping[str, object], label: str
) -> None:
    # exactness_summary also joins the retained per-row replay and bilinear maxima.
    try:
        _, passed = r588.exactness_summary(list(captures), exactness, label)
    except (KeyError, TypeError, ValueError, RuntimeError) as exactness_error:
        raise UnretainedInstrumentError(
            f"{label} replay/tensor exactness evidence is inconsistent before publishable evidence"
        ) from exactness_error
    if not passed:
        raise UnretainedInstrumentError(
            f"{label} replay/tensor exactness failed before publishable evidence"
        )
    if any(
        float(item["native_replay_relative_squared_error_by_row"]["maximum"]) > EXACT_BAR
        or float(item["bilinear_response_relative_squared_error"]) > EXACT_BAR
        for item in captures
    ):
        raise UnretainedInstrumentError(
            f"{label} retained exactness maximum exceeded {EXACT_BAR}"
        )


def evidence_from_legacy_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Extract only primitive evidence from an R584-shaped planted payload."""
    rows = r584.load_authority()
    manifest = build_forward_call_manifest(rows)
    provisional = payload.get("provisional_fit_selection")
    selected = payload.get("selected_component")
    opened = list(payload["evaluated_splits"])
    support = frozen_phase_support_census(rows, opened)
    return {
        "schema": EVIDENCE_SCHEMA,
        "rung": 590,
        "source_rung": 584,
        "result_path": str(OUT.relative_to(ROOT.parent.parent)),
        "receipt_path": str(RECEIPT.relative_to(ROOT.parent.parent)),
        "evaluated_splits": opened,
        "phase_support_census": support,
        "phase_support_census_sha256": canonical_sha256(support),
        "fit_exactness": payload["fit_exactness"],
        "select_exactness": payload["select_exactness"],
        "fit_capture_raw": payload["fit_capture_raw"],
        "select_capture_raw": payload["select_capture_raw"],
        "fit_raw": payload["fit_raw"],
        "fit_null_raw": payload["fit_null_raw"],
        "select_raw": payload["select_raw"],
        "select_null_raw": payload["select_null_raw"],
        "executed_forward_call_ids": expected_executed_call_ids(
            manifest, provisional, selected
        ),
    }


def _compare_exact(expected: object, observed: object, label: str) -> None:
    failures: list[str] = []
    r588.compare(expected, observed, label, failures)
    if failures:
        raise RuntimeError(";".join(failures[:20]))


def validate_intervention_capture_join(
    records: Sequence[Mapping[str, object]],
    capture_by_id: Mapping[str, Mapping[str, object]],
    label: str,
) -> None:
    """Bind every arm's unchanged endpoints to its captured native trajectory."""
    for record in records:
        row_id = str(record["row_id"])
        if row_id not in capture_by_id:
            raise RuntimeError(f"{label}.{row_id}: capture endpoint is missing")
        capture = capture_by_id[row_id]
        for field in (
            "native", "source_deleted",
            "source_deleted_logit_difference_squared_sum",
            "source_deleted_logit_vocabulary_count",
            "source_deleted_full_vocabulary_logit_rms",
        ):
            _compare_exact(
                capture[field], record[field], f"{label}.{row_id}.{field}"
            )


def derive_scientific_summary(
    evidence: Mapping[str, object], *, replicates: int = BOOTSTRAPS,
) -> dict[str, object]:
    """Reconstruct every report, split decision, and terminal from primitives."""
    r588.validate_standard_json(evidence)
    expected_keys = {
        "schema", "rung", "source_rung", "result_path", "receipt_path",
        "evaluated_splits", "fit_exactness", "select_exactness",
        "fit_capture_raw", "select_capture_raw", "fit_raw", "fit_null_raw",
        "select_raw", "select_null_raw", "executed_forward_call_ids",
        "phase_support_census", "phase_support_census_sha256",
    }
    if type(evidence) is not dict or set(evidence) != expected_keys:
        raise RuntimeError("R590 evidence fields changed")
    if evidence["schema"] != EVIDENCE_SCHEMA or evidence["rung"] != 590 \
            or evidence["source_rung"] != 584:
        raise RuntimeError("R590 evidence identity changed")
    if evidence["result_path"] != str(OUT.relative_to(ROOT.parent.parent)) \
            or evidence["receipt_path"] != str(RECEIPT.relative_to(ROOT.parent.parent)):
        raise RuntimeError("R590 evidence logical package paths changed")
    rows, helper = r588.load_authority()
    manifest = build_forward_call_manifest(rows)
    validate_forward_call_manifest(manifest, rows)

    opened = evidence["evaluated_splits"]
    if opened not in (["FIT"], ["FIT", "SELECT"]):
        raise RuntimeError("R590 opened split sequence is invalid")
    expected_support = frozen_phase_support_census(rows, opened)
    _compare_exact(
        expected_support, evidence["phase_support_census"],
        "evidence.phase_support_census",
    )
    expected_support_sha256 = canonical_sha256(expected_support)
    if evidence["phase_support_census_sha256"] != expected_support_sha256:
        raise RuntimeError("phase support census hash differs from frozen opened panels")
    fit_capture = list(evidence["fit_capture_raw"])
    fit_capture_map = r588.validate_capture(fit_capture, rows, "FIT")
    hard_abort_unretained_exactness(fit_capture, evidence["fit_exactness"], "FIT")
    bootstrapper = r588.Bootstrapper(replicates)

    if type(evidence["fit_raw"]) is not dict or set(evidence["fit_raw"]) != set(SELECTION_NAMES):
        raise RuntimeError("R590 FIT arm set differs from frozen selection")
    fit_raw, fit_reports = {}, {}
    for site, component in SELECTION:
        name = f"mlp{site}_{component}"
        fit_raw[name] = r588.validate_interventions(
            evidence["fit_raw"][name], rows, "FIT", site, component, name, helper
        )
        validate_intervention_capture_join(fit_raw[name], fit_capture_map, f"FIT.{name}")
        fit_reports[name] = r588.score_candidate(
            fit_raw[name], "FIT", f"FIT:{name}", bootstrapper
        )
    provisional = next(
        (name for name in SELECTION_NAMES if fit_reports[name]["passed_without_nulls"]),
        None,
    )

    selected = None
    fit_null_reports = None
    if provisional is None:
        if evidence["fit_null_raw"] is not None:
            raise RuntimeError("FIT null evidence exists without a provisional candidate")
    else:
        site = int(provisional[3:].split("_", 1)[0])
        component = provisional[3:].split("_", 1)[1]
        keys = {f"{provisional}:null:{null_name}" for null_name in NULLS}
        if type(evidence["fit_null_raw"]) is not dict \
                or set(evidence["fit_null_raw"]) != keys:
            raise RuntimeError("R590 FIT null arm set changed")
        fit_null_reports = {}
        for null_name in NULLS:
            key = f"{provisional}:null:{null_name}"
            null_raw = r588.validate_interventions(
                evidence["fit_null_raw"][key], rows, "FIT", site, component,
                f"null:{null_name}", helper, null_name=null_name,
            )
            validate_intervention_capture_join(null_raw, fit_capture_map, f"FIT.{key}")
            fit_null_reports[key] = r588.score_null(
                fit_raw[provisional], null_raw, "FIT", f"FIT:{key}",
                fit_reports[provisional], null_name, bootstrapper,
            )
        if all(report["passed"] for report in fit_null_reports.values()):
            selected = provisional

    select_reports = select_null_reports = None
    select_raw = None
    select_exactness = None
    if selected is None:
        if opened != ["FIT"]:
            raise RuntimeError("SELECT opened without a selected FIT component")
        for field in ("select_exactness", "select_capture_raw", "select_raw", "select_null_raw"):
            if evidence[field] is not None:
                raise RuntimeError(f"{field} exists while SELECT is closed")
    else:
        if opened != ["FIT", "SELECT"]:
            raise RuntimeError("selected component exists without SELECT opening")
        select_capture = list(evidence["select_capture_raw"])
        select_capture_map = r588.validate_capture(select_capture, rows, "SELECT")
        hard_abort_unretained_exactness(
            select_capture, evidence["select_exactness"], "SELECT"
        )
        select_exactness = dict(evidence["select_exactness"])
        site = int(selected[3:].split("_", 1)[0])
        selected_component = selected[3:].split("_", 1)[1]
        select_names = {f"mlp{site}_{component}" for component in COMPONENTS}
        if type(evidence["select_raw"]) is not dict or set(evidence["select_raw"]) != select_names:
            raise RuntimeError("R590 SELECT real arm set changed")
        select_raw, select_reports = {}, {}
        for component in COMPONENTS:
            name = f"mlp{site}_{component}"
            select_raw[name] = r588.validate_interventions(
                evidence["select_raw"][name], rows, "SELECT", site, component,
                name, helper,
            )
            validate_intervention_capture_join(
                select_raw[name], select_capture_map, f"SELECT.{name}"
            )
            select_reports[name] = r588.score_candidate(
                select_raw[name], "SELECT", f"SELECT:{name}", bootstrapper,
                frozen_scales=fit_reports[selected]["fit_scales"],
            )
        null_keys = {f"{selected}:null:{null_name}" for null_name in NULLS}
        if type(evidence["select_null_raw"]) is not dict \
                or set(evidence["select_null_raw"]) != null_keys:
            raise RuntimeError("R590 SELECT null arm set changed")
        select_null_reports = {}
        for null_name in NULLS:
            key = f"{selected}:null:{null_name}"
            null_raw = r588.validate_interventions(
                evidence["select_null_raw"][key], rows, "SELECT", site,
                selected_component, f"null:{null_name}", helper, null_name=null_name,
            )
            validate_intervention_capture_join(
                null_raw, select_capture_map, f"SELECT.{key}"
            )
            select_null_reports[key] = r588.score_null(
                select_raw[selected], null_raw, "SELECT", f"SELECT:{key}",
                select_reports[selected], null_name, bootstrapper,
            )

    fit_selected_pass = selected is not None
    select_selected_pass = bool(
        selected is not None
        and select_reports[selected]["passed_without_nulls"]
        and all(report["passed"] for report in select_null_reports.values())
    )
    reuse = bool(
        fit_selected_pass and fit_reports[selected]["all_representations_pass"]
        and select_selected_pass and select_reports[selected]["all_representations_pass"]
    )
    all_pass = bool(fit_selected_pass and select_selected_pass and reuse)
    interactions = {"fit": None, "select": None}
    if selected is not None:
        site = int(selected[3:].split("_", 1)[0])
        interactions["fit"] = r588.interaction_records(fit_raw, site)
        interactions["select"] = r588.interaction_records(select_raw, site)
    expected_ids = expected_executed_call_ids(manifest, provisional, selected)
    if evidence["executed_forward_call_ids"] != expected_ids:
        raise RuntimeError("saved executed call IDs differ from the frozen conditional schedule")
    trace_hash = canonical_sha256({
        key: bootstrapper.traces[key] for key in sorted(bootstrapper.traces)
    })
    return {
        "provisional_fit_selection": provisional,
        "selected_component": selected,
        "fit_reports": fit_reports,
        "fit_null_reports": fit_null_reports,
        "select_reports": select_reports,
        "select_null_reports": select_null_reports,
        "component_interactions": interactions,
        "fit_exactness": dict(evidence["fit_exactness"]),
        "select_exactness": select_exactness,
        "evaluated_splits": list(opened),
        "model_forwards": len(expected_ids),
        "pred_a_exact_prefix_and_bilinear_decomposition": True,
        "pred_b_selective_downstream_action_component": bool(
            fit_selected_pass and select_selected_pass
        ),
        "pred_c_cross_representation_reuse": reuse,
        "pred_d_evidence_derived_terminal": True,
        "all_required_gates_pass": all_pass,
        "decision": (
            "downstream_use_component_held" if all_pass
            else "downstream_use_decomposition_null"
        ),
        "next_step": (
            "independent_cpu_audit_then_FINAL_TEST_remains_separately_preregistered"
            if all_pass else
            "retain_R576_broad_carrier_and_do_not_promote_R582_component"
        ),
        "bootstrap_replicates_per_cell": replicates,
        "bootstrap_cell_count": len(bootstrapper.traces),
        "bootstrap_trace_sha256": trace_hash,
        "executed_forward_call_ids": expected_ids,
        "phase_support_census": expected_support,
        "phase_support_census_sha256": expected_support_sha256,
    }


def execution_plan(rows: Sequence[dict]) -> dict[str, object]:
    manifest = build_forward_call_manifest(rows)
    shape = validate_forward_call_manifest(manifest, rows)
    pricing = r584.price(rows)
    if pricing["literal_executable_maximum_forwards"] != len(manifest):
        raise RuntimeError("R584 price and R590 call-shape manifest disagree")
    support = frozen_phase_support_census(rows, AUTHORIZED_SPLITS)
    return {
        "schema": DRYRUN_SCHEMA,
        "status": "deterministic_cpu_dryrun_passed",
        "rung": 590,
        "source_rung": 584,
        "rows": len(rows),
        "fit_rows": sum(row["split"] == "FIT" for row in rows),
        "select_rows": sum(row["split"] == "SELECT" for row in rows),
        **pricing,
        "selection_order": list(SELECTION_NAMES),
        "forward_call_shape_contract": shape,
        "phase_support_census": support,
        "phase_support_census_sha256": canonical_sha256(support),
        "opened_splits": [],
        "forbidden_splits_opened": [],
        "model_loaded": False,
        "cuda_opened": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "input_sha256": source_hashes(),
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "note_sha256": sha256(NOTE),
        "result_namespace": str(OUT.relative_to(ROOT.parent.parent)),
        "receipt_namespace": str(RECEIPT.relative_to(ROOT.parent.parent)),
        "evidence_namespace": str(EVIDENCE_DIR.relative_to(ROOT.parent.parent)),
    }


def validate_dryrun(plan: Mapping[str, object]) -> None:
    result_contract.validate_standard_json(plan)
    if plan.get("schema") != DRYRUN_SCHEMA or plan.get("status") != "deterministic_cpu_dryrun_passed":
        raise RuntimeError("R590 dry-run identity changed")
    if plan.get("opened_splits") != [] or plan.get("forbidden_splits_opened") != []:
        raise RuntimeError("R590 dry run opened a scientific split")
    if plan.get("model_loaded") is not False or plan.get("cuda_opened") is not False:
        raise RuntimeError("R590 dry run loaded model or CUDA")
    result_contract.validate_execution_envelope(
        plan, min_forwards=0, max_forwards=0, exact_forwards=0,
        expected_backwards=0, expected_weights_updated=False,
        weights_updated_field="model_weights_updated",
    )
    rows = r584.load_authority()
    expected = execution_plan(rows)
    if dict(plan) != expected:
        raise RuntimeError("R590 dry run differs from current exact sources")


def run_dryrun(output_path: Path | None = None) -> dict[str, object]:
    rows = r584.load_authority()
    plan = execution_plan(rows)
    validate_dryrun(plan)
    if output_path is not None:
        output_path.write_bytes(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    return plan


def build_result(
    evidence: Mapping[str, object], *, evidence_sha256: str,
    checkpoint_sha256: str, elapsed_seconds: float,
    replicates: int = BOOTSTRAPS,
) -> dict[str, object]:
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash changed")
    derived = derive_scientific_summary(evidence, replicates=replicates)
    rows = r584.load_authority()
    plan = execution_plan(rows)
    return {
        "schema": RESULT_SCHEMA,
        "rung": 590,
        "source_rung": 584,
        "stage": "cached_value_downstream_bilinear_use_contract_replication",
        **derived,
        "execution_plan": plan,
        "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": checkpoint_sha256,
        "forbidden_splits_opened": [],
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "note_sha256": sha256(NOTE),
        "input_sha256": source_hashes(),
        "evidence_descriptor": {
            "path": str(EVIDENCE_FILE.relative_to(ROOT.parent.parent)),
            "sha256": evidence_sha256,
            "schema": EVIDENCE_SCHEMA,
        },
        "receipt_path": str(RECEIPT.relative_to(ROOT.parent.parent)),
        "elapsed_seconds": float(elapsed_seconds),
    }


def validate_result_against_evidence(
    result: Mapping[str, object], evidence: Mapping[str, object], *,
    replicates: int = BOOTSTRAPS,
) -> dict[str, object]:
    r588.validate_standard_json(result)
    if type(result) is not dict or result.get("schema") != RESULT_SCHEMA \
            or result.get("rung") != 590 or result.get("source_rung") != 584:
        raise RuntimeError("R590 result identity changed")
    derived = derive_scientific_summary(evidence, replicates=replicates)
    derived_keys = set(derived)
    expected_result_keys = derived_keys | {
        "schema", "rung", "source_rung", "stage", "execution_plan",
        "model_backwards", "model_weights_updated", "checkpoint_weights_sha256",
        "forbidden_splits_opened", "implementation_sha256", "test_sha256",
        "note_sha256", "input_sha256", "evidence_descriptor", "receipt_path",
        "elapsed_seconds",
    }
    if set(result) != expected_result_keys:
        raise RuntimeError("R590 result field set changed")
    evidence_digest = canonical_sha256(evidence)
    descriptor = result.get("evidence_descriptor")
    if descriptor != {
        "path": str(EVIDENCE_FILE.relative_to(ROOT.parent.parent)),
        "sha256": evidence_digest,
        "schema": EVIDENCE_SCHEMA,
    }:
        raise RuntimeError("R590 result does not bind exact evidence bytes")
    if result.get("receipt_path") != str(RECEIPT.relative_to(ROOT.parent.parent)):
        raise RuntimeError("R590 receipt path changed")
    for key, expected in derived.items():
        _compare_exact(expected, result.get(key), f"result.{key}")
    expected_constants = {
        "stage": "cached_value_downstream_bilinear_use_contract_replication",
        "execution_plan": execution_plan(r584.load_authority()),
        "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "forbidden_splits_opened": [],
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "note_sha256": sha256(NOTE),
        "input_sha256": source_hashes(),
    }
    for key, expected in expected_constants.items():
        _compare_exact(expected, result.get(key), f"result.{key}")
    elapsed = result.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or isinstance(elapsed, bool) \
            or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
        raise RuntimeError("R590 elapsed_seconds is invalid")
    return derived


def make_receipt(result_bytes: bytes, evidence_bytes: bytes, result: Mapping[str, object]) -> dict:
    result_digest = hashlib.sha256(result_bytes).hexdigest()
    evidence_digest = hashlib.sha256(evidence_bytes).hexdigest()
    package_id = hashlib.sha256(f"r590:{result_digest}:{evidence_digest}".encode()).hexdigest()
    return {
        "schema": RECEIPT_SCHEMA,
        "rung": 590,
        "source_rung": 584,
        "package_id": package_id,
        "result_path": str(OUT.relative_to(ROOT.parent.parent)),
        "result_sha256": result_digest,
        "evidence_path": str(EVIDENCE_FILE.relative_to(ROOT.parent.parent)),
        "evidence_sha256": evidence_digest,
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "note_sha256": sha256(NOTE),
        "checkpoint_weights_sha256": result["checkpoint_weights_sha256"],
        "evaluated_splits": result["evaluated_splits"],
        "model_forwards": result["model_forwards"],
        "model_backwards": 0,
        "model_weights_updated": False,
        "decision": result["decision"],
    }


def validate_receipt(
    receipt: Mapping[str, object], result_bytes: bytes, evidence_bytes: bytes,
    result: Mapping[str, object],
) -> None:
    r588.validate_standard_json(receipt)
    expected = make_receipt(result_bytes, evidence_bytes, result)
    if dict(receipt) != expected:
        raise RuntimeError("R590 receipt differs from exact result/evidence package")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_fsync(
    path: Path, payload: bytes, *, label: str, crash_injector: Callable[[str], None] | None,
) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if crash_injector is not None:
        crash_injector(label)


def create_stage_root(root: Path = ROOT) -> Path:
    stage = Path(tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=root))
    if stage.stat().st_dev != root.stat().st_dev:
        raise RuntimeError("R590 stage is not on the final output filesystem")
    _write_bytes_fsync(
        stage / STAGE_MARKER_NAME, STAGE_MARKER_BYTES,
        label="stage_marker", crash_injector=None,
    )
    _fsync_directory(stage)
    return stage


def stage_package(
    evidence: Mapping[str, object], *, checkpoint_sha256: str,
    elapsed_seconds: float, stage_root: Path | None = None,
    replicates: int = BOOTSTRAPS,
    crash_injector: Callable[[str], None] | None = None,
) -> tuple[Path, dict, dict]:
    stage_root = create_stage_root() if stage_root is None else stage_root
    stage_evidence = stage_root / "evidence"
    stage_evidence.mkdir()
    evidence_bytes = canonical_bytes(evidence)
    staged_evidence_file = stage_evidence / EVIDENCE_FILE.name
    _write_bytes_fsync(
        staged_evidence_file, evidence_bytes, label="evidence",
        crash_injector=crash_injector,
    )
    _fsync_directory(stage_evidence)
    result = build_result(
        evidence, evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        checkpoint_sha256=checkpoint_sha256, elapsed_seconds=elapsed_seconds,
        replicates=replicates,
    )
    validate_result_against_evidence(result, evidence, replicates=replicates)
    result_bytes = canonical_bytes(result)
    _write_bytes_fsync(
        stage_root / "result.json", result_bytes, label="result",
        crash_injector=crash_injector,
    )
    receipt = make_receipt(result_bytes, evidence_bytes, result)
    receipt_bytes = canonical_bytes(receipt)
    validate_receipt(receipt, result_bytes, evidence_bytes, result)
    _write_bytes_fsync(
        stage_root / "receipt.json", receipt_bytes, label="receipt",
        crash_injector=crash_injector,
    )
    _fsync_directory(stage_root)
    return stage_root, result, receipt


def publish_staged_package(
    stage_root: Path, *, out: Path = OUT, receipt_path: Path = RECEIPT,
    evidence_dir: Path = EVIDENCE_DIR,
    replicates: int = BOOTSTRAPS,
    crash_injector: Callable[[str], None] | None = None,
) -> None:
    moves = (
        ("evidence", stage_root / "evidence", evidence_dir),
        ("result", stage_root / "result.json", out),
        ("receipt", stage_root / "receipt.json", receipt_path),
    )
    if any(destination.exists() for _, _, destination in moves):
        raise RuntimeError("R590 final namespace became occupied")
    if any(not source.exists() for _, source, _ in moves):
        raise RuntimeError("R590 staged package is incomplete")
    staged_evidence_file = stage_root / "evidence" / EVIDENCE_FILE.name
    evidence = strict_load_json(staged_evidence_file)
    result = strict_load_json(stage_root / "result.json")
    receipt = strict_load_json(stage_root / "receipt.json")
    evidence_bytes = staged_evidence_file.read_bytes()
    result_bytes = (stage_root / "result.json").read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != result["evidence_descriptor"]["sha256"]:
        raise RuntimeError("staged evidence bytes differ from result descriptor")
    validate_result_against_evidence(result, evidence, replicates=replicates)
    validate_receipt(receipt, result_bytes, evidence_bytes, result)
    published = []
    try:
        for label, source, destination in moves:
            os.replace(source, destination)
            published.append((source, destination))
            _fsync_directory(destination.parent)
            if crash_injector is not None:
                crash_injector(f"published_{label}")
    except BaseException:
        for source, destination in reversed(published):
            if destination.exists() and not source.exists():
                os.replace(destination, source)
        _fsync_directory(stage_root)
        _fsync_directory(out.parent)
        raise
    marker = stage_root / STAGE_MARKER_NAME
    if marker.read_bytes() != STAGE_MARKER_BYTES:
        raise RuntimeError("R590 stage marker changed")
    marker.unlink()
    stage_root.rmdir()
    _fsync_directory(out.parent)


def _recognized_evidence_dir(path: Path, *, allow_incomplete: bool) -> None:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"unsafe unrecognized R590 evidence path: {path}")
    children = list(path.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise RuntimeError(f"unsafe unrecognized R590 evidence bytes: {path}")
    names = {child.name for child in children}
    if not names <= {EVIDENCE_FILE.name} or (not allow_incomplete and names != {EVIDENCE_FILE.name}):
        raise RuntimeError(f"unsafe unrecognized R590 evidence census: {path}")
    if EVIDENCE_FILE.name in names and not allow_incomplete:
        try:
            value = strict_load_json(path / EVIDENCE_FILE.name)
        except (OSError, TypeError, ValueError, RuntimeError) as evidence_error:
            raise RuntimeError(f"unsafe unrecognized R590 evidence bytes: {path}") from evidence_error
        if value.get("schema") != EVIDENCE_SCHEMA or value.get("rung") != 590:
            raise RuntimeError(f"unsafe unrecognized R590 evidence identity: {path}")


def _recognized_result(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"unsafe unrecognized R590 result path: {path}")
    try:
        value = strict_load_json(path)
    except (OSError, TypeError, ValueError, RuntimeError) as result_error:
        raise RuntimeError(f"unsafe unrecognized R590 result bytes: {path}") from result_error
    if value.get("schema") != RESULT_SCHEMA or value.get("rung") != 590 \
            or value.get("implementation_sha256") != sha256(SCRIPT) \
            or value.get("test_sha256") != sha256(TEST):
        raise RuntimeError(f"unsafe unrecognized R590 result bytes: {path}")


def _recognized_receipt(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"unsafe unrecognized R590 receipt path: {path}")
    try:
        value = strict_load_json(path)
    except (OSError, TypeError, ValueError, RuntimeError) as receipt_error:
        raise RuntimeError(f"unsafe unrecognized R590 receipt bytes: {path}") from receipt_error
    if value.get("schema") != RECEIPT_SCHEMA or value.get("rung") != 590 \
            or value.get("implementation_sha256") != sha256(SCRIPT) \
            or value.get("test_sha256") != sha256(TEST):
        raise RuntimeError(f"unsafe unrecognized R590 receipt bytes: {path}")


def _recognized_stage(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"unsafe unrecognized R590 stage: {path}")
    marker = path / STAGE_MARKER_NAME
    if not marker.is_file() or marker.is_symlink() or marker.read_bytes() != STAGE_MARKER_BYTES:
        raise RuntimeError(f"unsafe unrecognized R590 stage marker: {path}")
    allowed = {STAGE_MARKER_NAME, "evidence", "result.json", "receipt.json"}
    children = list(path.iterdir())
    if any(child.name not in allowed or child.is_symlink() for child in children):
        raise RuntimeError(f"unsafe unrecognized R590 stage bytes: {path}")
    if (path / "evidence").exists():
        _recognized_evidence_dir(path / "evidence", allow_incomplete=True)
    if (path / "result.json").exists():
        result = path / "result.json"
        if not result.is_file() or result.is_symlink():
            raise RuntimeError(f"unsafe unrecognized R590 staged result: {result}")
    if (path / "receipt.json").exists():
        receipt = path / "receipt.json"
        if not receipt.is_file() or receipt.is_symlink():
            raise RuntimeError(f"unsafe unrecognized R590 staged receipt: {receipt}")


def recover_stale_publication(
    *, root: Path = ROOT, out: Path = OUT, receipt_path: Path = RECEIPT,
    evidence_dir: Path = EVIDENCE_DIR,
) -> None:
    finals = {"result": out, "receipt": receipt_path, "evidence": evidence_dir}
    occupied = {name: path for name, path in finals.items() if path.exists()}
    stages = sorted(root.glob(STAGE_PREFIX + "*"))
    if len(occupied) == 3:
        raise RuntimeError("R590 complete output namespace already exists")
    if not occupied and not stages:
        return
    if not stages:
        raise RuntimeError("R590 occupied namespace has no recognized stage; refusing recovery")
    for stage in stages:
        _recognized_stage(stage)
    if "result" in occupied:
        _recognized_result(occupied["result"])
    if "receipt" in occupied:
        _recognized_receipt(occupied["receipt"])
    if "evidence" in occupied:
        _recognized_evidence_dir(occupied["evidence"], allow_incomplete=False)
    recovery = Path(tempfile.mkdtemp(prefix=RECOVERY_PREFIX, dir=root))
    for name, path in occupied.items():
        os.replace(path, recovery / f"partial-{name}-{path.name}")
    for index, stage in enumerate(stages):
        os.replace(stage, recovery / f"stage-{index}-{stage.name}")
    _fsync_directory(recovery)
    _fsync_directory(root)
    raise RuntimeError(f"recovered incomplete R590 publication into {recovery}; rerun preflight")


def validate_complete_package(
    *, out: Path = OUT, receipt_path: Path = RECEIPT,
    evidence_dir: Path = EVIDENCE_DIR, replicates: int = BOOTSTRAPS,
) -> dict[str, object]:
    result = strict_load_json(out)
    evidence = strict_load_json(evidence_dir / EVIDENCE_FILE.name)
    receipt = strict_load_json(receipt_path)
    evidence_bytes = (evidence_dir / EVIDENCE_FILE.name).read_bytes()
    result_bytes = out.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != result["evidence_descriptor"]["sha256"]:
        raise RuntimeError("published evidence bytes differ from result descriptor")
    validate_result_against_evidence(result, evidence, replicates=replicates)
    validate_receipt(receipt, result_bytes, evidence_bytes, result)
    return result


def _runtime_hard_abort(capture_raw, exactness, label: str) -> None:
    # Called immediately after capture, before any candidate intervention.
    hard_abort_unretained_exactness(capture_raw, exactness, label)


def run_science() -> dict[str, object]:
    """Execute frozen R584 science into the new R590 package."""
    started = time.time()
    recover_stale_publication()
    rows = r584.load_authority()
    manifest = build_forward_call_manifest(rows)
    validate_forward_call_manifest(manifest, rows)  # before model load
    plan = execution_plan(rows)
    if not DRYRUN.is_file() or strict_load_json(DRYRUN) != plan:
        raise RuntimeError("saved R590 dry run is absent or stale")
    model, checkpoint = r584.facade.load_bilin18(
        device="cuda", dtype=r584.torch.float32, verify_weights_sha256=True
    )
    if checkpoint.weights_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash changed")

    fit_cache, fit_capture_raw, fit_calls, fit_exactness = r584.capture_split(model, rows, "FIT")
    _runtime_hard_abort(fit_capture_raw, fit_exactness, "FIT")
    fit_raw, fit_reports = {}, {}
    for site, component in SELECTION:
        name = f"mlp{site}_{component}"
        fit_raw[name], calls = r584.evaluate_component(
            model, rows, "FIT", fit_cache, site, component
        )
        fit_calls += calls
        fit_reports[name] = r584.score_candidate(
            fit_raw[name], cell_prefix=f"FIT:{name}", authority_rows=rows
        )
    provisional = next(
        (name for name in SELECTION_NAMES if fit_reports[name]["passed_without_nulls"]), None
    )
    selected = None
    fit_null_raw = None
    if provisional is not None:
        site_text, component = provisional[3:].split("_", 1)
        site = int(site_text)
        fit_null_raw = {}
        null_maps = r584.r582.deterministic_null_maps(rows, "FIT")
        fit_null_reports = {}
        for null_name in NULLS:
            key = f"{provisional}:null:{null_name}"
            fit_null_raw[key], calls = r584.evaluate_component(
                model, rows, "FIT", fit_cache, site, component,
                null_maps[null_name], null_name,
            )
            fit_calls += calls
            fit_null_reports[key] = r584.score_null(
                fit_raw[provisional], fit_null_raw[key], cell_prefix=f"FIT:{key}",
                real_report=fit_reports[provisional], null_name=null_name,
                authority_rows=rows,
            )
        if all(report["passed"] for report in fit_null_reports.values()):
            selected = provisional

    opened = ["FIT"]
    select_capture_raw = select_exactness = select_raw = select_null_raw = None
    select_calls = 0
    if selected is not None:
        select_cache, select_capture_raw, select_calls, select_exactness = r584.capture_split(
            model, rows, "SELECT"
        )
        _runtime_hard_abort(select_capture_raw, select_exactness, "SELECT")
        selected_site = int(selected[3:].split("_", 1)[0])
        selected_component = selected[3:].split("_", 1)[1]
        select_raw, select_reports = {}, {}
        for component in COMPONENTS:
            name = f"mlp{selected_site}_{component}"
            select_raw[name], calls = r584.evaluate_component(
                model, rows, "SELECT", select_cache, selected_site, component
            )
            select_calls += calls
            select_reports[name] = r584.score_candidate(
                select_raw[name], cell_prefix=f"SELECT:{name}",
                frozen_scales=fit_reports[selected]["fit_scales"], authority_rows=rows,
            )
        select_null_raw, select_null_reports = {}, {}
        null_maps = r584.r582.deterministic_null_maps(rows, "SELECT")
        for null_name in NULLS:
            key = f"{selected}:null:{null_name}"
            select_null_raw[key], calls = r584.evaluate_component(
                model, rows, "SELECT", select_cache, selected_site,
                selected_component, null_maps[null_name], null_name,
            )
            select_calls += calls
            select_null_reports[key] = r584.score_null(
                select_raw[selected], select_null_raw[key], cell_prefix=f"SELECT:{key}",
                real_report=select_reports[selected], null_name=null_name,
                authority_rows=rows,
            )
        opened.append("SELECT")

    total_calls = fit_calls + select_calls
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "rung": 590,
        "source_rung": 584,
        "result_path": str(OUT.relative_to(ROOT.parent.parent)),
        "receipt_path": str(RECEIPT.relative_to(ROOT.parent.parent)),
        "evaluated_splits": opened,
        "phase_support_census": frozen_phase_support_census(rows, opened),
        "phase_support_census_sha256": canonical_sha256(
            frozen_phase_support_census(rows, opened)
        ),
        "fit_exactness": fit_exactness,
        "select_exactness": select_exactness,
        "fit_capture_raw": fit_capture_raw,
        "select_capture_raw": select_capture_raw,
        "fit_raw": fit_raw,
        "fit_null_raw": fit_null_raw,
        "select_raw": select_raw,
        "select_null_raw": select_null_raw,
        "executed_forward_call_ids": expected_executed_call_ids(
            manifest, provisional, selected
        ),
    }
    derived = derive_scientific_summary(evidence)
    if total_calls != derived["model_forwards"]:
        raise RuntimeError(
            f"literal forward count mismatch: observed={total_calls}, derived={derived['model_forwards']}"
        )
    stage_root = create_stage_root()
    try:
        stage_root, result, _ = stage_package(
            evidence, checkpoint_sha256=checkpoint.weights_sha256,
            elapsed_seconds=time.time() - started, stage_root=stage_root,
        )
        publish_staged_package(stage_root)
    except BaseException:
        # Preserve recognizable stage bytes for conservative managed recovery.
        raise
    print(json.dumps({
        key: result[key] for key in result
        if key.startswith("pred_") or key in {
            "all_required_gates_pass", "selected_component", "model_forwards",
            "evaluated_splits", "decision", "next_step",
        }
    }, indent=2, sort_keys=True, allow_nan=False))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--execute-science"]:
        run_science()
        return
    if arguments:
        raise SystemExit("R590 accepts only --execute-science")
    if os.environ.get("BQLIB_DRYRUN") != "1":
        raise SystemExit("R590 model execution requires explicit --execute-science")
    print(json.dumps(run_dryrun(DRYRUN), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
