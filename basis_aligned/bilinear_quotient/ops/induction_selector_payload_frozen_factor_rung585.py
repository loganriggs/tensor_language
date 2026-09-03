#!/usr/bin/env python3
"""R585 prospective selector x payload factor intervention.

The default dry run is CPU-only and model-free.  The scientific path is an
explicit opt-in and uses the hash-pinned R459 factor computation plus the
canonical equality contraction.  It never opens FINAL_TEST or OOD.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT.parent / "polynomial_causal"
OPS = ROOT / "ops"
SCRIPT = Path(__file__).resolve()
TEST = SCRIPT.with_name("test_induction_selector_payload_frozen_factor_rung585.py")
OUT = ROOT / "induction_selector_payload_frozen_factor_rung585_results.json"
RECEIPT = ROOT / "induction_selector_payload_frozen_factor_rung585_receipt.json"
DRYRUN = ROOT / "induction_selector_payload_frozen_factor_rung585_dryrun.json"
EVIDENCE_DIR = ROOT / "induction_selector_payload_frozen_factor_rung585_evidence"
STAGE_PREFIX = ".induction_selector_payload_frozen_factor_rung585_stage-"
RECOVERY_PREFIX = ".induction_selector_payload_frozen_factor_rung585_recovered-"

ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
AMENDMENT = POLY / "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_REPLACEMENT_AMENDMENT.md"
MANIFEST = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
DEPENDENCY_LOCK = ROOT / "induction_selector_payload_frozen_factor_rung585_dependency_lock.json"
R459_FACTOR = OPS / "equality_term_score_payload_rung459.py"
CANONICAL_TERM = OPS / "equality_term_subset_factorial_stage1.py"
INDUCTION = POLY / "circuit_induction_tensor.py"
FACADE = POLY / "bilin18_observed_model_facade.py"
R586_RESULT = ROOT / "induction_selector_payload_native_capability_rung586_results.json"
R586_RECEIPT = ROOT / "induction_selector_payload_native_capability_rung586_receipt.json"
R587_AUDIT = ROOT / "induction_selector_payload_native_capability_audit_rung587.json"
IMPLEMENTATION_REVIEW = POLY / (
    "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_IMPLEMENTATION_PREEXECUTION_REVIEW.md"
)
IMPLEMENTATION_ADVERSARIAL_TEST = OPS / (
    "test_induction_selector_payload_frozen_factor_rung585_implementation_adversarial.py"
)

AUTHORITY_HASHES = {
    ROWS: "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
    AMENDMENT: "98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf",
    MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    DEPENDENCY_LOCK: "908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7",
    R459_FACTOR: "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
    CANONICAL_TERM: "3caa753cd856ec87899936fe71137ce28e893f86433558f40a815afff61824af",
    INDUCTION: "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    R586_RESULT: "14e7414bc7cf6b4a6a221079ac378752602b021b8b411124149dcc2c311666b8",
    R586_RECEIPT: "afd7533b1838b7d230858696a059f9c3a5903e75f031aa0c86f175f4bc0d9384",
    R587_AUDIT: "72f0261fe32aa3d048c442ea1c08af932af6a368894610833e79aaaabf98bfe9",
    IMPLEMENTATION_REVIEW: "9bf8ae3c89d7c504bfdd42694771ef44bb87883429060d16335f0a1266d75a30",
    IMPLEMENTATION_ADVERSARIAL_TEST:
        "2567c3c5633575c2f4f8369328071025037b7c6f6c8a359f7870859b787a12e2",
}

SPLITS = ("FIT", "SELECT")
FORBIDDEN_SPLITS = ("FINAL_TEST", "OOD")
ARMS = ("score", "payload", "joint")
SITES = ((5, 5), (7, 3), (8, 3), (8, 4))
ROLES = ("A", "C")
TERM_NAMES = tuple(f"L{site}H{head}" for site, head in SITES)
BATCH = 32
TOLERANCE = 1e-5
VOCABULARY_SIZE = 50_304
BOOTSTRAPS = 2_000
EXPECTED_PHASE_PRICE = {"FIT": 459, "SELECT": 231}
EXPECTED_TOTAL_PRICE = 690
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"

RESULT_SCHEMA = "induction_selector_payload_frozen_factor_rung585_result_v2"
RECEIPT_SCHEMA = "induction_selector_payload_frozen_factor_rung585_receipt_v2"
DRYRUN_SCHEMA = "induction_selector_payload_frozen_factor_rung585_dryrun_v2"
EVIDENCE_SCHEMA = "induction_selector_payload_frozen_factor_rung585_evidence_v2"
PROSPECTIVE_STATUS = "prospective_thresholds_frozen_before_r585_outcomes"

# These names make the preregistered scientific questions visible to the
# repository's static pre-queue gate.  The detailed numerical bars and nulls
# remain authoritative in the hash-pinned replacement amendment.
REGISTERED_PREDICATES = {
    "pred_a_exact_factor_instrument":
        "the canonical equality-term reconstruction and replay instrument are exact",
    "pred_b_complete_joint_capacity":
        "the complete four-site joint term transfers the registered donor computation",
    "pred_c_selector_payload_factorization":
        "score-only and payload-only swaps follow their opposing registered predictions",
    "pred_d_active_control_selectivity":
        "the intervention remains selective on unrelated rows where its tensor delta is active",
}
EVIDENCE_DESCRIPTOR_FIELDS = {
    "path": str, "sha256": str, "bytes": int, "dtype": str,
    "shape": list, "row_order_sha256": str,
}
EXPECTED_OPERATION_COUNTS = {"FIT": 13_824, "SELECT": 6_912}
EXPECTED_OPERATION_SHA256 = "82169667d6f658b993f882b7b9951e07ae93149e5d5138fce548f6205e88cc5e"
HELD_ARRAY_SHAPES = {
    "native_e.npy": [2_592, 4, 2],
    "native_u.npy": [2_592, 4, 2, 1_152],
    "canonical_term.npy": [2_592, 4, 1_152],
    "native_head_output.npy": [2_592, 4, 1_152],
    "non_equality_remainder.npy": [2_592, 4, 1_152],
    "live_removed.npy": [16_848, 4, 1_152],
    "hook_delta.npy": [16_848, 4, 1_152],
}
HELD_JSONL_COUNTS = {
    "endpoint_measurements.jsonl": 2_592,
    "directed_arm_measurements.jsonl": 16_848,
    "factor_exactness.jsonl": 10_368,
}

RESULT_FIELD_TYPES = {
    "schema": str,
    "rung": int,
    "stage": str,
    "evidence_level": str,
    "threshold_status": str,
    "instrument_passes": bool,
    "terminal": str,
    "failed_clauses": list,
    "failure_classes": dict,
    "split_scores": dict,
    "raw_evidence": dict,
    "evidence_files": list,
    "model_forwards": int,
    "model_backwards": int,
    "model_weights_updated": bool,
    "checkpoint_weights_sha256": str,
    "implementation_sha256": str,
    "test_sha256": str,
    "source_sha256": dict,
    "dependency_lock_sha256": str,
    "evaluated_splits": list,
    "forbidden_splits_opened": list,
    "elapsed_seconds": float,
    "next_step": str,
}

RECEIPT_FIELD_TYPES = {
    "schema": str,
    "result_path": str,
    "result_sha256": str,
    "implementation_sha256": str,
    "test_sha256": str,
    "source_sha256": dict,
    "dependency_lock_sha256": str,
    "checkpoint_weights_sha256": str,
    "terminal": str,
    "model_forwards": int,
    "model_backwards": int,
    "model_weights_updated": bool,
    "evaluated_splits": list,
    "forbidden_splits_opened": list,
    "evidence_files": list,
    "next_step": str,
}

TERMINALS = {
    "invalid_instrument",
    "native_denominator_or_scale_null",
    "factor_capacity_null",
    "factorization_not_identified",
    "insufficient_active_controls",
    "broad_contextual_equality_write",
    "select_invalid_instrument",
    "select_native_denominator_or_scale_null",
    "select_factor_capacity_null",
    "select_factorization_not_identified",
    "select_insufficient_active_controls",
    "select_broad_contextual_equality_write",
    "held_operational_selector_payload_factorization",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def load_manifest():
    return load_module(MANIFEST, "r585_frozen_manifest")


def verify_authorities(*, parse_dependency: bool = True) -> dict[str, str]:
    observed = {}
    for path, expected in AUTHORITY_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen authority mismatch: {path}")
        observed[str(path)] = expected
    if parse_dependency:
        manifest = load_manifest()
        lock = strict_load_json(DEPENDENCY_LOCK)
        supplied = {
            str(lock["r586_result_path"]): AUTHORITY_HASHES[R586_RESULT],
            str(lock["r586_receipt_path"]): AUTHORITY_HASHES[R586_RECEIPT],
            str(lock["r587_audit_path"]): AUTHORITY_HASHES[R587_AUDIT],
        }
        decision = manifest.validate_dependency_lock(lock, supplied)
        if not decision["runnable"]:
            raise RuntimeError("not_executed_upstream_dependency")
        result = strict_load_json(R586_RESULT)
        receipt = strict_load_json(R586_RECEIPT)
        audit = strict_load_json(R587_AUDIT)
        if result.get("verdict") != lock["r586_verdict"] or (
            receipt.get("verdict") != lock["r586_verdict"]
            or receipt.get("result_sha256") != lock["r586_result_sha256"]
        ):
            raise RuntimeError("R586 parsed dependency fields disagree with lock")
        if audit.get("audit_verdict") != lock["r587_audit_verdict"] or (
            audit.get("source_result_sha256") != lock["r586_result_sha256"]
            or audit.get("source_receipt_sha256") != lock["r586_receipt_sha256"]
            or audit.get("independently_recomputed_scientific_verdict")
            != lock["r586_verdict"]
        ):
            raise RuntimeError("R587 parsed dependency fields disagree with lock")
    return observed


def strict_load_json(path: Path) -> Any:
    def reject(value: str):
        raise ValueError(f"non-standard JSON constant {value} in {path}")
    return json.loads(path.read_text(), parse_constant=reject)


def require_finite_json(value: object, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"nonfinite value at {path}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            require_finite_json(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"non-string JSON key at {path}")
            require_finite_json(child, f"{path}.{key}")
        return
    raise TypeError(f"non-JSON type {type(value).__name__} at {path}")


def _validate_fields(document: Mapping[str, object], fields: Mapping[str, type]) -> None:
    if set(document) != set(fields):
        raise ValueError(
            f"field mismatch: missing={sorted(set(fields)-set(document))}, "
            f"extra={sorted(set(document)-set(fields))}"
        )
    for field, expected in fields.items():
        if type(document[field]) is not expected:
            raise TypeError(f"{field} must be {expected.__name__}")
    require_finite_json(dict(document))


def _finite_array(array: np.ndarray, label: str) -> None:
    flat = array.reshape(-1)
    for start in range(0, flat.size, 1_000_000):
        if not bool(np.isfinite(flat[start:start + 1_000_000]).all()):
            raise ValueError(f"nonfinite evidence array: {label}")


def _strict_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-standard JSON constant {value}")
                    ),
                )
            except (TypeError, ValueError) as jsonl_error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from jsonl_error
            if type(row) is not dict:
                raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
            require_finite_json(row, f"{path}:{line_number}")
            rows.append(row)
    return rows


def _validate_held_evidence(
    result: Mapping[str, object],
    artifact_path_resolver,
) -> None:
    raw = result["raw_evidence"]
    required_raw = {
        "schema", "endpoint_count", "directed_arm_record_count",
        "endpoint_site_role_operation_counts", "endpoint_site_role_operation_sha256",
        "realized_endpoint_site_role_operations", "instrument_maxima",
    }
    if type(raw) is not dict or not required_raw <= set(raw):
        raise ValueError("held result lacks complete raw evidence")
    if raw["schema"] != EVIDENCE_SCHEMA or raw["endpoint_count"] != 2_592 or (
        raw["directed_arm_record_count"] != 16_848
    ):
        raise ValueError("held evidence census changed")
    if raw["endpoint_site_role_operation_counts"] != EXPECTED_OPERATION_COUNTS or (
        raw["endpoint_site_role_operation_sha256"] != EXPECTED_OPERATION_SHA256
    ):
        raise ValueError("held operation authority changed")
    realized_operations = raw["realized_endpoint_site_role_operations"]
    if set(realized_operations) != set(SPLITS) or any(
        realized_operations[split] != {
            "count": EXPECTED_OPERATION_COUNTS[split],
            "sha256": content_sha256([
                row for row in build_endpoint_site_role_operations(
                    build_execution_authority()["endpoints"]
                ) if row["split"] == split
            ]),
        }
        for split in SPLITS
    ):
        raise ValueError("held realized operation census changed")
    maxima = raw["instrument_maxima"]
    required_maxima = {
        "native_attention_reconstruction_max_abs",
        "equality_factor_max_abs",
        "equality_plus_independent_remainder_max_abs",
        "replay_native_logit_max_abs",
        "padding_tripwire_active_lengths",
    }
    if type(maxima) is not dict or set(maxima) != required_maxima:
        raise ValueError("held instrument maxima are incomplete")
    if any(float(maxima[key]) > TOLERANCE for key in required_maxima if key != "padding_tripwire_active_lengths"):
        raise ValueError("held instrument maximum exceeds tolerance")
    if sorted(maxima["padding_tripwire_active_lengths"]) != [19, 20, 21, 22, 27, 28, 29]:
        raise ValueError("held padding tripwire census changed")

    descriptors = result["evidence_files"]
    expected_names = set(HELD_ARRAY_SHAPES) | set(HELD_JSONL_COUNTS)
    if len(descriptors) != len(expected_names):
        raise ValueError("held evidence descriptor census changed")
    by_name = {}
    logical_root = EVIDENCE_DIR.relative_to(ROOT.parent.parent)
    for descriptor in descriptors:
        logical = Path(descriptor["path"])
        if logical.parent != logical_root or logical.name in by_name:
            raise ValueError("held evidence path is noncanonical or duplicated")
        by_name[logical.name] = descriptor
    if set(by_name) != expected_names:
        raise ValueError("held evidence file set changed")

    actual_paths = {
        name: artifact_path_resolver(ROOT.parent.parent / descriptor["path"])
        for name, descriptor in by_name.items()
    }
    arrays = {}
    for name, expected_shape in HELD_ARRAY_SHAPES.items():
        descriptor = by_name[name]
        if descriptor["shape"] != expected_shape or descriptor["dtype"] != "<f4":
            raise ValueError(f"held array schema changed: {name}")
        array = np.load(actual_paths[name], mmap_mode="r", allow_pickle=False)
        if list(array.shape) != expected_shape or array.dtype.str != "<f4":
            raise ValueError(f"held array bytes disagree with descriptor: {name}")
        _finite_array(array, name)
        arrays[name] = array

    json_rows = {}
    for name, expected_count in HELD_JSONL_COUNTS.items():
        descriptor = by_name[name]
        if descriptor["shape"] != [expected_count] or descriptor["dtype"] != "jsonl":
            raise ValueError(f"held JSONL schema changed: {name}")
        rows = _strict_jsonl(actual_paths[name])
        if len(rows) != expected_count:
            raise ValueError(f"held JSONL row census changed: {name}")
        json_rows[name] = rows

    endpoints = json_rows["endpoint_measurements.jsonl"]
    endpoint_order = [str(row["endpoint_id"]) for row in endpoints]
    if endpoints != sorted(endpoints, key=lambda row: (row["split"], row["endpoint_id"])) or (
        len(endpoint_order) != len(set(endpoint_order))
    ):
        raise ValueError("endpoint evidence order or membership changed")
    directed = json_rows["directed_arm_measurements.jsonl"]
    directed_order = [[row["directed_id"], row["arm"]] for row in directed]
    if directed != sorted(directed, key=lambda row: (row["split"], row["directed_id"], row["arm"])) or (
        len({tuple(row) for row in directed_order}) != len(directed_order)
    ):
        raise ValueError("directed evidence order or membership changed")
    factor_rows = json_rows["factor_exactness.jsonl"]
    factor_order = [[row["endpoint_id"], row["site"]] for row in factor_rows]
    if factor_rows != sorted(factor_rows, key=lambda row: (row["split"], row["endpoint_id"], row["site"])) or (
        len({tuple(row) for row in factor_order}) != len(factor_order)
    ):
        raise ValueError("factor exactness order or membership changed")
    if any(
        float(row[key]) > TOLERANCE
        for row in factor_rows
        for key in ("equality_factor_max_abs", "equality_plus_independent_remainder_max_abs")
    ):
        raise ValueError("factor exactness evidence exceeds tolerance")

    endpoint_hash = content_sha256(endpoint_order)
    directed_hash = content_sha256(directed_order)
    factor_hash = content_sha256(factor_order)
    for name in HELD_ARRAY_SHAPES:
        expected_hash = directed_hash if name in ("live_removed.npy", "hook_delta.npy") else endpoint_hash
        if by_name[name]["row_order_sha256"] != expected_hash:
            raise ValueError(f"array row order binding changed: {name}")
    if by_name["endpoint_measurements.jsonl"]["row_order_sha256"] != endpoint_hash or (
        by_name["directed_arm_measurements.jsonl"]["row_order_sha256"] != directed_hash
    ) or by_name["factor_exactness.jsonl"]["row_order_sha256"] != factor_hash:
        raise ValueError("JSONL row order binding changed")

    for start in range(0, 2_592, 64):
        stop = min(start + 64, 2_592)
        equality = np.sum(
            arrays["native_e.npy"][start:stop, :, :, None]
            * arrays["native_u.npy"][start:stop],
            axis=2,
        )
        if float(np.max(np.abs(equality - arrays["canonical_term.npy"][start:stop]))) > 5e-5:
            raise ValueError("saved equality factors do not reconstruct canonical term")
        reconstructed = (
            arrays["canonical_term.npy"][start:stop]
            + arrays["non_equality_remainder.npy"][start:stop]
        )
        if float(np.max(np.abs(reconstructed - arrays["native_head_output.npy"][start:stop]))) > 5e-5:
            raise ValueError("saved independent remainder does not reconstruct head output")


def validate_result(result: Mapping[str, object], *, artifact_path_resolver=None) -> None:
    _validate_fields(result, RESULT_FIELD_TYPES)
    if result["schema"] != RESULT_SCHEMA or result["rung"] != 585:
        raise ValueError("wrong result identity")
    if result["threshold_status"] != PROSPECTIVE_STATUS:
        raise ValueError("thresholds are not labeled prospective")
    if result["terminal"] not in TERMINALS:
        raise ValueError("unknown terminal")
    if result["evaluated_splits"] not in (["FIT"], ["FIT", "SELECT"]):
        raise ValueError("invalid FIT-first split opening")
    if result["forbidden_splits_opened"] != []:
        raise ValueError("forbidden split opened")
    expected_price = 459 if result["evaluated_splits"] == ["FIT"] else 690
    if not 0 <= result["model_forwards"] <= expected_price:
        raise ValueError("model-forward envelope exceeded")
    if "invalid_instrument" not in result["terminal"] and result["model_forwards"] != expected_price:
        raise ValueError("completed scientific phase did not use exact frozen price")
    if result["model_backwards"] != 0 or result["model_weights_updated"] is not False:
        raise ValueError("mutation envelope violated")
    if result["checkpoint_weights_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash mismatch")
    if result["dependency_lock_sha256"] != AUTHORITY_HASHES[DEPENDENCY_LOCK]:
        raise ValueError("wrong dependency lock")
    if result["implementation_sha256"] != sha256(SCRIPT):
        raise ValueError("result does not bind implementation")
    if result["test_sha256"] != sha256(TEST):
        raise ValueError("result does not bind owner test")
    if result["source_sha256"] != {str(path): digest for path, digest in AUTHORITY_HASHES.items()}:
        raise ValueError("result source provenance mismatch")
    resolver = artifact_path_resolver or (lambda path: path)
    for descriptor in result["evidence_files"]:
        if type(descriptor) is not dict:
            raise TypeError("evidence descriptor must be a dict")
        _validate_fields(descriptor, EVIDENCE_DESCRIPTOR_FIELDS)
        logical_path = ROOT.parent.parent / descriptor["path"]
        path = resolver(logical_path)
        if not path.is_file() or sha256(path) != descriptor["sha256"] or (
            path.stat().st_size != descriptor["bytes"]
        ):
            raise ValueError(f"evidence file binding failed: {path}")
        if descriptor["dtype"] == "jsonl":
            if descriptor["shape"] != [len(_strict_jsonl(path))]:
                raise ValueError(f"JSONL descriptor shape changed: {path}")
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != descriptor["shape"] or array.dtype.str != descriptor["dtype"]:
                raise ValueError(f"array descriptor disagrees with bytes: {path}")
            _finite_array(array, str(path))
    expected_terminal = terminal_from_failures(result["evaluated_splits"], result["failure_classes"])
    if result["terminal"] != expected_terminal:
        raise ValueError("terminal disagrees with deterministic precedence")
    expected_failed = [
        clause for key in sorted(result["failure_classes"])
        for clause in result["failure_classes"][key]
    ]
    if result["failed_clauses"] != expected_failed:
        raise ValueError("failed-clause list disagrees with failure classes")
    expected_instrument = not any(
        result["failure_classes"].get(key)
        for key in ("invalid_instrument", "select_invalid_instrument")
    )
    if result["instrument_passes"] is not expected_instrument:
        raise ValueError("instrument flag disagrees with instrument failures")
    held = result["terminal"] == "held_operational_selector_payload_factorization"
    if held != bool(result["instrument_passes"] and not result["failed_clauses"]):
        raise ValueError("held result disagrees with failed clauses")
    if held and result["evaluated_splits"] != ["FIT", "SELECT"]:
        raise ValueError("held result did not open SELECT")
    if held:
        _validate_held_evidence(result, resolver)
        execution = build_execution_authority()
        for split in SPLITS:
            report = result["split_scores"].get(split)
            if type(report) is not dict:
                raise ValueError(f"held split score missing: {split}")
            realized = validate_realized_bootstraps(
                report, split, execution["manifests"]
            )
            if report.get("bootstrap_realization") != realized:
                raise ValueError(f"held bootstrap realization metadata changed: {split}")


def validate_receipt(
    receipt: Mapping[str, object], result: Mapping[str, object], *, result_file: Path | None = None
) -> None:
    _validate_fields(receipt, RECEIPT_FIELD_TYPES)
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise ValueError("wrong receipt schema")
    if receipt["result_path"] != str(OUT.relative_to(ROOT.parent.parent)):
        raise ValueError("wrong canonical result_path")
    bindings = {
        "implementation_sha256": result["implementation_sha256"],
        "test_sha256": result["test_sha256"],
        "source_sha256": result["source_sha256"],
        "dependency_lock_sha256": result["dependency_lock_sha256"],
        "checkpoint_weights_sha256": result["checkpoint_weights_sha256"],
        "terminal": result["terminal"],
        "model_forwards": result["model_forwards"],
        "model_backwards": result["model_backwards"],
        "model_weights_updated": result["model_weights_updated"],
        "evaluated_splits": result["evaluated_splits"],
        "forbidden_splits_opened": result["forbidden_splits_opened"],
        "evidence_files": result["evidence_files"],
        "next_step": result["next_step"],
    }
    for field, expected in bindings.items():
        if receipt[field] != expected:
            raise ValueError(f"receipt/result mismatch: {field}")
    if receipt["result_sha256"] != content_sha256(result):
        raise ValueError("receipt result digest mismatch")
    if result_file is not None:
        encoded = result_file.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != receipt["result_sha256"]:
            raise ValueError("receipt does not bind exact result bytes")
        if json.loads(encoded, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value}")
        )) != result:
            raise ValueError("staged result bytes disagree with validated result")


def terminal_from_failures(
    evaluated_splits: Sequence[str], failure_classes: Mapping[str, Sequence[str]]
) -> str:
    precedence = (
        "invalid_instrument",
        "native_denominator_or_scale_null",
        "factor_capacity_null",
        "factorization_not_identified",
        "insufficient_active_controls",
        "broad_contextual_equality_write",
    )
    for label in precedence:
        if failure_classes.get(label):
            return label
    if list(evaluated_splits) == ["FIT", "SELECT"]:
        for label in precedence:
            key = "select_" + label
            if failure_classes.get(key):
                return key
        return "held_operational_selector_payload_factorization"
    raise ValueError("FIT passed but SELECT was not evaluated")


def _quantile(values: np.ndarray, q: float, method: str) -> float:
    return float(np.quantile(values.astype(np.float64), q, method=method))


def bootstrap_mean(
    values_by_group: Mapping[str, Sequence[float]], cell_id: str, *, replicates: int = BOOTSTRAPS
) -> dict[str, object]:
    manifest = load_manifest()
    groups = tuple(sorted(values_by_group))
    if not groups or any(not values_by_group[group] for group in groups):
        raise ValueError("empty bootstrap group")
    flat = [float(value) for group in groups for value in values_by_group[group]]
    if not all(math.isfinite(value) for value in flat):
        raise ValueError("nonfinite bootstrap input")
    statistics = np.empty(replicates, dtype=np.float64)
    draws = np.empty((replicates, len(groups)), dtype=">u2")
    for b in range(replicates):
        total = 0.0
        count = 0
        for k in range(len(groups)):
            index = manifest.bootstrap_draw_index(cell_id, b, k, len(groups))
            draws[b, k] = index
            selected = values_by_group[groups[index]]
            total += math.fsum(float(value) for value in selected)
            count += len(selected)
        statistics[b] = total / count
    return {
        "cell_id": cell_id,
        "group_ids": list(groups),
        "replicates": replicates,
        "point_mean": float(np.mean(np.asarray(flat, dtype=np.float64))),
        "lower95": _quantile(statistics, 0.025, "lower"),
        "upper95": _quantile(statistics, 0.975, "higher"),
        "draw_sha256": hashlib.sha256(draws.tobytes(order="C")).hexdigest(),
        "statistic_sha256": hashlib.sha256(statistics.astype(">f8").tobytes(order="C")).hexdigest(),
    }


def recovery_summary(
    rows: Sequence[Mapping[str, object]], cell_id_prefix: str, *, replicates: int = BOOTSTRAPS
) -> dict[str, object]:
    if not rows:
        raise ValueError("empty recovery cell")
    numerator = {str(row["group_id"]): [float(row["n"])] for row in rows}
    denominator = {str(row["group_id"]): [float(row["d"])] for row in rows}
    if len(numerator) != len(rows) or len(denominator) != len(rows):
        raise ValueError("recovery cell is not one row per semantic group")
    n_values = np.asarray([float(row["n"]) for row in rows], dtype=np.float64)
    d_values = np.asarray([float(row["d"]) for row in rows], dtype=np.float64)
    n_mean, d_mean = float(n_values.mean()), float(d_values.mean())
    n_median, d_median = float(np.median(n_values)), float(np.median(d_values))
    n_boot = bootstrap_mean(numerator, cell_id_prefix + "|numerator_mean", replicates=replicates)
    d_boot = bootstrap_mean(denominator, cell_id_prefix + "|denominator_mean", replicates=replicates)
    denominator_valid = bool(d_mean > 0 and d_median > 0 and d_boot["lower95"] > 0)
    return {
        "mean_numerator": n_mean,
        "median_numerator": n_median,
        "mean_denominator": d_mean,
        "median_denominator": d_median,
        "mean_recovery": n_mean / d_mean if denominator_valid else None,
        "median_recovery": n_median / d_median if denominator_valid else None,
        "positive_numerator_fraction": float(np.mean(n_values > 0)),
        "numerator_bootstrap": n_boot,
        "denominator_bootstrap": d_boot,
        "denominator_valid": denominator_valid,
    }


def donor_ce_summary(
    rows: Sequence[Mapping[str, object]], cell_id: str, *, replicates: int = BOOTSTRAPS
) -> dict[str, object]:
    values = {str(row["group_id"]): [float(row["q"])] for row in rows}
    return bootstrap_mean(values, cell_id + "|donor_ce_mean", replicates=replicates)


def realized_bootstrap_ids(value: object) -> list[str]:
    """Recover bootstrap identities from the actual nested score objects."""
    output = []
    if type(value) is dict:
        if {
            "cell_id", "group_ids", "replicates", "draw_sha256", "statistic_sha256"
        } <= set(value):
            output.append(str(value["cell_id"]))
        for child in value.values():
            output.extend(realized_bootstrap_ids(child))
    elif type(value) is list:
        for child in value:
            output.extend(realized_bootstrap_ids(child))
    return output


def validate_realized_bootstraps(
    reports: Mapping[str, object], split: str, manifests: Mapping[str, object]
) -> dict[str, object]:
    expected_rows = load_manifest().expected_bootstrap_cells(manifests)
    expected = [row["cell_id"] for row in expected_rows if row["cell_id"].startswith(split + "|")]
    realized = sorted(realized_bootstrap_ids(dict(reports)))
    if realized != expected or len(realized) != len(set(realized)):
        raise RuntimeError(f"{split} realized bootstrap census mismatch")
    return {
        "count": len(realized),
        "cell_ids_sha256": content_sha256(realized),
    }


def make_result_fixture(terminal: str) -> dict[str, object]:
    if terminal not in TERMINALS:
        raise ValueError("unknown planted terminal")
    held = terminal == "held_operational_selector_payload_factorization"
    evaluated = ["FIT", "SELECT"] if held or terminal.startswith("select_") else ["FIT"]
    failures = [] if held else [f"planted:{terminal}"]
    classes = {name: [] for name in TERMINALS if name != "held_operational_selector_payload_factorization"}
    if not held:
        classes[terminal] = list(failures)
    result = {
        "schema": RESULT_SCHEMA,
        "rung": 585,
        "stage": "prospective_frozen_selector_payload_factor_intervention",
        "evidence_level": "prospective_identification_screen",
        "threshold_status": PROSPECTIVE_STATUS,
        "instrument_passes": "invalid_instrument" not in terminal,
        "terminal": terminal,
        "failed_clauses": failures,
        "failure_classes": classes,
        "split_scores": {split: {"fixture": True} for split in evaluated},
        "raw_evidence": {"fixture": True, "row_count": 0},
        "evidence_files": [],
        "model_forwards": 690 if evaluated == ["FIT", "SELECT"] else 459,
        "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "source_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "dependency_lock_sha256": AUTHORITY_HASHES[DEPENDENCY_LOCK],
        "evaluated_splits": evaluated,
        "forbidden_splits_opened": [],
        "elapsed_seconds": 0.0,
        "next_step": (
            "independent_cpu_audit_then_translation_removal_and_ood_preregistration"
            if held else "preserve_terminal_and_do_not_search_sites_or_thresholds"
        ),
    }
    if not held:
        validate_result(result)
    return result


def make_receipt_fixture(result: Mapping[str, object]) -> dict[str, object]:
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "result_path": str(OUT.relative_to(ROOT.parent.parent)),
        "result_sha256": content_sha256(result),
        "implementation_sha256": result["implementation_sha256"],
        "test_sha256": result["test_sha256"],
        "source_sha256": result["source_sha256"],
        "dependency_lock_sha256": result["dependency_lock_sha256"],
        "checkpoint_weights_sha256": result["checkpoint_weights_sha256"],
        "terminal": result["terminal"],
        "model_forwards": result["model_forwards"],
        "model_backwards": result["model_backwards"],
        "model_weights_updated": result["model_weights_updated"],
        "evaluated_splits": result["evaluated_splits"],
        "forbidden_splits_opened": result["forbidden_splits_opened"],
        "evidence_files": result["evidence_files"],
        "next_step": result["next_step"],
    }
    validate_receipt(receipt, result)
    return receipt


def build_endpoint_site_role_operations(
    endpoints: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Materialize the exact ordered native-factor operation authority."""
    operations = [
        {
            "split": str(endpoint["split"]),
            "endpoint_id": str(endpoint["endpoint_id"]),
            "site": site,
            "role": role,
        }
        for endpoint in endpoints
        for site in TERM_NAMES
        for role in ROLES
    ]
    operations.sort(key=lambda row: (
        row["split"], row["endpoint_id"], row["site"], row["role"]
    ))
    counts = {
        split: sum(row["split"] == split for row in operations) for split in SPLITS
    }
    if counts != EXPECTED_OPERATION_COUNTS or content_sha256(operations) != EXPECTED_OPERATION_SHA256:
        raise RuntimeError("endpoint-site-role operation authority changed")
    return operations


def validate_realized_operations(
    expected: Sequence[Mapping[str, object]],
    realized: Sequence[Mapping[str, object]],
    split: str,
) -> dict[str, object]:
    expected_split = [dict(row) for row in expected if row["split"] == split]
    realized_sorted = sorted(
        [dict(row) for row in realized],
        key=lambda row: (row["split"], row["endpoint_id"], row["site"], row["role"]),
    )
    if realized_sorted != expected_split:
        raise RuntimeError(f"{split} endpoint-site-role operation census mismatch")
    return {
        "count": len(realized_sorted),
        "sha256": content_sha256(realized_sorted),
    }


def build_execution_authority() -> dict[str, object]:
    """Join the hash-pinned manifest to full semantic R578 coordinates."""
    verify_authorities()
    manifest = load_manifest()
    authority = manifest.build_authority_manifest()
    manifests = manifest.build_cell_manifests(authority)
    rows = [
        row for row in strict_load_json(ROWS)["rows"]
        if row["family_id"] in manifest.INCLUDED_FAMILIES and row["split"] in SPLITS
    ]
    raw_by_id = {str(row["row_id"]): row for row in rows}
    if len(raw_by_id) != 2_808:
        raise RuntimeError("R578 included-row census changed")

    endpoint_specs: dict[tuple[str, str], dict[str, object]] = {}
    directions = []
    manifest_directions = {row["directed_id"]: row for row in authority["directions"]}
    for row in rows:
        for prefix in ("base", "donor"):
            ids = list(map(int, row[prefix + "_ids"]))
            endpoint_id = manifest._sequence_id(ids)
            structure = row[prefix + "_structure"]
            # source_positions/payload_positions are stored in semantic A,C
            # order even when pair_order changes their physical prompt order.
            if set(structure["pair_order"]) != {"A", "C", "N"}:
                raise RuntimeError("semantic role inventory changed")
            sources = list(map(int, structure["source_positions"]))
            payloads = list(map(int, structure["payload_positions"]))
            query = int(structure["query_position"])
            if len(sources) != 2 or len(payloads) != 2 or any(
                payload != source + 1 for source, payload in zip(sources, payloads)
            ) or query != len(ids) - 1:
                raise RuntimeError("semantic source/payload/query coordinates changed")
            spec = {
                "split": str(row["split"]),
                "endpoint_id": endpoint_id,
                "token_ids": ids,
                "length": len(ids),
                "final_position": query,
                "source_positions": sources,
                "payload_positions": payloads,
                "condition": f"s{row[prefix + '_selector']}p{row[prefix + '_payload_assignment']}",
                "answer_id": int(row[prefix + "_answer_id"]),
                "other_answer_id": int(row[prefix + "_other_answer_id"]),
            }
            key = (spec["split"], endpoint_id)
            if key in endpoint_specs and endpoint_specs[key] != spec:
                raise RuntimeError("cross-arm semantic endpoint metadata mismatch")
            endpoint_specs[key] = spec
        for direction in ("base_to_donor", "donor_to_base"):
            recipient, donor = ("base", "donor") if direction == "base_to_donor" else ("donor", "base")
            directed_id = f"{row['row_id']}:{direction}"
            frozen = manifest_directions[directed_id]
            item = {
                **{key: frozen[key] for key in (
                    "split", "directed_id", "row_id", "group_id", "family", "variant",
                    "direction", "recipient_condition", "recipient_is_coherent",
                    "donor_is_coherent", "donor_coherence_sign", "answer_changes", "control_kind",
                )},
                "recipient_endpoint_id": manifest._sequence_id(row[recipient + "_ids"]),
                "donor_endpoint_id": manifest._sequence_id(row[donor + "_ids"]),
                "recipient_answer_id": int(row[recipient + "_answer_id"]),
                "donor_answer_id": int(row[donor + "_answer_id"]),
                "recipient_other_answer_id": int(row[recipient + "_other_answer_id"]),
                "donor_other_answer_id": int(row[donor + "_other_answer_id"]),
            }
            if item["recipient_endpoint_id"] != frozen["recipient_endpoint_id"] or (
                item["donor_endpoint_id"] != frozen["donor_endpoint_id"]
            ):
                raise RuntimeError("direction-to-endpoint mapping changed")
            directions.append(item)
    endpoint_list = sorted(endpoint_specs.values(), key=lambda row: (row["split"], row["endpoint_id"]))
    directions.sort(key=lambda row: (row["split"], row["directed_id"]))
    operations = build_endpoint_site_role_operations(endpoint_list)
    # Canonical serialization and census hashes are model-free authorities for
    # every later batch and evidence record.
    return {
        "endpoints": endpoint_list,
        "directions": directions,
        "endpoint_site_role_operations": operations,
        "manifests": manifests,
        "control_scale_lookup": manifest.build_control_scale_lookup(manifests),
        "bootstrap_cells": manifest.expected_bootstrap_cells(manifests),
        "endpoint_manifest_sha256": content_sha256(endpoint_list),
        "direction_manifest_sha256": content_sha256(directions),
        "endpoint_site_role_operation_sha256": content_sha256(operations),
        "target_cell_ids_sha256": content_sha256(
            [row["cell_id"] for row in manifests["target_cells"]]
        ),
        "control_cell_ids_sha256": content_sha256(
            [row["cell_id"] for row in manifests["control_cells"]]
        ),
        "coverage_key_sha256": content_sha256(manifests["coverage_keys"]),
        "structural_identity_sha256": content_sha256(manifests["structural_identities"]),
        "bootstrap_cell_ids_sha256": content_sha256(
            [row["cell_id"] for row in manifest.expected_bootstrap_cells(manifests)]
        ),
        "control_scale_lookup_sha256": content_sha256(
            manifest.build_control_scale_lookup(manifests)
        ),
    }


def make_batch_schedule(
    records: Sequence[Mapping[str, object]], *, endpoint_field: str, mixed: bool
) -> list[list[Mapping[str, object]]]:
    """Make exact-price batches; mixed schedule tests semantic padding."""
    if mixed:
        ordered = sorted(records, key=lambda row: str(row[endpoint_field]))
    else:
        ordered = sorted(records, key=lambda row: (int(row["length"]), str(row[endpoint_field])))
    batches = [ordered[start:start + BATCH] for start in range(0, len(ordered), BATCH)]
    if len(batches) != math.ceil(len(records) / BATCH):
        raise AssertionError("batch schedule price changed")
    return batches


def endpoint_schedules(execution: Mapping[str, object], split: str) -> dict[str, list[list[Mapping[str, object]]]]:
    endpoints = [row for row in execution["endpoints"] if row["split"] == split]
    capture = make_batch_schedule(endpoints, endpoint_field="endpoint_id", mixed=True)
    comparator = make_batch_schedule(endpoints, endpoint_field="endpoint_id", mixed=False)
    expected = 54 if split == "FIT" else 27
    if len(capture) != expected or len(comparator) != expected:
        raise RuntimeError(f"{split} endpoint price changed")
    capture_padding = {
        row["endpoint_id"]: max(int(item["length"]) for item in batch)
        for batch in capture for row in batch
    }
    comparator_padding = {
        row["endpoint_id"]: max(int(item["length"]) for item in batch)
        for batch in comparator for row in batch
    }
    lengths = sorted({int(row["length"]) for row in endpoints})
    max_length = max(lengths)
    for length in lengths:
        candidates = [row for row in endpoints if int(row["length"]) == length]
        if length < max_length and not any(
            capture_padding[row["endpoint_id"]] > length
            and comparator_padding[row["endpoint_id"]] == length
            for row in candidates
        ):
            raise RuntimeError(f"no padded/unpadded comparator for length {length}")
    return {"capture": capture, "comparator": comparator}


def direction_batches(execution: Mapping[str, object], split: str) -> list[list[Mapping[str, object]]]:
    endpoint_by_id = {
        row["endpoint_id"]: row for row in execution["endpoints"] if row["split"] == split
    }
    records = []
    for row in execution["directions"]:
        if row["split"] != split:
            continue
        record = dict(row)
        record["length"] = endpoint_by_id[row["recipient_endpoint_id"]]["length"]
        records.append(record)
    batches = make_batch_schedule(records, endpoint_field="directed_id", mixed=True)
    expected = 117 if split == "FIT" else 59
    if len(batches) != expected:
        raise RuntimeError(f"{split} direction price changed")
    return batches


def _load_runtime_modules():
    # Imports occur only after explicit scientific execution.  Dry-run and tests
    # therefore cannot load a checkpoint or accidentally touch CUDA.
    for path in (ROOT, OPS, POLY):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import torch.nn.functional as functional
    facade = load_module(FACADE, "r585_pinned_observed_facade")
    induction = load_module(INDUCTION, "r585_pinned_induction_term")
    return torch, functional, facade, induction


def _linear(functional, value, weight):
    return functional.linear(weight=weight.to(device=value.device, dtype=value.dtype), input=value)


def factorize_attention_event(event, endpoint_specs, *, torch, functional, induction):
    """Compute native attention plus independently reconstructed final-query factors."""
    native_write, next_value = event.block.attn(event.state, event.first_value)
    state, attention = event.state, event.block.attn
    batch, length, width = state.shape
    if width != 1152 or len(endpoint_specs) != batch:
        raise RuntimeError("factor hook shape changed")
    q = _linear(functional, state, attention.c_q.weight).view(batch, length, 9, 128)
    k = _linear(functional, state, attention.c_k.weight).view(batch, length, 9, 128)
    q2 = _linear(functional, state, attention.c_q2.weight).view(batch, length, 9, 128)
    k2 = _linear(functional, state, attention.c_k2.weight).view(batch, length, 9, 128)
    raw_value = _linear(functional, state, attention.c_v.weight).view(batch, length, 9, 128)
    value = (1 - attention.lamb) * raw_value + attention.lamb * event.first_value.view_as(raw_value)
    cos, sin = attention.rotary(q)
    attention_module = sys.modules[type(attention).__module__]
    q = attention_module.apply_rotary_emb(functional.rms_norm(q, (128,)), cos, sin)
    k = attention_module.apply_rotary_emb(functional.rms_norm(k, (128,)), cos, sin)
    q2 = attention_module.apply_rotary_emb(functional.rms_norm(q2, (128,)), cos, sin)
    k2 = attention_module.apply_rotary_emb(functional.rms_norm(k2, (128,)), cos, sin)
    score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / 128
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / 128
    pattern = score1 * score2
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    pattern = pattern.masked_fill(~causal, 0)
    heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    flattened = heads.transpose(1, 2).contiguous().view(batch, length, width)
    reconstructed_write = _linear(functional, flattened, attention.c_proj.weight)
    for label, tensor in (
        ("native_write", native_write), ("value", value), ("pattern", pattern),
        ("heads", heads), ("reconstructed_write", reconstructed_write),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"nonfinite attention factor tensor: {label}")
    full_error = float((reconstructed_write.float() - native_write.float()).abs().max().cpu())
    terms: list[dict[str, object]] = []
    for local, spec in enumerate(endpoint_specs):
        query = int(spec["final_position"])
        if query >= length:
            raise RuntimeError("semantic query outside batch")
        row_terms = {}
        for site, head in SITES:
            if site != event.site:
                continue
            weight = attention.c_proj.weight[:, head * 128:(head + 1) * 128]
            projected_values = functional.linear(value[local, :, head].float(), weight.float())
            role_e, role_u = [], []
            for source, payload in zip(spec["source_positions"], spec["payload_positions"]):
                support = int(event.tokens[local, int(payload) - 1]) == int(event.tokens[local, query])
                role_e.append(pattern[local, head, query, int(payload)].float() if support else torch.zeros((), device=state.device))
                role_u.append(projected_values[int(payload)])
            factor_term = sum((edge * payload for edge, payload in zip(role_e, role_u)), torch.zeros(1152, device=state.device))
            canonical_head = induction.contract_induction_fetch(
                pattern[:, head].float(), value[:, :, head].float(), event.tokens
            )[local, query]
            non_equality_head = induction.contract_without_induction_fetch(
                pattern[:, head].float(), value[:, :, head].float(), event.tokens
            )[local, query]
            canonical_term = functional.linear(canonical_head, weight.float())
            remainder = functional.linear(non_equality_head, weight.float())
            head_output = functional.linear(heads[local, head, query].float(), weight.float())
            factor_error = float((factor_term - canonical_term).abs().max().cpu())
            reconstruction_error = float((canonical_term + remainder - head_output).abs().max().cpu())
            row_terms[f"L{site}H{head}"] = {
                "e": tuple(float(edge.detach().cpu()) for edge in role_e),
                "u": tuple(payload.detach().cpu().contiguous() for payload in role_u),
                "term": factor_term.detach().cpu().contiguous(),
                "canonical": canonical_term.detach().cpu().contiguous(),
                "head_output": head_output.detach().cpu().contiguous(),
                "remainder": remainder.detach().cpu().contiguous(),
                "factor_error": factor_error,
                "reconstruction_error": reconstruction_error,
            }
        terms.append(row_terms)
    return native_write, next_value, terms, full_error


def combine_frozen_term(recipient: Mapping[str, object], donor: Mapping[str, object], arm: str, *, torch, device):
    if arm not in ("replay", *ARMS):
        raise ValueError("invalid factor arm")
    score_source = donor if arm in ("score", "joint") else recipient
    payload_source = donor if arm in ("payload", "joint") else recipient
    edges = score_source["e"]
    payloads = payload_source["u"]
    return sum(
        (float(edge) * payload.to(device=device, dtype=torch.float32) for edge, payload in zip(edges, payloads)),
        torch.zeros(1152, device=device, dtype=torch.float32),
    )


def build_frozen_insertion_cache(directions, factor_cache, *, torch):
    """Materialize all four directed terms before any intervention forward."""
    frozen = {}
    failures = []
    for direction in directions:
        for term_name in TERM_NAMES:
            recipient = factor_cache[(direction["recipient_endpoint_id"], term_name)]
            donor = factor_cache[(direction["donor_endpoint_id"], term_name)]
            for arm in ("replay", *ARMS):
                value = combine_frozen_term(
                    recipient, donor, arm, torch=torch, device="cpu"
                ).detach().cpu().contiguous()
                frozen[(direction["directed_id"], arm, term_name)] = value
            replay_error = float((
                frozen[(direction["directed_id"], "replay", term_name)].float()
                - recipient["canonical"].float()
            ).abs().max())
            if replay_error > TOLERANCE:
                failures.append(
                    f"frozen_replay_canonical:{direction['directed_id']}:{term_name}:{replay_error}"
                )
    return frozen, failures


def _padded_tokens(batch, *, torch, device):
    length = max(int(row["length"]) for row in batch)
    tokens = torch.full((len(batch), length), 50_256, dtype=torch.long, device=device)
    for index, row in enumerate(batch):
        ids = torch.as_tensor(row["token_ids"], dtype=torch.long, device=device)
        tokens[index, : len(ids)] = ids
    return tokens


def _native_dispatchers():
    def attention(event):
        return event.block.attn(event.state, event.first_value)
    def mlp(event):
        return event.block.mlp(event.state)
    return attention, mlp


def _logit_measurement(logits, spec, *, torch) -> dict[str, object]:
    vector = logits[int(spec["final_position"])].float().detach().cpu().contiguous()
    if not bool(torch.isfinite(vector).all()):
        raise RuntimeError("nonfinite model logits")
    answer = int(spec["answer_id"])
    other = int(spec["other_answer_id"])
    log_normalizer = float(torch.logsumexp(vector, dim=-1))
    answer_logit, other_logit = float(vector[answer]), float(vector[other])
    return {
        "answer_id": answer,
        "other_answer_id": other,
        "answer_logit": answer_logit,
        "other_logit": other_logit,
        "correct_margin": answer_logit - other_logit,
        "log_normalizer": log_normalizer,
        "correct_ce": log_normalizer - answer_logit,
        "full_logits": vector,
    }


def collect_capture_replay(model, batches, *, torch, functional, facade, induction):
    """Capture every native endpoint factor before any directed intervention."""
    device = next(model.parameters()).device
    factors: dict[tuple[str, str], dict[str, object]] = {}
    measurements = {}
    padding = {}
    failures = []
    realized_operations = []
    exactness = {
        "native_attention_reconstruction_max_abs": 0.0,
        "equality_factor_max_abs": 0.0,
        "equality_plus_independent_remainder_max_abs": 0.0,
    }
    calls = 0
    with torch.inference_mode():
        for batch in batches:
            tokens = _padded_tokens(batch, torch=torch, device=device)
            batch_terms: dict[int, dict[str, object]] = {i: {} for i in range(len(batch))}
            max_full_error = 0.0

            def attention(event):
                nonlocal max_full_error
                if event.site not in {site for site, _ in SITES}:
                    return event.block.attn(event.state, event.first_value)
                write, next_value, terms, full_error = factorize_attention_event(
                    event, batch, torch=torch, functional=functional, induction=induction
                )
                max_full_error = max(max_full_error, full_error)
                modified = write.clone()
                for local, row_terms in enumerate(terms):
                    query = int(batch[local]["final_position"])
                    total_delta = torch.zeros(1152, device=device, dtype=write.dtype)
                    for term_name, term in row_terms.items():
                        batch_terms[local][term_name] = term
                        total_delta += (
                            term["term"].to(device=device, dtype=write.dtype)
                            - term["canonical"].to(device=device, dtype=write.dtype)
                        )
                    modified[local, query] += total_delta
                return modified, next_value

            def mlp(event):
                return event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(
                model, tokens, attention, mlp, require_production=True
            )
            calls += 1
            if max_full_error > TOLERANCE:
                failures.append(f"native_attention_reconstruction:{max_full_error}")
            exactness["native_attention_reconstruction_max_abs"] = max(
                exactness["native_attention_reconstruction_max_abs"], max_full_error
            )
            for local, spec in enumerate(batch):
                endpoint = str(spec["endpoint_id"])
                if set(batch_terms[local]) != set(TERM_NAMES):
                    failures.append(f"factor_capture_incomplete:{endpoint}")
                for term_name, term in batch_terms[local].items():
                    for role in ROLES:
                        realized_operations.append({
                            "split": str(spec["split"]),
                            "endpoint_id": endpoint,
                            "site": term_name,
                            "role": role,
                        })
                    if term["factor_error"] > TOLERANCE:
                        failures.append(f"canonical_factor:{endpoint}:{term_name}")
                    if term["reconstruction_error"] > TOLERANCE:
                        failures.append(f"head_reconstruction:{endpoint}:{term_name}")
                    exactness["equality_factor_max_abs"] = max(
                        exactness["equality_factor_max_abs"], float(term["factor_error"])
                    )
                    exactness["equality_plus_independent_remainder_max_abs"] = max(
                        exactness["equality_plus_independent_remainder_max_abs"],
                        float(term["reconstruction_error"]),
                    )
                    factors[(endpoint, term_name)] = term
                measurements[endpoint] = _logit_measurement(logits[local], spec, torch=torch)
                padding[endpoint] = int(tokens.shape[1])
    return factors, measurements, padding, calls, failures, realized_operations, exactness


def collect_native_comparator(model, batches, *, torch, facade):
    device = next(model.parameters()).device
    measurements, padding = {}, {}
    calls = 0
    attention, mlp = _native_dispatchers()
    with torch.inference_mode():
        for batch in batches:
            tokens = _padded_tokens(batch, torch=torch, device=device)
            logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
            calls += 1
            for local, spec in enumerate(batch):
                endpoint = str(spec["endpoint_id"])
                measurements[endpoint] = _logit_measurement(logits[local], spec, torch=torch)
                padding[endpoint] = int(tokens.shape[1])
    return measurements, padding, calls


def capture_instrument_failures(replay, native, replay_padding, native_padding, endpoint_specs):
    failures = []
    maximum_logit_error = 0.0
    for endpoint, replay_row in replay.items():
        native_row = native[endpoint]
        error = float((replay_row["full_logits"] - native_row["full_logits"]).abs().max())
        maximum_logit_error = max(maximum_logit_error, error)
        if error > TOLERANCE:
            failures.append(f"replay_native_logits:{endpoint}:{error}")
    lengths = sorted({int(row["length"]) for row in endpoint_specs})
    maximum = max(lengths)
    active_lengths = []
    for length in lengths:
        rows = [row for row in endpoint_specs if int(row["length"]) == length]
        if length < maximum and not any(
            replay_padding[row["endpoint_id"]] > length
            and native_padding[row["endpoint_id"]] == length
            for row in rows
        ):
            failures.append(f"padding_tripwire_dead:length{length}")
        elif length < maximum:
            active_lengths.append(length)
    return failures, {
        "replay_native_logit_max_abs": maximum_logit_error,
        "padding_tripwire_active_lengths": active_lengths,
    }


def collect_intervention_arm(
    model, batches, arm, endpoint_specs, frozen_insertions, replay, native,
    *, torch, functional, facade, induction,
):
    device = next(model.parameters()).device
    records = []
    vector_rows = []
    calls = 0
    with torch.inference_mode():
        for batch in batches:
            specs = [endpoint_specs[(row["split"], row["recipient_endpoint_id"])] for row in batch]
            tokens = _padded_tokens(specs, torch=torch, device=device)
            per_row = {
                i: {"live": {}, "delta": {}, "factor_errors": [], "hook_errors": []}
                for i in range(len(batch))
            }

            def attention(event):
                if event.site not in {site for site, _ in SITES}:
                    return event.block.attn(event.state, event.first_value)
                write, next_value, live_terms, _ = factorize_attention_event(
                    event, specs, torch=torch, functional=functional, induction=induction
                )
                modified = write.clone()
                for local, current in enumerate(live_terms):
                    query = int(specs[local]["final_position"])
                    total_delta = torch.zeros(1152, device=device, dtype=write.dtype)
                    direction = batch[local]
                    for term_name, live in current.items():
                        per_row[local]["factor_errors"].extend(
                            [float(live["factor_error"]), float(live["reconstruction_error"])]
                        )
                        inserted = frozen_insertions[
                            (direction["directed_id"], arm, term_name)
                        ].to(device=device, dtype=torch.float32)
                        live_removed = live["canonical"].to(device=device, dtype=torch.float32)
                        delta = inserted - live_removed
                        if not bool(torch.isfinite(delta).all()):
                            raise RuntimeError("nonfinite intervention delta")
                        total_delta += delta.to(write.dtype)
                        per_row[local]["live"][term_name] = live_removed.detach().cpu().contiguous()
                        per_row[local]["delta"][term_name] = delta.detach().cpu().contiguous()
                    before = modified[local, query].clone()
                    modified[local, query] += total_delta
                    observed = modified[local, query].float() - before.float()
                    per_row[local]["hook_errors"].append(
                        float((observed - total_delta.float()).abs().max().cpu())
                    )
                return modified, next_value

            def mlp(event):
                return event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
            calls += 1
            for local, direction in enumerate(batch):
                spec = specs[local]
                measurement = _logit_measurement(logits[local], spec, torch=torch)
                replay_row = replay[direction["recipient_endpoint_id"]]
                donor_native = native[direction["donor_endpoint_id"]]
                recipient_native = native[direction["recipient_endpoint_id"]]
                donor_answer = int(direction["donor_answer_id"])
                recipient_answer = int(direction["recipient_answer_id"])
                other = int(direction["recipient_other_answer_id"])

                def logit(state, token):
                    if token == state["answer_id"]:
                        return float(state["answer_logit"])
                    if token == state["other_answer_id"]:
                        return float(state["other_logit"])
                    return float(state["full_logits"][token])

                def ce(state, token):
                    return float(state["log_normalizer"]) - logit(state, token)

                if direction["answer_changes"]:
                    m_i = logit(measurement, donor_answer) - logit(measurement, recipient_answer)
                    m_r = logit(replay_row, donor_answer) - logit(replay_row, recipient_answer)
                    m_d = logit(donor_native, donor_answer) - logit(donor_native, recipient_answer)
                    m_x = logit(recipient_native, donor_answer) - logit(recipient_native, recipient_answer)
                    n_value, d_value = m_i - m_r, m_d - m_x
                    q_value = ce(replay_row, donor_answer) - ce(measurement, donor_answer)
                else:
                    sign = int(direction["donor_coherence_sign"] or 1)
                    c_i = logit(measurement, recipient_answer) - logit(measurement, other)
                    c_r = logit(replay_row, recipient_answer) - logit(replay_row, other)
                    c_d = logit(donor_native, recipient_answer) - logit(donor_native, other)
                    c_x = logit(recipient_native, recipient_answer) - logit(recipient_native, other)
                    n_value, d_value = sign * (c_i - c_r), sign * (c_d - c_x)
                    q_value = sign * (ce(replay_row, recipient_answer) - ce(measurement, recipient_answer))
                squared = float((measurement["full_logits"] - replay_row["full_logits"]).double().square().sum())
                delta_norms = [
                    float(per_row[local]["delta"][term].float().norm()) for term in TERM_NAMES
                ]
                record = {
                    **{key: direction[key] for key in (
                        "split", "directed_id", "row_id", "group_id", "family", "variant",
                        "recipient_condition", "direction", "control_kind", "answer_changes",
                    )},
                    "arm": arm,
                    "recipient_endpoint_id": direction["recipient_endpoint_id"],
                    "donor_endpoint_id": direction["donor_endpoint_id"],
                    "recipient_answer_id": recipient_answer,
                    "donor_answer_id": donor_answer,
                    "other_answer_id": other,
                    "replay_correct_margin": float(replay_row["correct_margin"]),
                    "correct_margin": float(measurement["correct_margin"]),
                    "replay_correct_ce": float(replay_row["correct_ce"]),
                    "correct_ce": float(measurement["correct_ce"]),
                    "n": float(n_value),
                    "d": float(d_value),
                    "q": float(q_value),
                    "insertion_activity": float(np.median(np.asarray(delta_norms, dtype=np.float64))),
                    "per_site_delta_norms": delta_norms,
                    "live_factor_max_error": max(per_row[local]["factor_errors"], default=0.0),
                    "hook_delta_sum_max_error": max(per_row[local]["hook_errors"], default=0.0),
                    "vocab_squared_difference_sum": squared,
                    "vocab_size": int(measurement["full_logits"].numel()),
                    "vocab_rms": math.sqrt(squared / int(measurement["full_logits"].numel())),
                    "answer_logit": float(measurement["answer_logit"]),
                    "other_logit": float(measurement["other_logit"]),
                    "log_normalizer": float(measurement["log_normalizer"]),
                }
                records.append(record)
                vector_rows.append({
                    "directed_id": direction["directed_id"],
                    "arm": arm,
                    "live": [per_row[local]["live"][term] for term in TERM_NAMES],
                    "delta": [per_row[local]["delta"][term] for term in TERM_NAMES],
                    **({"full_logits": measurement["full_logits"]} if (
                        direction["family"] == "two_valid_sources_selector_swap"
                        or direction["control_kind"] == "lag"
                        or (
                            direction["family"] == "match_break_payload_preserved"
                            and direction["direction"] == "base_to_donor"
                            and arm in ("score", "joint")
                        )
                        or (
                            direction["family"] == "match_break_payload_preserved"
                            and direction["direction"] == "donor_to_base"
                            and arm == "payload"
                        )
                    ) else {}),
                })
    return records, vector_rows, calls


def _cell_key(row: Mapping[str, object]) -> str:
    return "|".join(str(row[key]) for key in (
        "split", "family", "variant", "recipient_condition", "direction"
    ))


def _records_for(records, cell, arm):
    output = [row for row in records if _cell_key(row) == cell["cell_id"] and row["arm"] == arm]
    if len(output) != len(cell["group_ids"]) or {
        row["group_id"] for row in output
    } != set(cell["group_ids"]):
        raise RuntimeError(f"evidence membership mismatch: {cell['cell_id']}:{arm}")
    return sorted(output, key=lambda row: row["group_id"])


def compute_fit_scales(records, manifests):
    scales = {}
    family_for_arm = {
        "score": "two_valid_sources_selector_swap",
        "payload": "payload_swap_match_preserved",
        "joint": "two_valid_sources_selector_swap",
    }
    for arm in ARMS:
        for condition in ("s0p0", "s0p1", "s1p0", "s1p1"):
            cells = [
                cell for cell in manifests["target_cells"]
                if cell["split"] == "FIT" and cell["family"] == family_for_arm[arm]
                and cell["recipient_condition"] == condition
            ]
            if len(cells) != 1:
                raise RuntimeError(f"non-unique FIT scale cell: {arm}:{condition}")
            rows = _records_for(records, cells[0], arm)
            insertion = float(np.median([row["insertion_activity"] for row in rows]))
            margin = float(np.median([abs(row["n"]) for row in rows]))
            vocabulary = float(np.median([row["vocab_rms"] for row in rows]))
            valid = all(math.isfinite(value) and value > 0 for value in (insertion, margin, vocabulary))
            scales[f"{arm}:{condition}"] = {
                "target_cell_id": cells[0]["cell_id"],
                "insertion": insertion,
                "margin": margin,
                "vocabulary": vocabulary,
                "valid": valid,
            }
    return scales


def _transfer_gate(report, ce_report, recovery_threshold, fraction_threshold):
    return bool(
        report["denominator_valid"]
        and report["mean_recovery"] >= recovery_threshold
        and report["median_recovery"] >= recovery_threshold
        and report["numerator_bootstrap"]["lower95"] > 0
        and report["positive_numerator_fraction"] >= fraction_threshold
        and ce_report["lower95"] > 0
    )


def score_split(records, split, manifests, fit_scales, *, replicates=BOOTSTRAPS):
    """Score one split exactly; thresholds are frozen prospective gates."""
    failures = {
        "invalid_instrument": [],
        "native_denominator_or_scale_null": [],
        "factor_capacity_null": [],
        "factorization_not_identified": [],
        "insufficient_active_controls": [],
        "broad_contextual_equality_write": [],
    }
    reports = {"targets": {}, "joint_diagonal": {}, "controls": {}, "coverage": {}}
    target_cells = [cell for cell in manifests["target_cells"] if cell["split"] == split]
    control_cells = [cell for cell in manifests["control_cells"] if cell["split"] == split]
    for cell in target_cells:
        family = cell["family"]
        if family == "selector_payload_joint_answer_preserved":
            by_arm = {arm: _records_for(records, cell, arm) for arm in ARMS}
            values = {}
            for metric, arm in (("single_score_harm_mean", "score"), (
                "single_payload_harm_mean", "payload"), ("factorial_interaction_mean", "joint")):
                by_group = {}
                for index, row in enumerate(by_arm[arm]):
                    replay = row["replay_correct_margin"]
                    if arm == "score":
                        value = replay - row["correct_margin"]
                    elif arm == "payload":
                        value = replay - row["correct_margin"]
                    else:
                        score_row, payload_row = by_arm["score"][index], by_arm["payload"][index]
                        if not (score_row["group_id"] == payload_row["group_id"] == row["group_id"]):
                            raise RuntimeError("joint diagonal group alignment changed")
                        value = (
                            row["correct_margin"] - score_row["correct_margin"]
                            - payload_row["correct_margin"] + replay
                        ) / 4.0
                    by_group[row["group_id"]] = [float(value)]
                cell_id = cell["cell_id"] + f"|{arm}|{metric}"
                values[metric] = bootstrap_mean(by_group, cell_id, replicates=replicates)
            joint_rows = by_arm["joint"]
            scale = fit_scales[f"joint:{cell['recipient_condition']}"]["vocabulary"]
            report = {
                **values,
                "replay_positive_fraction": float(np.mean([
                    row["replay_correct_margin"] > 0 for row in joint_rows
                ])),
                "joint_positive_fraction": float(np.mean([
                    row["correct_margin"] > 0 for row in joint_rows
                ])),
                "mean_joint_minus_replay_ce": float(np.mean([
                    row["correct_ce"] - row["replay_correct_ce"] for row in joint_rows
                ])),
                "median_joint_vocab_rms": float(np.median([row["vocab_rms"] for row in joint_rows])),
                "vocabulary_scale": scale,
            }
            report["passes"] = bool(
                report["replay_positive_fraction"] >= .75
                and report["joint_positive_fraction"] >= .75
                and values["single_score_harm_mean"]["lower95"] > 0
                and values["single_payload_harm_mean"]["lower95"] > 0
                and values["factorial_interaction_mean"]["lower95"] > 0
                and report["mean_joint_minus_replay_ce"] <= .10
                and report["median_joint_vocab_rms"] <= .25 * scale
            )
            reports["joint_diagonal"][cell["cell_id"]] = report
            if not report["passes"]:
                failures["factorization_not_identified"].append(cell["cell_id"])
            continue

        for arm in ARMS:
            # Structural replay identities do not instantiate recovery cells.
            if family == "two_valid_sources_selector_swap" and arm == "payload":
                continue
            if family == "match_break_payload_preserved" and (
                cell["direction"] == "donor_to_base" and arm == "payload"
            ):
                continue
            rows = _records_for(records, cell, arm)
            prefix = cell["cell_id"] + f"|{arm}"
            recovery = recovery_summary(rows, prefix, replicates=replicates)
            report = {"recovery": recovery}
            if not recovery["denominator_valid"]:
                failures["native_denominator_or_scale_null"].append(prefix)
            intended = (
                (family == "two_valid_sources_selector_swap" and arm in ("score", "joint"))
                or (family == "payload_swap_match_preserved" and arm in ("payload", "joint"))
                or (family == "match_break_payload_preserved" and arm in ("score", "joint"))
            )
            if intended:
                ce = donor_ce_summary(rows, prefix, replicates=replicates)
                fraction = .70 if family == "match_break_payload_preserved" else .75
                report["donor_ce"] = ce
                report["passes"] = _transfer_gate(recovery, ce, .30, fraction)
                if not report["passes"] and recovery["denominator_valid"]:
                    failures["factor_capacity_null"].append(prefix)
            else:
                report["passes"] = bool(
                    recovery["denominator_valid"] and abs(recovery["mean_recovery"]) <= .25
                )
                if not report["passes"] and recovery["denominator_valid"]:
                    failures["factorization_not_identified"].append(prefix)
            reports["targets"][prefix] = report

        if family == "payload_swap_match_preserved":
            payload = reports["targets"][cell["cell_id"] + "|payload"]["recovery"]
            joint = reports["targets"][cell["cell_id"] + "|joint"]["recovery"]
            if payload["denominator_valid"] and joint["denominator_valid"] and (
                joint["mean_recovery"] < payload["mean_recovery"] - .10
            ):
                failures["factorization_not_identified"].append(cell["cell_id"] + "|joint_below_payload")

    if split == "FIT" and any(not scale["valid"] for scale in fit_scales.values()):
        failures["native_denominator_or_scale_null"].extend(
            f"scale:{name}" for name, scale in fit_scales.items() if not scale["valid"]
        )

    active_by_key = {}
    for cell in control_cells:
        for arm in ARMS:
            rows = _records_for(records, cell, arm)
            scale = fit_scales[f"{arm}:{cell['recipient_condition']}"]
            structurally_excluded = cell["control_kind"] == "lag" and arm == "payload"
            active_fraction = float(np.mean([
                row["insertion_activity"] >= .10 * scale["insertion"] for row in rows
            ]))
            adequately_active = bool(not structurally_excluded and active_fraction >= .75)
            key = f"{arm}|{cell['direction']}|{cell['recipient_condition']}"
            if adequately_active:
                active_by_key.setdefault(key, set()).add(cell["control_kind"])
            report = {
                "adequately_active": adequately_active,
                "structurally_excluded": structurally_excluded,
                "active_fraction": active_fraction,
                "median_absolute_margin_change": float(np.median([
                    abs(row["correct_margin"] - row["replay_correct_margin"]) for row in rows
                ])),
                "median_vocab_rms": float(np.median([row["vocab_rms"] for row in rows])),
                "mean_ce_change": float(np.mean([
                    row["correct_ce"] - row["replay_correct_ce"] for row in rows
                ])),
                "positive_correct_margin_fraction": float(np.mean([
                    row["correct_margin"] > 0 for row in rows
                ])),
                "scales": scale,
            }
            report["passes"] = bool(
                not adequately_active or (
                    report["median_absolute_margin_change"] <= .25 * scale["margin"]
                    and report["median_vocab_rms"] <= .25 * scale["vocabulary"]
                    and report["mean_ce_change"] <= .10
                    and report["positive_correct_margin_fraction"] >= .75
                )
            )
            identity = cell["cell_id"] + f"|{arm}"
            reports["controls"][identity] = report
            if adequately_active and not report["passes"]:
                failures["broad_contextual_equality_write"].append(identity)

    for arm in ARMS:
        for direction in ("base_to_donor", "donor_to_base"):
            for condition in ("s0p0", "s0p1", "s1p0", "s1p1"):
                key = f"{arm}|{direction}|{condition}"
                kinds = sorted(active_by_key.get(key, set()))
                reports["coverage"][key] = {"active_families": kinds, "passes": len(kinds) >= 2}
                if len(kinds) < 2:
                    failures["insufficient_active_controls"].append(key)
    reports["bootstrap_realization"] = validate_realized_bootstraps(
        reports, split, manifests
    )
    return reports, failures


def structural_identity_failures(records, vector_rows, manifests, replay):
    record_by_key = {(row["directed_id"], row["arm"]): row for row in records}
    by_key = {(row["directed_id"], row["arm"]): row for row in vector_rows}
    failures, evidence = [], []
    for identity in manifests["structural_identities"]:
        cell = next(
            cell for cell in (*manifests["target_cells"], *manifests["control_cells"])
            if cell["cell_id"] == identity["cell_id"]
        )
        for directed_id in cell["directed_ids"]:
            if (directed_id, "score") not in record_by_key:
                continue
            left = identity["left_arm"]
            right = identity["right_arm"]
            recipient = record_by_key[(directed_id, "score")]["recipient_endpoint_id"]
            left_logits = by_key[(directed_id, left)]["full_logits"] if left != "replay" else replay[recipient]["full_logits"]
            right_logits = by_key[(directed_id, right)]["full_logits"] if right != "replay" else replay[recipient]["full_logits"]
            error = float((left_logits - right_logits).abs().max())
            evidence.append({"directed_id": directed_id, **identity, "max_abs": error})
            if error > TOLERANCE:
                failures.append(f"structural_identity:{directed_id}:{left}={right}:{error}")
    return failures, evidence


def planted_intervention_records(execution, *, null=False):
    """Outcome-free fixture spanning the literal cell manifest."""
    rows = []
    manifests = execution["manifests"]
    cells = (*manifests["target_cells"], *manifests["control_cells"])
    for cell in cells:
        for arm in ARMS:
            for group_id in cell["group_ids"]:
                family = cell["family"]
                n_value, d_value, q_value = .5, 1.0, .2
                replay_margin, margin = 2.0, 2.0
                vocab_rms = .02
                if family == "payload_swap_match_preserved" and arm == "score":
                    n_value = .10
                if family == "match_break_payload_preserved" and arm == "payload":
                    n_value = .10 if cell["direction"] == "base_to_donor" else 0.0
                if family == "selector_payload_joint_answer_preserved":
                    margin = 1.0
                    if arm == "joint":
                        margin = 1.0
                if cell["role"] == "control":
                    margin = 1.99
                    n_value, d_value, q_value = .01, 1.0, 0.0
                if family == "two_valid_sources_selector_swap" and arm == "payload":
                    n_value = 0.0
                if null and family == "two_valid_sources_selector_swap" and arm == "score" \
                        and cell["split"] == "FIT" and cell["variant"] == "payload_assignment_0":
                    n_value, q_value = -.10, -.05
                rows.append({
                    "split": cell["split"],
                    "directed_id": cell["directed_ids"][cell["group_ids"].index(group_id)],
                    "row_id": "fixture-row",
                    "group_id": group_id,
                    "family": family,
                    "variant": cell["variant"],
                    "recipient_condition": cell["recipient_condition"],
                    "direction": cell["direction"],
                    "control_kind": cell["control_kind"],
                    "answer_changes": family not in (
                        "selector_payload_joint_answer_preserved",
                        "match_break_payload_preserved",
                        "irrelevant_source_edit", "irrelevant_payload_edit",
                        "copy_relation_preserved_nuisance_change",
                    ),
                    "arm": arm,
                    "recipient_endpoint_id": "fixture-recipient",
                    "donor_endpoint_id": "fixture-donor",
                    "recipient_answer_id": 1,
                    "donor_answer_id": 2,
                    "other_answer_id": 2,
                    "replay_correct_margin": replay_margin,
                    "correct_margin": margin,
                    "replay_correct_ce": 0.2,
                    "correct_ce": 0.21,
                    "n": n_value,
                    "d": d_value,
                    "q": q_value,
                    "insertion_activity": 1.0 if cell["role"] == "target" else .2,
                    "per_site_delta_norms": [.2, .2, .2, .2],
                    "live_factor_max_error": 0.0,
                    "hook_delta_sum_max_error": 0.0,
                    "vocab_squared_difference_sum": vocab_rms * vocab_rms * VOCABULARY_SIZE,
                    "vocab_size": VOCABULARY_SIZE,
                    "vocab_rms": vocab_rms if cell["role"] == "control" or family == "selector_payload_joint_answer_preserved" else .2,
                    "answer_logit": margin / 2,
                    "other_logit": -margin / 2,
                    "log_normalizer": 2.0,
                })
    return rows


def _array_descriptor(
    path: Path,
    array: np.ndarray,
    row_order: Sequence[object],
    *,
    logical_path: Path | None = None,
) -> dict[str, object]:
    logical_path = path if logical_path is None else logical_path
    return {
        "path": str(logical_path.relative_to(ROOT.parent.parent)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "row_order_sha256": content_sha256(list(row_order)),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_evidence(
    execution, factors, records, vector_rows, replay, native, *,
    evidence_dir: Path, logical_evidence_dir: Path = EVIDENCE_DIR,
    crash_injector=None,
):
    """Write a complete evidence tree into a non-final staging directory."""
    if evidence_dir.exists():
        raise RuntimeError(f"R585 evidence staging namespace already exists: {evidence_dir}")
    endpoints = sorted(
        [row for row in execution["endpoints"] if all(
            (row["endpoint_id"], name) in factors for name in TERM_NAMES
        )],
        key=lambda row: (row["split"], row["endpoint_id"]),
    )
    endpoint_index = {row["endpoint_id"]: index for index, row in enumerate(endpoints)}
    e = np.empty((len(endpoints), 4, 2), dtype="<f4")
    u = np.empty((len(endpoints), 4, 2, 1152), dtype="<f4")
    term = np.empty((len(endpoints), 4, 1152), dtype="<f4")
    head = np.empty_like(term)
    remainder = np.empty_like(term)
    factor_exactness = []
    for endpoint in endpoints:
        i = endpoint_index[endpoint["endpoint_id"]]
        for j, name in enumerate(TERM_NAMES):
            factor = factors[(endpoint["endpoint_id"], name)]
            e[i, j] = np.asarray(factor["e"], dtype="<f4")
            u[i, j] = np.stack([value.numpy() for value in factor["u"]]).astype("<f4")
            term[i, j] = factor["canonical"].numpy().astype("<f4")
            head[i, j] = factor["head_output"].numpy().astype("<f4")
            remainder[i, j] = factor["remainder"].numpy().astype("<f4")
            factor_exactness.append({
                "split": endpoint["split"],
                "endpoint_id": endpoint["endpoint_id"],
                "site": name,
                "equality_factor_max_abs": float(factor["factor_error"]),
                "equality_plus_independent_remainder_max_abs": float(
                    factor["reconstruction_error"]
                ),
            })
    arrays = [("native_e", e), ("native_u", u), ("canonical_term", term),
              ("native_head_output", head), ("non_equality_remainder", remainder)]
    record_split = {
        (row["directed_id"], row["arm"]): row["split"] for row in records
    }
    ordered_vectors = sorted(
        vector_rows,
        key=lambda row: (
            record_split[(row["directed_id"], row["arm"])],
            row["directed_id"], row["arm"],
        ),
    )
    if vector_rows:
        live = np.stack([[value.numpy() for value in row["live"]] for row in ordered_vectors]).astype("<f4")
        delta = np.stack([[value.numpy() for value in row["delta"]] for row in ordered_vectors]).astype("<f4")
        arrays.extend((("live_removed", live), ("hook_delta", delta)))
    for name, array in arrays:
        _finite_array(array, name)
    require_finite_json(factor_exactness, "factor_exactness")
    require_finite_json(list(records), "directed_records")
    evidence_dir.mkdir(parents=False)
    descriptors = []
    for name, array in arrays:
        path = EVIDENCE_DIR / f"{name}.npy"
        staged_path = evidence_dir / path.name
        np.save(staged_path, array, allow_pickle=False)
        with staged_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if crash_injector is not None:
            crash_injector(f"evidence-write:{staged_path.name}")
        row_order = (
            [[row["directed_id"], row["arm"]] for row in ordered_vectors]
            if name in ("live_removed", "hook_delta")
            else [row["endpoint_id"] for row in endpoints]
        )
        descriptors.append(_array_descriptor(
            staged_path, array, row_order, logical_path=logical_evidence_dir / path.name
        ))

    endpoint_path = evidence_dir / "endpoint_measurements.jsonl"
    with endpoint_path.open("w") as handle:
        for endpoint in endpoints:
            identifier = endpoint["endpoint_id"]
            payload = {
                **endpoint,
                "replay": {key: value for key, value in replay[identifier].items() if key != "full_logits"},
                "native": {key: value for key, value in native[identifier].items() if key != "full_logits"},
            }
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if crash_injector is not None:
        crash_injector(f"evidence-write:{endpoint_path.name}")
    descriptors.append({
        "path": str((logical_evidence_dir / endpoint_path.name).relative_to(ROOT.parent.parent)),
        "sha256": sha256(endpoint_path),
        "bytes": endpoint_path.stat().st_size, "dtype": "jsonl", "shape": [len(endpoints)],
        "row_order_sha256": content_sha256([row["endpoint_id"] for row in endpoints]),
    })
    record_path = evidence_dir / "directed_arm_measurements.jsonl"
    ordered_records = sorted(records, key=lambda row: (row["split"], row["directed_id"], row["arm"]))
    with record_path.open("w") as handle:
        for record in ordered_records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if crash_injector is not None:
        crash_injector(f"evidence-write:{record_path.name}")
    descriptors.append({
        "path": str((logical_evidence_dir / record_path.name).relative_to(ROOT.parent.parent)),
        "sha256": sha256(record_path),
        "bytes": record_path.stat().st_size, "dtype": "jsonl", "shape": [len(records)],
        "row_order_sha256": content_sha256([
            [row["directed_id"], row["arm"]] for row in ordered_records
        ]),
    })
    factor_exactness.sort(key=lambda row: (row["split"], row["endpoint_id"], row["site"]))
    factor_path = evidence_dir / "factor_exactness.jsonl"
    with factor_path.open("w") as handle:
        for row in factor_exactness:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if crash_injector is not None:
        crash_injector(f"evidence-write:{factor_path.name}")
    descriptors.append({
        "path": str((logical_evidence_dir / factor_path.name).relative_to(ROOT.parent.parent)),
        "sha256": sha256(factor_path), "bytes": factor_path.stat().st_size,
        "dtype": "jsonl", "shape": [len(factor_exactness)],
        "row_order_sha256": content_sha256([
            [row["endpoint_id"], row["site"]] for row in factor_exactness
        ]),
    })
    _fsync_directory(evidence_dir)
    return descriptors


def validate_primitive_logit_identities(records):
    failures = []
    for row in records:
        primitive_fields = (
            "answer_logit", "other_logit", "correct_margin", "log_normalizer",
            "correct_ce", "vocab_squared_difference_sum", "vocab_rms",
            "live_factor_max_error", "hook_delta_sum_max_error",
        )
        nonfinite = [
            field for field in primitive_fields
            if not math.isfinite(float(row.get(field, 0.0)))
        ]
        if nonfinite or int(row.get("vocab_size", 0)) <= 0:
            failures.append(
                f"nonfinite_primitive:{row['directed_id']}:{row['arm']}:{','.join(nonfinite)}"
            )
            continue
        margin = float(row["answer_logit"]) - float(row["other_logit"])
        ce = float(row["log_normalizer"]) - float(row["answer_logit"])
        rms = math.sqrt(float(row["vocab_squared_difference_sum"]) / int(row["vocab_size"]))
        if abs(margin - float(row["correct_margin"])) > 1e-7:
            failures.append(f"primitive_margin:{row['directed_id']}:{row['arm']}")
        if abs(ce - float(row["correct_ce"])) > 1e-7:
            failures.append(f"primitive_ce:{row['directed_id']}:{row['arm']}")
        if abs(rms - float(row["vocab_rms"])) > 1e-7:
            failures.append(f"primitive_vocab_rms:{row['directed_id']}:{row['arm']}")
        if float(row.get("live_factor_max_error", 0.0)) > TOLERANCE:
            failures.append(f"live_factor:{row['directed_id']}:{row['arm']}")
        if float(row.get("hook_delta_sum_max_error", 0.0)) > TOLERANCE:
            failures.append(f"hook_delta:{row['directed_id']}:{row['arm']}")
    return failures


def _prefix_select_failures(failures):
    return {"select_" + key: list(values) for key, values in failures.items()}


def _merge_failure_classes(*mappings):
    output = {}
    for mapping in mappings:
        for key, values in mapping.items():
            output.setdefault(key, []).extend(values)
    for label in TERMINALS:
        if label != "held_operational_selector_payload_factorization":
            output.setdefault(label, [])
    return output


def recover_stale_publication(
    *, root: Path = ROOT, out: Path = OUT, receipt: Path = RECEIPT,
    evidence: Path = EVIDENCE_DIR,
) -> None:
    """Quarantine interrupted final/staging paths without deleting evidence."""
    finals = {"result": out, "receipt": receipt, "evidence": evidence}
    occupied = {name: path for name, path in finals.items() if path.exists()}
    stale = sorted(root.glob(STAGE_PREFIX + "*"))
    if len(occupied) == len(finals):
        raise RuntimeError("R585 complete output namespace already exists")
    if not occupied and not stale:
        return
    recovery = Path(tempfile.mkdtemp(prefix=RECOVERY_PREFIX, dir=root))
    for name, path in occupied.items():
        os.replace(path, recovery / f"partial-{name}-{path.name}")
    for index, path in enumerate(stale):
        os.replace(path, recovery / f"stage-{index}-{path.name}")
    _fsync_directory(recovery)
    _fsync_directory(root)
    raise RuntimeError(f"recovered incomplete R585 publication into {recovery}; rerun preflight")


def create_stage_root(root: Path = ROOT) -> Path:
    stage = Path(tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=root))
    if stage.stat().st_dev != root.stat().st_dev:
        raise RuntimeError("R585 stage is not on the output filesystem")
    return stage


def _write_bytes_fsync(
    path: Path, payload: bytes, *, label: str | None = None, crash_injector=None
) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if crash_injector is not None:
        crash_injector(label or path.name)


def _staging_resolver(stage_root: Path):
    stage_evidence = stage_root / "evidence"

    def resolve(logical: Path) -> Path:
        if logical.parent == EVIDENCE_DIR:
            return stage_evidence / logical.name
        return logical

    return resolve


def publish_staged_package(
    stage_root: Path,
    *,
    out: Path = OUT,
    receipt: Path = RECEIPT,
    evidence: Path = EVIDENCE_DIR,
    crash_injector=None,
) -> None:
    """Publish three individually atomic paths, with receipt last as commit marker."""
    moves = [
        ("evidence", stage_root / "evidence", evidence),
        ("result", stage_root / "result.json", out),
        ("receipt", stage_root / "receipt.json", receipt),
    ]
    if any(destination.exists() for _, _, destination in moves):
        raise RuntimeError("R585 final namespace became occupied before publication")
    if any(not source.exists() for _, source, _ in moves):
        raise RuntimeError("R585 staged package is incomplete")
    published = []
    try:
        for label, source, destination in moves:
            os.replace(source, destination)
            published.append((label, source, destination))
            _fsync_directory(destination.parent)
            if crash_injector is not None:
                crash_injector(label)
    except BaseException:
        for _, source, destination in reversed(published):
            if destination.exists() and not source.exists():
                os.replace(destination, source)
        _fsync_directory(stage_root)
        _fsync_directory(out.parent)
        raise
    stage_root.rmdir()
    _fsync_directory(out.parent)


def _finish_result(result, stage_root: Path, *, crash_injector=None):
    require_finite_json(result)
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    staged_result = stage_root / "result.json"
    _write_bytes_fsync(
        staged_result, encoded,
        label="staged-result-write", crash_injector=crash_injector,
    )
    receipt = make_receipt_fixture(result)
    if receipt["result_sha256"] != hashlib.sha256(encoded).hexdigest():
        raise RuntimeError("canonical result encoding changed")
    staged_receipt = stage_root / "receipt.json"
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _write_bytes_fsync(
        staged_receipt, receipt_bytes,
        label="staged-receipt-write", crash_injector=crash_injector,
    )
    validate_result(result, artifact_path_resolver=_staging_resolver(stage_root))
    validate_receipt(receipt, result, result_file=staged_result)
    _fsync_directory(stage_root)
    publish_staged_package(stage_root, crash_injector=crash_injector)
    return receipt


def run_science() -> dict[str, object]:
    """Execute the exact frozen FIT-first experiment when explicitly requested."""
    started = time.time()
    recover_stale_publication()
    execution = build_execution_authority()
    torch, functional, facade, induction = _load_runtime_modules()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True
    )
    if checkpoint.weights_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash changed")
    endpoint_specs = {(row["split"], row["endpoint_id"]): row for row in execution["endpoints"]}
    all_factors, all_replay, all_native = {}, {}, {}
    all_records, all_vectors = [], []
    calls = 0
    split_scores = {}
    fit_scales = None
    fit_failures = {name: [] for name in (
        "invalid_instrument", "native_denominator_or_scale_null", "factor_capacity_null",
        "factorization_not_identified", "insufficient_active_controls",
        "broad_contextual_equality_write",
    )}
    select_failures = {}
    evaluated = []
    structural_evidence = []
    realized_operation_evidence = {}
    instrument_maxima = {
        "native_attention_reconstruction_max_abs": 0.0,
        "equality_factor_max_abs": 0.0,
        "equality_plus_independent_remainder_max_abs": 0.0,
        "replay_native_logit_max_abs": 0.0,
        "padding_tripwire_active_lengths": [],
    }

    for split in SPLITS:
        if split == "SELECT" and any(fit_failures.values()):
            break
        evaluated.append(split)
        schedules = endpoint_schedules(execution, split)
        split_endpoints = [row for row in execution["endpoints"] if row["split"] == split]
        (
            factors, replay, replay_padding, capture_calls, instrument,
            realized_operations, capture_exactness,
        ) = collect_capture_replay(
            model, schedules["capture"], torch=torch, functional=functional,
            facade=facade, induction=induction,
        )
        realized_operation_evidence[split] = validate_realized_operations(
            execution["endpoint_site_role_operations"], realized_operations, split
        )
        native, native_padding, comparator_calls = collect_native_comparator(
            model, schedules["comparator"], torch=torch, facade=facade
        )
        calls += capture_calls + comparator_calls
        comparator_failures, comparator_exactness = capture_instrument_failures(
            replay, native, replay_padding, native_padding, split_endpoints
        )
        instrument += comparator_failures
        for key, value in capture_exactness.items():
            instrument_maxima[key] = max(float(instrument_maxima[key]), float(value))
        instrument_maxima["replay_native_logit_max_abs"] = max(
            float(instrument_maxima["replay_native_logit_max_abs"]),
            float(comparator_exactness["replay_native_logit_max_abs"]),
        )
        instrument_maxima["padding_tripwire_active_lengths"] = sorted(set(
            instrument_maxima["padding_tripwire_active_lengths"]
            + comparator_exactness["padding_tripwire_active_lengths"]
        ))
        all_factors.update(factors)
        all_replay.update(replay)
        all_native.update(native)
        current_failures = {name: [] for name in fit_failures}
        current_failures["invalid_instrument"].extend(instrument)
        split_records, split_vectors = [], []
        if not instrument:
            batches = direction_batches(execution, split)
            split_directions = [row for row in execution["directions"] if row["split"] == split]
            frozen_insertions, freeze_failures = build_frozen_insertion_cache(
                split_directions, all_factors, torch=torch
            )
            current_failures["invalid_instrument"].extend(freeze_failures)
            for arm in ARMS:
                records, vectors, arm_calls = collect_intervention_arm(
                    model, batches, arm, endpoint_specs, frozen_insertions, all_replay, all_native,
                    torch=torch, functional=functional, facade=facade, induction=induction,
                )
                split_records.extend(records)
                split_vectors.extend(vectors)
                calls += arm_calls
            current_failures["invalid_instrument"].extend(
                validate_primitive_logit_identities(split_records)
            )
            structural, evidence = structural_identity_failures(
                split_records, split_vectors, execution["manifests"], all_replay
            )
            current_failures["invalid_instrument"].extend(structural)
            structural_evidence.extend(evidence)
            for vector_row in split_vectors:
                vector_row.pop("full_logits", None)
            if not current_failures["invalid_instrument"]:
                if split == "FIT":
                    fit_scales = compute_fit_scales(split_records, execution["manifests"])
                assert fit_scales is not None
                report, scientific = score_split(
                    split_records, split, execution["manifests"], fit_scales
                )
                split_scores[split] = report
                for key, values in scientific.items():
                    current_failures[key].extend(values)
        all_records.extend(split_records)
        all_vectors.extend(split_vectors)
        if split == "FIT":
            fit_failures = current_failures
        else:
            select_failures = _prefix_select_failures(current_failures)

    failure_classes = _merge_failure_classes(fit_failures, select_failures)
    terminal = terminal_from_failures(evaluated, failure_classes)
    expected_calls = EXPECTED_PHASE_PRICE["FIT"] + (
        EXPECTED_PHASE_PRICE["SELECT"] if evaluated == ["FIT", "SELECT"] else 0
    )
    if not failure_classes.get("invalid_instrument") and not failure_classes.get("select_invalid_instrument") \
            and calls != expected_calls:
        raise RuntimeError(f"scientific forward price changed: {calls} != {expected_calls}")
    require_finite_json(all_records, "all_records")
    require_finite_json(structural_evidence, "structural_evidence")
    require_finite_json(instrument_maxima, "instrument_maxima")
    if any(
        clause.startswith("nonfinite_")
        for values in failure_classes.values() for clause in values
    ):
        raise RuntimeError("nonfinite scientific evidence rejected before publication")
    stage_root = create_stage_root()
    evidence_files = write_evidence(
        execution, all_factors, all_records, all_vectors, all_replay, all_native,
        evidence_dir=stage_root / "evidence",
    )
    failed = [clause for key in sorted(failure_classes) for clause in failure_classes[key]]
    result = {
        "schema": RESULT_SCHEMA,
        "rung": 585,
        "stage": "prospective_frozen_selector_payload_factor_intervention",
        "evidence_level": "prospective_identification_screen",
        "threshold_status": PROSPECTIVE_STATUS,
        "instrument_passes": not any(
            failure_classes.get(key) for key in ("invalid_instrument", "select_invalid_instrument")
        ),
        "terminal": terminal,
        "failed_clauses": failed,
        "failure_classes": failure_classes,
        "split_scores": split_scores,
        "raw_evidence": {
            "schema": EVIDENCE_SCHEMA,
            "endpoint_count": len(all_replay),
            "directed_arm_record_count": len(all_records),
            "endpoint_site_role_operation_counts": EXPECTED_OPERATION_COUNTS,
            "endpoint_site_role_operation_sha256": execution[
                "endpoint_site_role_operation_sha256"
            ],
            "realized_endpoint_site_role_operations": realized_operation_evidence,
            "instrument_maxima": instrument_maxima,
            "structural_identity_checks": structural_evidence,
            "endpoint_manifest_sha256": execution["endpoint_manifest_sha256"],
            "direction_manifest_sha256": execution["direction_manifest_sha256"],
            "target_cell_ids_sha256": execution["target_cell_ids_sha256"],
            "control_cell_ids_sha256": execution["control_cell_ids_sha256"],
            "coverage_key_sha256": execution["coverage_key_sha256"],
            "structural_identity_sha256": execution["structural_identity_sha256"],
            "bootstrap_cell_ids_sha256": execution["bootstrap_cell_ids_sha256"],
            "control_scale_lookup_sha256": execution["control_scale_lookup_sha256"],
            "fit_scales": fit_scales or {},
        },
        "evidence_files": evidence_files,
        "model_forwards": calls,
        "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "source_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "dependency_lock_sha256": AUTHORITY_HASHES[DEPENDENCY_LOCK],
        "evaluated_splits": evaluated,
        "forbidden_splits_opened": [],
        "elapsed_seconds": float(time.time() - started),
        "next_step": (
            "independent_cpu_audit_then_translation_removal_and_ood_preregistration"
            if terminal == "held_operational_selector_payload_factorization"
            else "preserve_terminal_and_do_not_search_sites_or_thresholds"
        ),
    }
    _finish_result(result, stage_root)
    return result


def run_dryrun() -> dict[str, object]:
    verify_authorities()
    execution = build_execution_authority()
    for split in SPLITS:
        endpoint_schedules(execution, split)
        direction_batches(execution, split)
    planted = planted_intervention_records(execution)
    scales = compute_fit_scales(planted, execution["manifests"])
    fit_report, fit_failures = score_split(
        planted, "FIT", execution["manifests"], scales, replicates=16
    )
    select_report, select_failures = score_split(
        planted, "SELECT", execution["manifests"], scales, replicates=16
    )
    if any(fit_failures.values()) or any(select_failures.values()):
        raise RuntimeError("planted held fixture failed")
    planted_null = planted_intervention_records(execution, null=True)
    null_scales = compute_fit_scales(planted_null, execution["manifests"])
    _, null_failures = score_split(
        planted_null, "FIT", execution["manifests"], null_scales, replicates=16
    )
    if not null_failures["factor_capacity_null"]:
        raise RuntimeError("planted scientific null did not fail capacity")
    held = make_result_fixture("held_operational_selector_payload_factorization")
    null = make_result_fixture("factor_capacity_null")
    invalid = make_result_fixture("invalid_instrument")
    for fixture in (held, null, invalid):
        validate_receipt(make_receipt_fixture(fixture), fixture)
    try:
        validate_result(held)
    except ValueError as held_fixture_error:
        if "raw evidence" not in str(held_fixture_error):
            raise
    else:
        raise RuntimeError("held fixture without evidence was accepted")
    manifest = load_manifest()
    held_lock, held_hashes = manifest.build_planted_dependency_fixture(True)
    null_lock, null_hashes = manifest.build_planted_dependency_fixture(False)
    dependency_checks = {
        "held": manifest.validate_dependency_lock(held_lock, held_hashes),
        "null": manifest.validate_dependency_lock(null_lock, null_hashes),
    }
    dryrun = {
        "schema": DRYRUN_SCHEMA,
        "status": "deterministic_cpu_dryrun_passed",
        "model_loaded": False,
        "cuda_opened": False,
        "outcomes_opened": [],
        "upstream_dependency_records_parsed": [
            str(R586_RESULT), str(R586_RECEIPT), str(R587_AUDIT)
        ],
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "source_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "canonical_names": {
            "sites": list(TERM_NAMES), "roles": list(ROLES), "arms": list(ARMS),
            "splits": list(SPLITS), "forbidden_splits": list(FORBIDDEN_SPLITS),
        },
        "census": {
            "endpoints": {split: sum(row["split"] == split for row in execution["endpoints"]) for split in SPLITS},
            "directions": {split: sum(row["split"] == split for row in execution["directions"]) for split in SPLITS},
            "endpoint_site_role_operations": {
                split: sum(
                    row["split"] == split
                    for row in execution["endpoint_site_role_operations"]
                ) for split in SPLITS
            },
            "target_cells_per_split": 20,
            "control_cells_per_split": 32,
            "coverage_keys_per_split": 24,
            "eligible_control_arm_cells_per_split": 88,
            "bootstrap_cells_per_split": 124,
        },
        "manifest_hashes": {key: execution[key] for key in (
            "endpoint_manifest_sha256", "direction_manifest_sha256", "target_cell_ids_sha256",
            "endpoint_site_role_operation_sha256",
            "control_cell_ids_sha256", "coverage_key_sha256", "structural_identity_sha256",
            "bootstrap_cell_ids_sha256",
            "control_scale_lookup_sha256",
        )},
        "price": {"FIT": 459, "SELECT": 231, "maximum": 690, "backwards": 0, "updates": 0},
        "evidence_contract": {
            "array_shapes": HELD_ARRAY_SHAPES,
            "jsonl_counts": HELD_JSONL_COUNTS,
            "checkpoint_weights_sha256": CHECKPOINT_SHA256,
            "independent_remainder": "contract_without_induction_fetch",
            "finite_before_final_write": True,
        },
        "publication_contract": {
            "stage_prefix": STAGE_PREFIX,
            "recovery_prefix": RECOVERY_PREFIX,
            "same_filesystem": True,
            "atomic_renames": ["evidence", "result", "receipt"],
            "receipt_is_commit_marker": True,
            "incomplete_paths_quarantined_without_deletion": True,
        },
        "dependency_checks": dependency_checks,
        "planted_terminals": {
            "held": held["terminal"], "scientific_null": null["terminal"],
            "instrument_failure": invalid["terminal"],
        },
        "planted_score_hashes": {
            "FIT": content_sha256(fit_report), "SELECT": content_sha256(select_report),
        },
        "fixture_bootstrap_replicates": 16,
        "production_bootstrap_replicates": BOOTSTRAPS,
    }
    require_finite_json(dryrun)
    DRYRUN.write_text(json.dumps(dryrun, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return dryrun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-science", action="store_true")
    args = parser.parse_args()
    # ops/enqueue.sh invokes every candidate with this environment variable and
    # no command-line arguments.  Treat that path as the model-free dry run;
    # scientific execution still requires the explicit flag.
    if os.environ.get("BQLIB_DRYRUN") == "1" and not args.execute_science:
        args.dry_run = True
    if args.dry_run == args.execute_science:
        parser.error("choose exactly one of --dry-run or --execute-science")
    payload = run_dryrun() if args.dry_run else run_science()
    print(json.dumps({
        "schema": payload["schema"],
        "terminal": payload.get("terminal"),
        "status": payload.get("status"),
        "model_forwards": payload.get("model_forwards", 0),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
