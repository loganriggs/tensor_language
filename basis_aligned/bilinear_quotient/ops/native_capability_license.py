"""Fail-closed staged native-capability licenses for later causal experiments.

The native-only stage owns no causal interventions.  It records one frozen
authority and a complete set of registered capability cells.  A deterministic
license can be created only from a passing immutable result.  A causal runner
calls :func:`validate_causal_preflight` before enqueue or model loading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from circuit_managed_entry import ManagedEntryError, _safe_open_bytes
import circuit_experiment_spec as framework
import circuit_fast_screen_managed_runner as managed


SHA256 = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class CapabilityLicenseError(ValueError):
    """A native capability result or causal license failed closed."""


@dataclass(frozen=True)
class CapabilityCell:
    cell_id: str
    expected_count: int
    minimum_accuracy: float


@dataclass(frozen=True)
class CapabilityGate:
    capability_id: str
    authority_path: Path
    expected_authority_file_sha256: str
    authority_logical_sha256: str
    cells: tuple[CapabilityCell, ...]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise CapabilityLicenseError(f"{label} must be a lowercase SHA-256")
    return value


def _read(path: Path, label: str) -> bytes:
    try:
        return _safe_open_bytes(path)
    except (OSError, ManagedEntryError) as error:
        raise CapabilityLicenseError(f"cannot safely read {label}: {path}") from error


def _json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    data = _read(path, label)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapabilityLicenseError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CapabilityLicenseError(f"{label} must be a JSON object")
    return value, data


def _cell_payload(gate: CapabilityGate) -> list[dict[str, object]]:
    return [asdict(cell) for cell in gate.cells]


def cells_sha256(gate: CapabilityGate) -> str:
    return framework.canonical_sha256(_cell_payload(gate))


def validate_gate(gate: CapabilityGate) -> str:
    if not isinstance(gate.capability_id, str) or not IDENTIFIER.fullmatch(gate.capability_id):
        raise CapabilityLicenseError("capability_id is invalid")
    if not isinstance(gate.authority_path, Path):
        raise CapabilityLicenseError("authority_path must be a Path")
    _require_sha(gate.expected_authority_file_sha256, "authority file hash")
    _require_sha(gate.authority_logical_sha256, "authority logical hash")
    if not gate.cells or len({cell.cell_id for cell in gate.cells}) != len(gate.cells):
        raise CapabilityLicenseError("registered capability cells must be nonempty and unique")
    for cell in gate.cells:
        if not isinstance(cell.cell_id, str) or not cell.cell_id \
                or type(cell.expected_count) is not int or cell.expected_count <= 0 \
                or type(cell.minimum_accuracy) not in (int, float) \
                or not math.isfinite(cell.minimum_accuracy) \
                or not 0.0 <= cell.minimum_accuracy <= 1.0:
            raise CapabilityLicenseError("registered capability cell is invalid")
    observed = _hash_bytes(_read(gate.authority_path, "authority"))
    if observed != gate.expected_authority_file_sha256:
        raise CapabilityLicenseError("authority file hash changed")
    return observed


def _summarize(gate: CapabilityGate, evidence: Sequence[Mapping[str, object]]):
    if not isinstance(evidence, (list, tuple)):
        raise CapabilityLicenseError("native evidence must be a list or tuple")
    expected = {cell.cell_id: cell for cell in gate.cells}
    grouped: dict[str, list[Mapping[str, object]]] = {cell_id: [] for cell_id in expected}
    example_ids = []
    normalized = []
    for raw in evidence:
        if set(raw) != {"example_id", "cell_id", "correct", "full_vocab_CE",
                       "answer_minus_foil_margin"}:
            raise CapabilityLicenseError("native evidence has unknown or missing fields")
        item = dict(raw)
        cell_id, example_id = item["cell_id"], item["example_id"]
        if cell_id not in expected or not isinstance(example_id, str) or not example_id:
            raise CapabilityLicenseError("native evidence is outside the registered cells")
        if type(item["correct"]) is not bool:
            raise CapabilityLicenseError("native correctness must be boolean")
        for key in ("full_vocab_CE", "answer_minus_foil_margin"):
            if type(item[key]) not in (int, float) or not math.isfinite(float(item[key])):
                raise CapabilityLicenseError("native metric must be finite")
        example_ids.append(example_id)
        grouped[str(cell_id)].append(item)
        normalized.append(item)
    if len(example_ids) != len(set(example_ids)):
        raise CapabilityLicenseError("native evidence example IDs are duplicated")
    summaries, passed = {}, True
    for cell in gate.cells:
        items = grouped[cell.cell_id]
        if len(items) != cell.expected_count:
            raise CapabilityLicenseError(
                f"cell {cell.cell_id} has {len(items)} rows, expected {cell.expected_count}")
        accuracy = sum(bool(item["correct"]) for item in items) / len(items)
        cell_passed = accuracy >= cell.minimum_accuracy
        summaries[cell.cell_id] = {
            "count": len(items), "accuracy": accuracy,
            "minimum_accuracy": cell.minimum_accuracy, "passed": cell_passed,
            "mean_full_vocab_CE": sum(float(x["full_vocab_CE"]) for x in items) / len(items),
            "mean_answer_minus_foil_margin": sum(
                float(x["answer_minus_foil_margin"]) for x in items) / len(items),
        }
        passed &= cell_passed
    return normalized, summaries, passed


def finalize_native_capability(
    gate: CapabilityGate,
    evidence: Sequence[Mapping[str, object]],
    result_path: Path,
) -> tuple[dict[str, object], str]:
    """Validate and atomically finalize one frozen native-only capability result."""
    authority_hash = validate_gate(gate)
    normalized, summaries, passed = _summarize(gate, evidence)
    result = {
        "schema": "native_capability_result_v1",
        "capability_id": gate.capability_id,
        "terminal": "pass" if passed else "fail",
        "native_only": True,
        "causal_interventions": 0,
        "authority_path": str(gate.authority_path),
        "authority_file_sha256": authority_hash,
        "authority_logical_sha256": gate.authority_logical_sha256,
        "registered_cells": _cell_payload(gate),
        "registered_cells_sha256": cells_sha256(gate),
        "cells": summaries,
        "evidence": normalized,
    }
    payload = managed.atomic_create_json(result_path, result)
    return result, _hash_bytes(payload)


def evaluate_and_finalize_native_capability(
    gate: CapabilityGate,
    evaluator: Callable[[], Sequence[Mapping[str, object]]],
    result_path: Path,
) -> tuple[dict[str, object], str]:
    """Verify the authority before invoking a native-only evaluator, then finalize."""
    validate_gate(gate)
    return finalize_native_capability(gate, evaluator(), result_path)


def validate_native_capability_result(
    gate: CapabilityGate, result: Mapping[str, object]
) -> bool:
    """Recompute every registered cell; never trust a stored terminal label alone."""
    expected_fields = {
        "schema", "capability_id", "terminal", "native_only", "causal_interventions",
        "authority_path", "authority_file_sha256", "authority_logical_sha256",
        "registered_cells", "registered_cells_sha256", "cells", "evidence",
    }
    if set(result) != expected_fields:
        raise CapabilityLicenseError("capability result has unknown or missing fields")
    bindings = {
        "schema": "native_capability_result_v1",
        "capability_id": gate.capability_id,
        "native_only": True,
        "causal_interventions": 0,
        "authority_path": str(gate.authority_path),
        "authority_file_sha256": gate.expected_authority_file_sha256,
        "authority_logical_sha256": gate.authority_logical_sha256,
        "registered_cells": _cell_payload(gate),
        "registered_cells_sha256": cells_sha256(gate),
    }
    if any(result.get(key) != value for key, value in bindings.items()):
        raise CapabilityLicenseError("capability result bindings are mismatched")
    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        raise CapabilityLicenseError("capability result evidence must be a list")
    normalized, summaries, passed = _summarize(gate, evidence)
    if normalized != evidence or summaries != result.get("cells") \
            or result.get("terminal") != ("pass" if passed else "fail"):
        raise CapabilityLicenseError("capability result summary or terminal is not reproducible")
    return passed


def issue_capability_license(
    gate: CapabilityGate,
    result_path: Path,
    license_path: Path,
    *,
    causal_candidate_id: str,
) -> tuple[dict[str, object], str]:
    """Atomically emit a deterministic license, but only for a complete passing result."""
    validate_gate(gate)
    result, result_bytes = _json(result_path, "capability result")
    if not validate_native_capability_result(gate, result):
        raise CapabilityLicenseError("capability result is failed, incomplete, or mismatched")
    if not isinstance(causal_candidate_id, str) or not IDENTIFIER.fullmatch(causal_candidate_id):
        raise CapabilityLicenseError("causal candidate ID is invalid")
    license_value = {
        "schema": "native_capability_license_v1",
        "capability_id": gate.capability_id,
        "causal_candidate_id": causal_candidate_id,
        "authority_file_sha256": gate.expected_authority_file_sha256,
        "authority_logical_sha256": gate.authority_logical_sha256,
        "registered_cells_sha256": cells_sha256(gate),
        "capability_result_path": str(result_path),
        "capability_result_sha256": _hash_bytes(result_bytes),
        "decision": "pass",
    }
    payload = managed.atomic_create_json(license_path, license_value)
    return license_value, _hash_bytes(payload)


def validate_causal_preflight(
    gate: CapabilityGate,
    result_path: Path,
    license_path: Path,
    *,
    expected_license_sha256: str,
    causal_candidate_id: str,
) -> dict[str, object]:
    """Fail closed before a causal runner is enqueued or loads a model."""
    validate_gate(gate)
    if not isinstance(causal_candidate_id, str) or not IDENTIFIER.fullmatch(causal_candidate_id):
        raise CapabilityLicenseError("causal candidate ID is invalid")
    _require_sha(expected_license_sha256, "expected license hash")
    license_value, license_bytes = _json(license_path, "capability license")
    if _hash_bytes(license_bytes) != expected_license_sha256:
        raise CapabilityLicenseError("capability license hash changed")
    result, result_bytes = _json(result_path, "capability result")
    expected_license = {
        "schema": "native_capability_license_v1",
        "capability_id": gate.capability_id,
        "causal_candidate_id": causal_candidate_id,
        "authority_file_sha256": gate.expected_authority_file_sha256,
        "authority_logical_sha256": gate.authority_logical_sha256,
        "registered_cells_sha256": cells_sha256(gate),
        "capability_result_path": str(result_path),
        "capability_result_sha256": _hash_bytes(result_bytes),
        "decision": "pass",
    }
    if license_value != expected_license:
        raise CapabilityLicenseError("capability license bindings are incomplete or mismatched")
    if not validate_native_capability_result(gate, result):
        raise CapabilityLicenseError("licensed capability result no longer proves a pass")
    return license_value
