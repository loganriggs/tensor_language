"""Side-effect-free validation for saved experimental result contracts.

This module deliberately knows nothing about a particular circuit, intervention,
or model.  It validates the boring but essential boundary between a preregistered
experiment and the JSON result that later analysis will trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Mapping, Sequence


class ContractError(ValueError):
    """Raised when a result is not admissible under its declared contract."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELD_KINDS = {
    "scalar",
    "optional_scalar",
    "string",
    "optional_string",
    "boolean",
    "integer",
    "number",
    "list",
    "dict",
    "null",
}


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _strict_json_tree(value: object, path: str = "$") -> None:
    """Reject Python conveniences that are not literal JSON values."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{path}: non-finite number is not standard JSON")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _strict_json_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path}: JSON object key {key!r} is not a string")
            _strict_json_tree(item, f"{path}.{key}")
        return
    raise ContractError(f"{path}: {type(value).__name__} is not a literal JSON type")


def validate_standard_json(value: object) -> str:
    """Return canonical JSON, rejecting NaN/Infinity and non-JSON Python types."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"not finite standard JSON: {error}") from error
    _strict_json_tree(value)
    return encoded


def _field_path(path: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(path, str):
        parts = tuple(path.split("."))
    elif _is_sequence(path):
        parts = tuple(path)
    else:
        raise ContractError("field path must be a dotted string or sequence of strings")
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ContractError(f"invalid field path: {path!r}")
    return parts


def _resolve(value: object, path: str | Sequence[str]) -> object:
    parts = _field_path(path)
    current = value
    traversed: list[str] = []
    for part in parts:
        traversed.append(part)
        if not isinstance(current, Mapping) or part not in current:
            raise ContractError(f"missing required field: {'.'.join(traversed)}")
        current = current[part]
    return current


def _is_scalar(value: object) -> bool:
    return isinstance(value, (str, bool, int, float))


def _matches_kind(value: object, kind: str) -> bool:
    if kind == "scalar":
        return _is_scalar(value)
    if kind == "optional_scalar":
        return value is None or _is_scalar(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "optional_string":
        return value is None or isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "list":
        return isinstance(value, list)
    if kind == "dict":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return False


def validate_declared_types(
    payload: Mapping[str, object], declarations: Mapping[str, str]
) -> None:
    """Validate exact declared field shapes; dotted paths address nested fields."""
    for path, kind in declarations.items():
        if kind not in _FIELD_KINDS:
            raise ContractError(f"{path}: unknown declared field type {kind!r}")
        value = _resolve(payload, path)
        if not _matches_kind(value, kind):
            raise ContractError(
                f"{path}: expected declared {kind}, got {type(value).__name__}"
            )


def _string_tuple(name: str, values: Sequence[str], *, nonempty: bool = True) -> tuple[str, ...]:
    if not _is_sequence(values):
        raise ContractError(f"{name} must be a list or tuple of strings")
    result = tuple(values)
    if nonempty and not result:
        raise ContractError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in result):
        raise ContractError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ContractError(f"{name} contains duplicates")
    return result


def validate_split_closure(
    records: Sequence[Mapping[str, object]],
    *,
    declared_opened_splits: Sequence[str],
    expected_opened_splits: Sequence[str],
    allowed_splits: Sequence[str],
    forbidden_splits: Sequence[str] = (),
    split_field: str = "split",
) -> tuple[str, ...]:
    """Require the declared, preregistered, and observed splits to close exactly."""
    declared = _string_tuple("declared opened splits", declared_opened_splits)
    expected = _string_tuple("expected opened splits", expected_opened_splits)
    allowed = set(_string_tuple("allowed splits", allowed_splits))
    forbidden = set(_string_tuple("forbidden splits", forbidden_splits, nonempty=False))
    if declared != expected:
        raise ContractError(
            f"opened split declaration mismatch: declared={declared}, expected={expected}"
        )
    if not set(expected) <= allowed:
        raise ContractError("expected opened splits are not a subset of allowed splits")
    if set(expected) & forbidden:
        raise ContractError("an expected opened split is also forbidden")
    observed: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractError(f"evidence record {index} is not a mapping")
        split = record.get(split_field)
        if not isinstance(split, str) or not split:
            raise ContractError(f"evidence record {index} has invalid {split_field}")
        observed.add(split)
    if observed != set(expected):
        raise ContractError(
            f"observed split closure mismatch: observed={sorted(observed)}, "
            f"expected={sorted(expected)}"
        )
    return expected


def _index_rows(
    records: Sequence[Mapping[str, object]],
    *,
    label: str,
    row_id_field: str,
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractError(f"{label} record {index} is not a mapping")
        row_id = record.get(row_id_field)
        if not isinstance(row_id, str) or not row_id:
            raise ContractError(f"{label} record {index} has invalid {row_id_field}")
        if row_id in indexed:
            raise ContractError(f"duplicate {label} row ID: {row_id}")
        indexed[row_id] = record
    return indexed


def validate_exact_membership(
    evidence_records: Sequence[Mapping[str, object]],
    authority_records: Sequence[Mapping[str, object]],
    *,
    opened_splits: Sequence[str],
    row_id_field: str = "row_id",
    split_field: str = "split",
    group_fields: Sequence[str] = ("group_id",),
) -> dict[str, int]:
    """Require exactly the authority rows and group assignments for opened splits."""
    opened = set(_string_tuple("opened splits", opened_splits))
    groups = _string_tuple("group fields", group_fields)
    authority_all = _index_rows(
        authority_records, label="authority", row_id_field=row_id_field
    )
    selected: dict[str, Mapping[str, object]] = {}
    for row_id, record in authority_all.items():
        split = record.get(split_field)
        if not isinstance(split, str) or not split:
            raise ContractError(f"authority row {row_id} has invalid {split_field}")
        for group_field in groups:
            group = record.get(group_field)
            if not isinstance(group, str) or not group:
                raise ContractError(f"authority row {row_id} has invalid {group_field}")
        if split in opened:
            selected[row_id] = record
    if not selected:
        raise ContractError("authority has no rows in the opened splits")
    evidence = _index_rows(evidence_records, label="evidence", row_id_field=row_id_field)
    missing = sorted(set(selected) - set(evidence))
    extra = sorted(set(evidence) - set(selected))
    if missing or extra:
        raise ContractError(
            f"exact row membership mismatch: missing={missing}, extra={extra}"
        )
    for row_id, observed in evidence.items():
        expected = selected[row_id]
        if observed.get(split_field) != expected.get(split_field):
            raise ContractError(f"row {row_id}: {split_field} disagrees with authority")
        for group_field in groups:
            if observed.get(group_field) != expected.get(group_field):
                raise ContractError(
                    f"row {row_id}: {group_field} disagrees with authority"
                )
    authority_group_keys = {
        tuple(record[group] for group in groups) for record in selected.values()
    }
    evidence_group_keys = {
        tuple(record[group] for group in groups) for record in evidence.values()
    }
    if evidence_group_keys != authority_group_keys:  # Defensive; row checks imply this.
        raise ContractError("exact group membership mismatch")
    return {"rows": len(selected), "groups": len(authority_group_keys)}


def _nonnegative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{field_name} must be a nonnegative integer")
    return value


def validate_execution_envelope(
    payload: Mapping[str, object],
    *,
    min_forwards: int,
    max_forwards: int,
    exact_forwards: int | None = None,
    expected_backwards: int = 0,
    expected_weights_updated: bool = False,
    forwards_field: str = "model_forwards",
    backwards_field: str = "model_backwards",
    weights_updated_field: str = "weights_updated",
) -> dict[str, object]:
    """Validate the declared model-call and mutation price."""
    lower = _nonnegative_int(min_forwards, "minimum forwards")
    upper = _nonnegative_int(max_forwards, "maximum forwards")
    backwards_expected = _nonnegative_int(expected_backwards, "expected backwards")
    if lower > upper:
        raise ContractError("minimum forwards exceeds maximum forwards")
    if exact_forwards is not None:
        exact = _nonnegative_int(exact_forwards, "exact forwards")
        if not lower <= exact <= upper:
            raise ContractError("exact forwards lies outside the forward envelope")
    forwards = _nonnegative_int(_resolve(payload, forwards_field), forwards_field)
    backwards = _nonnegative_int(_resolve(payload, backwards_field), backwards_field)
    updated = _resolve(payload, weights_updated_field)
    if not isinstance(updated, bool):
        raise ContractError(f"{weights_updated_field} must be a boolean")
    if not lower <= forwards <= upper:
        raise ContractError(
            f"{forwards_field}={forwards} outside declared envelope [{lower}, {upper}]"
        )
    if exact_forwards is not None and forwards != exact_forwards:
        raise ContractError(f"{forwards_field}={forwards}, expected exactly {exact_forwards}")
    if backwards != backwards_expected:
        raise ContractError(
            f"{backwards_field}={backwards}, expected {backwards_expected}"
        )
    if updated is not expected_weights_updated:
        raise ContractError(
            f"{weights_updated_field}={updated}, expected {expected_weights_updated}"
        )
    return {
        "model_forwards": forwards,
        "model_backwards": backwards,
        "weights_updated": updated,
    }


def validate_provenance_hashes(
    hashes: Mapping[str, object],
    *,
    required_keys: Sequence[str],
    expected_hashes: Mapping[str, str] | None = None,
    allow_extra: bool = True,
) -> tuple[str, ...]:
    """Require SHA-256-shaped values and optionally bind exact expected bytes."""
    if not isinstance(hashes, Mapping):
        raise ContractError("provenance hashes must be a mapping")
    required = set(_string_tuple("required provenance keys", required_keys, nonempty=False))
    expected = {} if expected_hashes is None else dict(expected_hashes)
    required |= set(expected)
    missing = sorted(required - set(hashes))
    if missing:
        raise ContractError(f"missing required provenance hashes: {missing}")
    if not allow_extra:
        extra = sorted(set(hashes) - required)
        if extra:
            raise ContractError(f"unexpected provenance hashes: {extra}")
    for key, digest in hashes.items():
        if not isinstance(key, str) or not key:
            raise ContractError("provenance hash keys must be nonempty strings")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ContractError(f"provenance hash {key!r} is not lowercase SHA-256")
    for key, digest in expected.items():
        if hashes[key] != digest:
            raise ContractError(f"provenance hash mismatch: {key}")
    return tuple(sorted(hashes))


@dataclass(frozen=True)
class ResultContract:
    """A circuit-independent contract for one flattened result evidence table."""

    opened_splits: tuple[str, ...]
    allowed_splits: tuple[str, ...]
    max_model_forwards: int
    field_types: Mapping[str, str] = field(default_factory=dict)
    forbidden_splits: tuple[str, ...] = ()
    min_model_forwards: int = 0
    exact_model_forwards: int | None = None
    expected_model_backwards: int = 0
    expected_weights_updated: bool = False
    required_provenance: tuple[str, ...] = ()
    expected_provenance: Mapping[str, str] = field(default_factory=dict)
    allow_extra_provenance: bool = True
    opened_splits_field: str = "evaluated_splits"
    provenance_field: str = "input_sha256"
    forwards_field: str = "model_forwards"
    backwards_field: str = "model_backwards"
    weights_updated_field: str = "weights_updated"
    row_id_field: str = "row_id"
    split_field: str = "split"
    group_fields: tuple[str, ...] = ("group_id",)


def validate_result_contract(
    payload: Mapping[str, object],
    evidence_records: Sequence[Mapping[str, object]],
    authority_records: Sequence[Mapping[str, object]],
    contract: ResultContract,
) -> dict[str, object]:
    """Validate a complete result boundary and return a small audit summary."""
    # Validate all three inputs, not only the summary.  This catches a non-finite
    # value hidden down a raw-evidence/null branch.
    payload_json = validate_standard_json(payload)
    validate_standard_json(evidence_records)
    validate_standard_json(authority_records)
    validate_declared_types(payload, contract.field_types)

    declared_splits = _resolve(payload, contract.opened_splits_field)
    if not isinstance(declared_splits, list):
        raise ContractError(f"{contract.opened_splits_field} must be a JSON list")
    opened = validate_split_closure(
        evidence_records,
        declared_opened_splits=declared_splits,
        expected_opened_splits=contract.opened_splits,
        allowed_splits=contract.allowed_splits,
        forbidden_splits=contract.forbidden_splits,
        split_field=contract.split_field,
    )
    membership = validate_exact_membership(
        evidence_records,
        authority_records,
        opened_splits=opened,
        row_id_field=contract.row_id_field,
        split_field=contract.split_field,
        group_fields=contract.group_fields,
    )
    execution = validate_execution_envelope(
        payload,
        min_forwards=contract.min_model_forwards,
        max_forwards=contract.max_model_forwards,
        exact_forwards=contract.exact_model_forwards,
        expected_backwards=contract.expected_model_backwards,
        expected_weights_updated=contract.expected_weights_updated,
        forwards_field=contract.forwards_field,
        backwards_field=contract.backwards_field,
        weights_updated_field=contract.weights_updated_field,
    )
    provenance = _resolve(payload, contract.provenance_field)
    if not isinstance(provenance, Mapping):
        raise ContractError(f"{contract.provenance_field} must be a mapping")
    provenance_keys = validate_provenance_hashes(
        provenance,
        required_keys=contract.required_provenance,
        expected_hashes=contract.expected_provenance,
        allow_extra=contract.allow_extra_provenance,
    )
    return {
        **membership,
        "opened_splits": list(opened),
        **execution,
        "provenance_keys": list(provenance_keys),
        "canonical_payload_bytes": len(payload_json.encode("utf-8")),
    }
