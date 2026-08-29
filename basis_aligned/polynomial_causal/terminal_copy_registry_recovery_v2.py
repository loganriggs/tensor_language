"""Fail-closed registry replay for the terminal-copy v2 row transaction.

The sole relaxation from the ordinary registry census is a missing row output named
by its own authority when the matching terminal failure proves that the transaction
published no rows, manifest, or receipt.  Every other missing row-like reference is
still fatal.  Both the authority and failure bytes are bound into the returned ledger.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import prepare_block3_native_down_behavioral_port_v1_rows as natural
import freeze_gauge_transport_triangle_unique_rows_v1 as failed_transaction


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = natural.REGISTRY
EXPECTED_AUTHORITY = (
    ROOT / "basis_aligned/polynomial_causal/"
    "gauge_transport_triangle_unique_rows_v1_authority.json"
).resolve()
EXPECTED_AUTHORITY_SHA256 = "5f7435150561ef385c9a4ee51e2040c4a029e98faefbfe1bc0f92612d820498e"
EXPECTED_INTERNAL_AUTHORITY_SHA256 = "8901a7446f70358e7e058013bb81c72f477c8636f5a1f76088307eda437025b5"
EXPECTED_FAILURE_SHA256 = "91859b52b55b8be8ac05dc61f26b95fd43cdb92db7b8c39dfa72d226df41eb58"
EXPECTED_V1_FAILURE = (
    ROOT / "basis_aligned/bilinear_quotient/terminal_copy_induction_v1_rows_failure.json"
).resolve()
EXPECTED_V1_FAILURE_SHA256 = "0f3908b43678c4acd0f4d24d6188d1ac6047085a50065e3632aa44f6f568938f"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json(path: Path) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"registry JSON changed while reading: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"registry JSON is not an object: {path}")
    return payload, before


def _resolve(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("failed authority output path is malformed")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _looks_row_like(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith((".pt", ".pth")):
        return False
    path = Path(value)
    name = path.name.lower()
    return any(part.startswith(".rowcache") for part in path.parts) or any(
        term in name for term in (
            "row", "fineweb", "eval_token", "oracle_corpus", "source_document",
        )
    )


def _missing_row_paths(payload: Any) -> set[Path]:
    missing: set[Path] = set()
    for value in REGISTRY.walk_json(payload):
        if _looks_row_like(value):
            path = _resolve(value)
            if not path.is_file():
                missing.add(path)
    return missing


def _replace_path(value: Any, target: Path) -> Any:
    if isinstance(value, dict):
        return {key: _replace_path(child, target) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_path(child, target) for child in value]
    if _looks_row_like(value) and _resolve(value) == target:
        return None
    return value


def validate_failed_unmaterialized_authority(
    authority_path: Path, payload: Mapping[str, Any], missing_row: Path,
    *, expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an exact waiver ledger or reject the missing row reference."""

    authority_path = authority_path.resolve()
    expected = dict(expected_identity or {
        "authority_path": str(EXPECTED_AUTHORITY),
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "internal_authority_sha256": EXPECTED_INTERNAL_AUTHORITY_SHA256,
        "failure_sha256": EXPECTED_FAILURE_SHA256,
    })
    if str(authority_path) != expected.get("authority_path") or (
        file_sha256(authority_path) != expected.get("authority_sha256")
    ) or payload.get("authority_sha256") != expected.get("internal_authority_sha256"):
        raise RuntimeError("missing row authority is not the preregistered exact authority")
    if not authority_path.name.endswith("_authority.json"):
        raise RuntimeError("missing row reference is not owned by its own authority")
    outputs = payload.get("outputs")
    schema = payload.get("schema")
    if not isinstance(outputs, Mapping) or not isinstance(schema, str) or not schema.endswith(
        "_authority"
    ):
        raise RuntimeError("missing row authority schema is malformed")
    required = {"authority", "failure", "lock", "manifest", "receipt", "rows"}
    if set(outputs) != required:
        raise RuntimeError("missing row authority has incomplete output closure")
    resolved = {name: _resolve(outputs[name]) for name in required}
    if resolved["authority"] != authority_path or resolved["rows"] != missing_row.resolve():
        raise RuntimeError("missing row is not the exact output of this authority")
    failure_path = resolved["failure"]
    if not failure_path.is_file():
        raise RuntimeError("missing row authority has no terminal failure artifact")
    failure, failure_sha = _stable_json(failure_path)
    expected_failure_schema = schema.removesuffix("_authority") + "_failure"
    if failure_sha != expected.get("failure_sha256") or set(failure) != {
        "schema", "status", "exception_type", "exception_message",
        "rows_exists", "manifest_exists", "receipt_exists",
    } or (
        failure.get("schema") != expected_failure_schema
        or failure.get("status") != "terminal_failure_no_receipt"
        or failure.get("rows_exists") is not False
        or failure.get("manifest_exists") is not False
        or failure.get("receipt_exists") is not False
    ):
        raise RuntimeError("terminal failure does not prove an unmaterialized row transaction")
    for kind in ("rows", "manifest", "receipt"):
        if resolved[kind].exists():
            raise RuntimeError(f"failed authority unexpectedly materialized {kind}")
    if resolved["lock"].exists():
        raise RuntimeError("failed authority still has an owner lock")
    staging = sorted({
        str(candidate.resolve())
        for kind in ("rows", "manifest", "receipt", "failure")
        for candidate in resolved[kind].parent.glob(f".{resolved[kind].name}.tmp-*")
    })
    if staging:
        raise RuntimeError("failed authority still has staging artifacts")
    if resolved["failure"] == resolved["rows"] or resolved["failure"] == authority_path:
        raise RuntimeError("failed authority output paths alias one another")
    return {
        "kind": "failed_unmaterialized_registry",
        "authority_path": str(authority_path),
        "authority_sha256": file_sha256(authority_path),
        "failure_path": str(failure_path),
        "failure_sha256": failure_sha,
        "internal_authority_sha256": payload["authority_sha256"],
        "authority_schema": schema,
        "authority_status": payload.get("status"),
        "source_closure": payload.get("source_closure"),
        "json_pointer": "$.outputs.rows",
        "declared_outputs": {key: str(resolved[key]) for key in sorted(resolved)},
        "omitted_missing_row_path": str(missing_row.resolve()),
        "absent_manifest_path": str(resolved["manifest"]),
        "absent_receipt_path": str(resolved["receipt"]),
        "proof": {
            "status": failure["status"],
            "rows_exists": False,
            "manifest_exists": False,
            "receipt_exists": False,
            "lock_exists": False,
            "staging_artifacts": [],
        },
        "exclusion_scope": "one_reference_only",
    }


def load_registry_exclusions(
    registry_files: tuple[Path, ...],
) -> tuple[
    tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    dict[str, str], dict[str, str], list[dict[str, Any]],
]:
    documents: set[str] = set()
    indices: set[int] = set()
    full_rows: set[tuple[int, ...]] = set()
    prefixes: set[tuple[int, ...]] = set()
    registry_hashes: dict[str, str] = {}
    tensor_specs: dict[Path, list[dict[str, str]]] = {natural.REFERENCE_ROWS.resolve(): []}
    waivers: list[dict[str, Any]] = []

    for path in registry_files:
        path = path.resolve()
        payload, digest = _stable_json(path)
        registry_hashes[str(path)] = digest
        sanitized: Any = copy.deepcopy(payload)
        missing = _missing_row_paths(payload)
        if missing and (path != EXPECTED_AUTHORITY or missing != {_resolve(payload.get(
            "outputs", {}
        ).get("rows"))}):
            raise RuntimeError("registry has an unregistered missing row-like reference")
        for missing_row in sorted(missing):
            failed_transaction.validate_authority(payload)
            waiver = validate_failed_unmaterialized_authority(path, payload, missing_row)
            waivers.append(waiver)
            sanitized = _replace_path(sanitized, missing_row)
        for tensor_path, specifications in REGISTRY.referenced_row_specs(sanitized).items():
            tensor_specs.setdefault(tensor_path, []).extend(specifications)
        for value in REGISTRY.walk_json(sanitized):
            if not isinstance(value, dict):
                continue
            document = value.get("document_id")
            index = value.get("dataset_document_index")
            if isinstance(document, str) and document:
                documents.add(document)
            if isinstance(index, int):
                indices.add(index)

    tensor_hashes: dict[str, str] = {}
    for path in sorted(tensor_specs):
        tensors, digest = REGISTRY.load_verified_row_tensor(path, tensor_specs[path])
        tensor_hashes[str(path)] = digest
        for tensor in tensors:
            for row in tensor:
                values = tuple(int(item) for item in row.tolist())
                full_rows.add(values)
                prefixes.add(values[: natural.PREFIX_LENGTH])
    waivers.sort(key=lambda item: (item["authority_path"], item["omitted_missing_row_path"]))
    if len(waivers) != 1:
        raise RuntimeError("v2 expected exactly one preregistered failed-authority exclusion")
    if not EXPECTED_V1_FAILURE.is_file() or file_sha256(
        EXPECTED_V1_FAILURE
    ) != EXPECTED_V1_FAILURE_SHA256:
        raise RuntimeError("terminal-copy v1 failure lineage changed")
    waivers[0]["terminal_copy_v1_failure_path"] = str(EXPECTED_V1_FAILURE)
    waivers[0]["terminal_copy_v1_failure_sha256"] = EXPECTED_V1_FAILURE_SHA256
    return (documents, indices, full_rows, prefixes), registry_hashes, tensor_hashes, waivers


def verify_snapshot(
    *, commit: str, sources: Mapping[str, str], registry_files: tuple[Path, ...],
    registry_hashes: Mapping[str, str], tensor_hashes: Mapping[str, str],
    prior: Any, parquet: Path, registry_waivers: Sequence[Mapping[str, Any]],
) -> None:
    if natural.source_closure(commit) != dict(sources):
        raise RuntimeError("fresh-row source closure changed")
    current_registry = natural.discover_registry_files()
    if current_registry != registry_files:
        raise RuntimeError("fresh-row registry membership changed")
    current_prior, current_hashes, current_tensor_hashes, current_waivers = (
        load_registry_exclusions(current_registry)
    )
    if natural.discover_registry_files() != current_registry:
        raise RuntimeError("fresh-row registry membership changed during replay")
    if current_hashes != dict(registry_hashes):
        raise RuntimeError("fresh-row registry files changed")
    if current_tensor_hashes != dict(tensor_hashes) or current_prior != prior:
        raise RuntimeError("fresh-row exclusion tensors changed")
    if current_waivers != list(registry_waivers):
        raise RuntimeError("failed-unmaterialized registry proof changed")
    if parquet.stat().st_size != natural.BASE.local.PINNED_SIZE or (
        file_sha256(parquet) != natural.BASE.local.PINNED_SHA256
    ):
        raise RuntimeError("pinned ordered FineWeb parquet changed")
