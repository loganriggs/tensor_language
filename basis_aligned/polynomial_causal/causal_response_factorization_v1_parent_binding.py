"""Outcome-blind parent binding for causal-response factorization v1.

This module hashes and validates the completed FIT transaction without importing
torch or deserializing the response bundle.  Its output is suitable for inclusion in
a later factor-analysis authority that must be frozen before any response value is
read.  It has no model, corpus, optimizer, or EVAL capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FitParentPaths:
    authority: Path
    bundle: Path
    manifest: Path
    receipt: Path
    failure: Path
    terminal: Path


PRODUCTION_PATHS = FitParentPaths(
    authority=HERE / "causal_response_tensor_v1_fit_authority.json",
    bundle=HERE / "causal_response_tensor_v1_fit_bundle.pt",
    manifest=HERE / "causal_response_tensor_v1_fit_manifest.json",
    receipt=HERE / "causal_response_tensor_v1_fit_receipt.json",
    failure=HERE / "causal_response_tensor_v1_fit_failure.json",
    terminal=HERE / "causal_response_tensor_v1_fit_terminal_claim.json",
)


def _logical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _stable_record(path: Path) -> tuple[dict[str, Any], bytes]:
    """Return the FIT lifecycle's exact regular-file record and stable bytes."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"FIT parent is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        first = b"".join(chunks)
        middle = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            second_chunks.append(chunk)
        second = b"".join(second_chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    verification = os.open(path, flags)
    try:
        verification_before = os.fstat(verification)
        verification_chunks: list[bytes] = []
        while chunk := os.read(verification, 8 << 20):
            verification_chunks.append(chunk)
        verification_raw = b"".join(verification_chunks)
        verification_after = os.fstat(verification)
    finally:
        os.close(verification)
    identity = (before.st_dev, before.st_ino)
    metadata = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if (
        (middle.st_dev, middle.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
        or (path_stat.st_dev, path_stat.st_ino) != identity
        or (verification_before.st_dev, verification_before.st_ino) != identity
        or (verification_after.st_dev, verification_after.st_ino) != identity
        or (middle.st_size, middle.st_mtime_ns, middle.st_ctime_ns) != metadata
        or (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != metadata
        or (path_stat.st_size, path_stat.st_mtime_ns, path_stat.st_ctime_ns)
        != metadata
        or (
            verification_before.st_size,
            verification_before.st_mtime_ns,
            verification_before.st_ctime_ns,
        ) != metadata
        or (
            verification_after.st_size,
            verification_after.st_mtime_ns,
            verification_after.st_ctime_ns,
        ) != metadata
        or first != second
        or first != verification_raw
    ):
        raise RuntimeError(f"FIT parent changed during stable read: {path}")
    raw = first
    if len(raw) != before.st_size:
        raise RuntimeError(f"FIT parent size changed during stable read: {path}")
    return ({
        "path": str(path),
        "present": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": before.st_size,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
    }, raw)


def _plain_json(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if type(value) is not dict:
        raise RuntimeError(f"{label} is not a plain JSON object")
    return value


def _validate_protocol(protocol: object) -> dict[str, Any]:
    if type(protocol) is not dict:
        raise RuntimeError("FIT parent protocol is not a plain object")
    expected = {
        "role": "FIT",
        "rows": 496,
        "source_documents": 343,
        "positions": 256,
        "sources": 49,
        "targets": 49,
        "phases": ["full", "residual"],
        "batch_size": 4,
        "batches": 124,
        "outer_forwards": 12_400,
        "projection_event_shape": [2, 49, 124],
        "capture_event_shape": [6, 124],
        "model_dtype": "torch.float32",
        "authorized_for_eval": False,
        "authorized_for_factor_selection": False,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise RuntimeError("FIT parent protocol differs from the frozen experiment")
    return protocol


def fit_parent_binding_without_tensor_load(
    paths: FitParentPaths = PRODUCTION_PATHS,
) -> dict[str, Any]:
    """Validate a successful FIT transaction and return an outcome-blind binding.

    The response bundle is read only as opaque bytes.  No tensor deserialization or
    scientific statistic is possible in this module.
    """

    if not isinstance(paths, FitParentPaths):
        raise TypeError("FIT parent paths must be an exact FitParentPaths value")
    if paths.failure.exists():
        raise RuntimeError("FIT parent ended in failure, not a successful receipt")
    terminal_record, terminal_raw = _stable_record(paths.terminal)
    receipt_record, receipt_raw = _stable_record(paths.receipt)
    if terminal_raw != receipt_raw or (
        terminal_record["device"], terminal_record["inode"]
    ) != (receipt_record["device"], receipt_record["inode"]):
        raise RuntimeError("FIT terminal and receipt are not the same published inode")
    terminal = _plain_json(terminal_raw, "FIT terminal receipt")
    if set(terminal) != {
        "schema", "kind", "authority_artifact_sha256",
        "authority_logical_sha256", "aggregate", "payload",
    } or terminal.get("schema") != "causal_response_tensor_v1_fit_terminal" or (
        terminal.get("kind") != "receipt"
    ):
        raise RuntimeError("FIT terminal is not the exact success-receipt schema")
    if not _is_sha256(terminal["authority_artifact_sha256"]) or not _is_sha256(
        terminal["authority_logical_sha256"]
    ):
        raise RuntimeError("FIT terminal authority hashes are malformed")

    aggregate = terminal["aggregate"]
    if type(aggregate) is not dict or set(aggregate) != {
        "authority", "bundle", "manifest"
    }:
        raise RuntimeError("FIT receipt aggregate schema changed")
    current: dict[str, dict[str, Any]] = {}
    raw_json: dict[str, bytes] = {}
    for name, path in (
        ("authority", paths.authority),
        ("bundle", paths.bundle),
        ("manifest", paths.manifest),
    ):
        record, raw = _stable_record(path)
        if aggregate[name] != record:
            raise RuntimeError(f"FIT receipt-bound {name} artifact changed")
        current[name] = record
        raw_json[name] = raw

    authority = _plain_json(raw_json["authority"], "FIT authority")
    if set(authority) != {
        "schema", "status", "source_closure", "independent_audit", "parents",
        "protocol", "output_paths", "outcome_access_before_authority",
        "authorized_for_fit_execution", "authorized_for_eval", "authority_sha256",
    } or authority.get("schema") != "causal_response_tensor_v1_fit_authority" or (
        authority.get("status")
        != "frozen_before_any_parent_tensor_or_bilin18_model_load"
        or authority.get("authorized_for_fit_execution") is not True
        or authority.get("authorized_for_eval") is not False
    ):
        raise RuntimeError("FIT authority semantics changed")
    authority_body = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    if authority.get("authority_sha256") != _logical_sha256(authority_body) or (
        current["authority"]["sha256"] != terminal["authority_artifact_sha256"]
        or authority["authority_sha256"] != terminal["authority_logical_sha256"]
    ):
        raise RuntimeError("FIT authority identity does not replay")
    protocol = _validate_protocol(authority["protocol"])
    expected_paths = {
        "authority": str(paths.authority),
        "bundle": str(paths.bundle),
        "manifest": str(paths.manifest),
        "receipt": str(paths.receipt),
        "failure": str(paths.failure),
        "terminal": str(paths.terminal),
    }
    if any(authority["output_paths"].get(key) != value for key, value in expected_paths.items()):
        raise RuntimeError("FIT authority output namespace differs from the bound parent")

    manifest = _plain_json(raw_json["manifest"], "FIT manifest")
    if set(manifest) != {
        "schema", "status", "authority_artifact_sha256",
        "authority_logical_sha256", "bundle", "bundle_summary", "protocol",
        "authorized_for_eval", "manifest_sha256",
    } or manifest.get("schema") != "causal_response_tensor_v1_fit_manifest" or (
        manifest.get("status") != "complete_fit_bundle_semantically_replayed"
        or manifest.get("authorized_for_eval") is not False
    ):
        raise RuntimeError("FIT manifest semantics changed")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _logical_sha256(manifest_body) or (
        manifest["authority_artifact_sha256"] != current["authority"]["sha256"]
        or manifest["authority_logical_sha256"] != authority["authority_sha256"]
        or manifest["protocol"] != protocol
        or manifest["bundle"] != {
            "path": str(paths.bundle),
            "sha256": current["bundle"]["sha256"],
            "bytes": current["bundle"]["bytes"],
        }
    ):
        raise RuntimeError("FIT manifest identity does not replay")

    payload = terminal["payload"]
    if type(payload) is not dict or set(payload) != {
        "status", "authorized_for_eval", "checkpoint",
        "model_state_sha256_before", "model_state_sha256_after", "outer_forwards",
        "projection_event_shape", "capture_event_shape",
    } or payload.get("status") != "complete" or (
        payload.get("authorized_for_eval") is not False
        or payload.get("outer_forwards") != 12_400
        or payload.get("projection_event_shape") != [2, 49, 124]
        or payload.get("capture_event_shape") != [6, 124]
        or payload.get("model_state_sha256_before")
        != payload.get("model_state_sha256_after")
    ):
        raise RuntimeError("FIT success receipt payload does not replay")

    body = {
        "schema": "causal_response_factorization_v1_fit_parent_binding",
        "receipt_sha256": receipt_record["sha256"],
        "terminal_sha256": terminal_record["sha256"],
        "authority_artifact_sha256": current["authority"]["sha256"],
        "authority_logical_sha256": authority["authority_sha256"],
        "bundle_sha256": current["bundle"]["sha256"],
        "bundle_bytes": current["bundle"]["bytes"],
        "manifest_artifact_sha256": current["manifest"]["sha256"],
        "manifest_logical_sha256": manifest["manifest_sha256"],
        "source_closure_sha256": authority["source_closure"]["sha256"],
        "fit_protocol": protocol,
        "tensor_values_deserialized": False,
        "authorized_for_eval": False,
    }
    return {**body, "binding_sha256": _logical_sha256(body)}
