"""FIT parent binding by content identity, for artifacts re-materialized from git.

`causal_response_factorization_v1_parent_binding.fit_parent_binding_without_tensor_load`
compares every receipt-bound artifact against the FIT receipt's aggregate record
including its *physical* identity (device, inode, mtime).  That defends a live
transaction on one filesystem against in-place replacement.  It cannot replay after
the repository is cloned onto a fresh instance: bytes and SHA-256 are unchanged, but
inode and mtime are necessarily new, so the published function fails closed forever.

This module is the single, explicit deviation.  It repeats the published validation
verbatim except that receipt-bound aggregate records are compared by content identity
(`path`, `sha256`, `bytes`) and the physical-identity mismatch is returned separately
by `physical_identity_deviation()` so that any authority built on it carries the
recorded-versus-observed inode/mtime for each artifact.  The returned binding body is
byte-for-byte the body the published function would return on the original
filesystem, so `binding_sha256` is stable across machines.  The published module is
not modified; its bytes remain pinned by earlier source closures.
"""

from __future__ import annotations

from typing import Any

import causal_response_factorization_v1_parent_binding as parent
from causal_response_factorization_v1_parent_binding import (
    FitParentPaths, PRODUCTION_PATHS, _is_sha256, _logical_sha256, _plain_json,
    _require_absent, _stable_record, _validate_frozen_parent_hashes,
    _validate_historical_source_closure, _validate_independent_audit,
    _validate_protocol,
)


CONTENT_IDENTITY_KEYS = ("path", "sha256", "bytes")
PHYSICAL_IDENTITY_KEYS = ("device", "inode", "mtime_ns", "ctime_ns")
RECEIPT_BOUND = ("authority", "bundle", "manifest")


def _same_content_identity(expected: object, observed: object) -> bool:
    if type(expected) is not dict or type(observed) is not dict or set(expected) != set(
        observed
    ):
        return False
    return all(expected[key] == observed[key] for key in CONTENT_IDENTITY_KEYS)


def _validated_terminal(paths: FitParentPaths) -> tuple[dict[str, Any], dict[str, Any], bytes, dict[str, Any], bytes]:
    if not isinstance(paths, FitParentPaths):
        raise TypeError("FIT parent paths must be an exact FitParentPaths value")
    _require_absent(paths.failure, "FIT failure terminal")
    _require_absent(paths.lock, "FIT owner lock")
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
    if type(aggregate) is not dict or set(aggregate) != set(RECEIPT_BOUND):
        raise RuntimeError("FIT receipt aggregate schema changed")
    return terminal, terminal_record, terminal_raw, receipt_record, receipt_raw


def physical_identity_deviation(paths: FitParentPaths = PRODUCTION_PATHS) -> dict[str, Any]:
    """Recorded-versus-observed physical identity of every receipt-bound artifact."""

    terminal, *_ = _validated_terminal(paths)
    aggregate = terminal["aggregate"]
    artifacts: dict[str, Any] = {}
    for name in RECEIPT_BOUND:
        record, _ = _stable_record(getattr(paths, name))
        expected = aggregate[name]
        if not _same_content_identity(expected, record):
            raise RuntimeError(f"FIT receipt-bound {name} content identity changed")
        artifacts[name] = {
            "content_identity_matches": True,
            "sha256": record["sha256"],
            "bytes": record["bytes"],
            "physical_identity_matches": all(
                expected[key] == record[key] for key in ("device", "inode", "mtime_ns")
            ),
            "recorded": {key: expected[key] for key in PHYSICAL_IDENTITY_KEYS},
            "observed": {key: record[key] for key in PHYSICAL_IDENTITY_KEYS},
        }
    return {
        "reason": (
            "artifacts re-materialized from git on a fresh instance; content identity "
            "(sha256, bytes) replays, physical identity (device, inode, mtime) cannot"
        ),
        "comparison_used_for_binding": list(CONTENT_IDENTITY_KEYS),
        "artifacts": artifacts,
    }


def fit_parent_binding_by_content_identity(
    paths: FitParentPaths = PRODUCTION_PATHS,
) -> dict[str, Any]:
    """The published binding, with receipt-bound artifacts compared by content."""

    terminal, terminal_record, terminal_raw, receipt_record, receipt_raw = (
        _validated_terminal(paths)
    )
    aggregate = terminal["aggregate"]
    current: dict[str, dict[str, Any]] = {}
    raw_json: dict[str, bytes] = {}
    for name in RECEIPT_BOUND:
        record, raw = _stable_record(getattr(paths, name))
        if not _same_content_identity(aggregate[name], record):
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
    source_closure_sha256 = _validate_historical_source_closure(
        authority["source_closure"]
    )
    _validate_independent_audit(
        authority["independent_audit"], authority["source_closure"]
    )
    _validate_frozen_parent_hashes(authority["parents"])
    protocol = _validate_protocol(authority["protocol"])
    expected_paths = {
        "authority": str(paths.authority),
        "bundle": str(paths.bundle),
        "manifest": str(paths.manifest),
        "receipt": str(paths.receipt),
        "failure": str(paths.failure),
        "terminal": str(paths.terminal),
        "lock": str(paths.lock),
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
    _require_absent(paths.failure, "FIT failure terminal")
    _require_absent(paths.lock, "FIT owner lock")

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
        "source_closure_sha256": source_closure_sha256,
        "fit_protocol": protocol,
        "tensor_values_deserialized": False,
        "authorized_for_eval": False,
    }
    # Close the time-of-check/time-of-use window exactly as the published binding does:
    # every receipt-bound artifact is replayed after all semantic work, and the physical
    # identity observed at the START of this call must still hold at the END.
    final_terminal_record, final_terminal_raw = _stable_record(paths.terminal)
    final_receipt_record, final_receipt_raw = _stable_record(paths.receipt)
    if (
        final_terminal_record != terminal_record
        or final_receipt_record != receipt_record
        or final_terminal_raw != terminal_raw
        or final_receipt_raw != receipt_raw
    ):
        raise RuntimeError("FIT terminal or receipt changed during parent validation")
    for name in RECEIPT_BOUND:
        final_record, _ = _stable_record(getattr(paths, name))
        if final_record != current[name] or not _same_content_identity(
            aggregate[name], final_record
        ):
            raise RuntimeError(f"FIT receipt-bound {name} changed during terminal replay")
    _require_absent(paths.failure, "FIT failure terminal")
    _require_absent(paths.lock, "FIT owner lock")
    return {**body, "binding_sha256": _logical_sha256(body)}
