"""Transactional fit-artifact publisher for suffix-transport v1.

The numerical fit owner deliberately returns in-memory CPU records.  This module is
the sole create-only publication boundary for those records.  It injects the
canonical denominator child namespace, publishes ledger then manifest then receipt,
and replays all semantic and lifecycle bindings before granting selection authority.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_lifecycle as lifecycle
import early_mlp_suffix_transport_v1_observational_authority as final_authority
import early_mlp_suffix_transport_v1_runtime as runtime


LEDGER_KIND = "early_mlp_suffix_transport_v1_fit_ledger"
MANIFEST_KIND = "early_mlp_suffix_transport_v1_fit_manifest"
RECEIPT_KIND = "early_mlp_suffix_transport_v1_fit_receipt"


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


def _tensor_tree(value: Any, *, clone: bool) -> Any:
    if torch.is_tensor(value):
        if value.device.type != "cpu" or value.requires_grad or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("fit publication tensors must be finite detached CPU values")
        tensor = value.detach().contiguous()
        return tensor.clone() if clone else {
            "tensor_sha256": runtime.tensor_identity_sha256(tensor),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("fit publication mappings require nonempty string keys")
        return {key: _tensor_tree(value[key], clone=clone) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_tensor_tree(item, clone=clone) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("fit publication contains a nonfinite scalar")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"fit publication contains unsupported {type(value).__name__}")


def _identity(value: Any) -> str:
    return runtime.logical_identity_sha256(_tensor_tree(value, clone=False))


def build_fit_ledger(
    *, fit_records: Mapping[str, Any], denominator_pass: fit.DenominatorPass,
    fit_execution_sha256: str, fit_role_tensor_sha256: str,
) -> dict[str, Any]:
    """Build the canonical weights-only ledger without publishing it."""

    if not isinstance(fit_records, Mapping) or not fit_records or not isinstance(
        denominator_pass, fit.DenominatorPass
    ):
        raise TypeError("fit publication requires records and a denominator pass")
    _sha256("fit execution", fit_execution_sha256)
    _sha256("fit role tensor", fit_role_tensor_sha256)
    records = _tensor_tree(fit_records, clone=True)
    body = {
        "schema_version": 1, "kind": LEDGER_KIND,
        "fit_execution_sha256": fit_execution_sha256,
        "fit_role_tensor_sha256": fit_role_tensor_sha256,
        "fit_records": records,
        final_authority.DENOMINATOR_LEDGER_KEY: (
            final_authority.denominator_pass_payload(denominator_pass)
        ),
    }
    return {**body, "ledger_payload_sha256": _identity(body)}


def validate_fit_ledger(value: Any) -> Mapping[str, Any]:
    keys = {
        "schema_version", "kind", "fit_execution_sha256", "fit_role_tensor_sha256",
        "fit_records", final_authority.DENOMINATOR_LEDGER_KEY,
        "ledger_payload_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys or value.get(
        "schema_version"
    ) != 1 or value.get("kind") != LEDGER_KIND:
        raise RuntimeError("fit ledger schema changed")
    _sha256("fit ledger execution", value.get("fit_execution_sha256"))
    _sha256("fit ledger role", value.get("fit_role_tensor_sha256"))
    _sha256("fit ledger payload", value.get("ledger_payload_sha256"))
    _tensor_tree(value["fit_records"], clone=False)
    denominator = final_authority._restore_denominator_pass(
        value[final_authority.DENOMINATOR_LEDGER_KEY]
    )
    body = {key: value[key] for key in value if key != "ledger_payload_sha256"}
    if _identity(body) != value["ledger_payload_sha256"]:
        raise RuntimeError("fit ledger payload identity changed")
    return {**dict(value), "denominator_pass": denominator}


def _manifest(
    *, ledger: Mapping[str, Any], manifest_records: Mapping[str, Any],
    paths: lifecycle.ArtifactPaths,
) -> dict[str, Any]:
    records = _tensor_tree(manifest_records, clone=True)
    # JSON publication may not contain tensors even if the ledger may.
    if any(torch.is_tensor(item) for item in _walk(records)):
        raise ValueError("fit manifest records cannot contain tensors")
    return {
        "schema_version": 1, "kind": MANIFEST_KIND,
        "status": "fit_ledger_frozen_before_selection",
        "authority": "none", "authorized_for_selection": False,
        "fit_execution_sha256": ledger["fit_execution_sha256"],
        "fit_role_tensor_sha256": ledger["fit_role_tensor_sha256"],
        "ledger_payload_sha256": ledger["ledger_payload_sha256"],
        "denominator_pass_sha256": ledger["denominator_pass"].sha256,
        "fit_ledger": lifecycle.artifact_binding(paths.fit_ledger),
        "fit_manifest_records": records,
    }


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def _receipt(
    *, ledger: Mapping[str, Any], manifest: Mapping[str, Any],
    source_closure: Mapping[str, Any], protected_before: Mapping[str, Any],
    paths: lifecycle.ArtifactPaths,
) -> dict[str, Any]:
    return {
        "schema_version": 1, "kind": RECEIPT_KIND,
        "status": "fit_frozen_before_selection",
        "authority": "early_mlp_suffix_transport_v1_fit_unlock",
        "authorized_for_selection": True,
        "rows_receipt": lifecycle.artifact_binding(paths.rows_receipt),
        "rows_manifest": lifecycle.artifact_binding(paths.rows_manifest),
        "fit_ledger": lifecycle.artifact_binding(paths.fit_ledger),
        "fit_manifest": lifecycle.artifact_binding(paths.fit_manifest),
        "fit_execution_sha256": ledger["fit_execution_sha256"],
        "fit_role_tensor_sha256": ledger["fit_role_tensor_sha256"],
        "ledger_payload_sha256": ledger["ledger_payload_sha256"],
        "source_commit": source_closure["source_commit"],
        "source_hashes": dict(source_closure["source_hashes"]),
        "protected_before": dict(protected_before),
        final_authority.DENOMINATOR_AUTHORITY_KEY: (
            final_authority.denominator_pass_authority_payload(
                ledger["denominator_pass"], paths=paths,
            )
        ),
    }


def validate_fit_publication(
    *, paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if not all(path.is_file() for path in (
        paths.fit_ledger, paths.fit_manifest, paths.fit_receipt,
    )):
        raise RuntimeError("fit publication is incomplete")
    ledger_raw = torch.load(paths.fit_ledger, map_location="cpu", weights_only=True)
    ledger = validate_fit_ledger(ledger_raw)
    manifest = json.loads(paths.fit_manifest.read_text())
    manifest_keys = {
        "schema_version", "kind", "status", "authority",
        "authorized_for_selection", "fit_execution_sha256",
        "fit_role_tensor_sha256", "ledger_payload_sha256",
        "denominator_pass_sha256", "fit_ledger", "fit_manifest_records",
    }
    expected_manifest = {
        "schema_version": 1, "kind": MANIFEST_KIND,
        "status": "fit_ledger_frozen_before_selection", "authority": "none",
        "authorized_for_selection": False,
        "fit_execution_sha256": ledger["fit_execution_sha256"],
        "fit_role_tensor_sha256": ledger["fit_role_tensor_sha256"],
        "ledger_payload_sha256": ledger["ledger_payload_sha256"],
        "denominator_pass_sha256": ledger["denominator_pass"].sha256,
        "fit_ledger": lifecycle.artifact_binding(paths.fit_ledger),
    }
    if not isinstance(manifest, Mapping) or set(manifest) != manifest_keys or any(
        manifest.get(key) != expected for key, expected in expected_manifest.items()
    ):
        raise RuntimeError("fit manifest differs from its ledger")
    _tensor_tree(manifest["fit_manifest_records"], clone=False)
    receipt = json.loads(paths.fit_receipt.read_text())
    required = {
        "schema_version": 1, "kind": RECEIPT_KIND,
        "status": "fit_frozen_before_selection",
        "authority": "early_mlp_suffix_transport_v1_fit_unlock",
        "authorized_for_selection": True,
        "rows_receipt": lifecycle.artifact_binding(paths.rows_receipt),
        "rows_manifest": lifecycle.artifact_binding(paths.rows_manifest),
        "fit_ledger": lifecycle.artifact_binding(paths.fit_ledger),
        "fit_manifest": lifecycle.artifact_binding(paths.fit_manifest),
        "fit_execution_sha256": ledger["fit_execution_sha256"],
        "fit_role_tensor_sha256": ledger["fit_role_tensor_sha256"],
        "ledger_payload_sha256": ledger["ledger_payload_sha256"],
    }
    extra = {
        "source_commit", "source_hashes", "protected_before",
        final_authority.DENOMINATOR_AUTHORITY_KEY,
    }
    if not isinstance(receipt, Mapping) or set(receipt) != set(required) | extra or any(
        receipt.get(key) != expected for key, expected in required.items()
    ):
        raise RuntimeError("fit receipt differs from its ledger/manifest")
    child = receipt[final_authority.DENOMINATOR_AUTHORITY_KEY]
    expected_child = final_authority.denominator_pass_authority_payload(
        ledger["denominator_pass"], paths=paths,
    )
    if child != expected_child:
        raise RuntimeError("fit receipt denominator child changed")
    lifecycle.verify_source_closure(receipt["source_commit"], receipt["source_hashes"])
    lifecycle.require_protected_snapshot(
        tuple(Path(key) for key in receipt["protected_before"]),
        receipt["protected_before"],
    )
    return ledger, manifest, receipt


def publish_fit_artifacts(
    *, fit_records: Mapping[str, Any], manifest_records: Mapping[str, Any],
    denominator_pass: fit.DenominatorPass, fit_execution_sha256: str,
    fit_role_tensor_sha256: str, source_closure: Mapping[str, Any],
    protected_before: Mapping[str, Any], lock_nonce: str,
    paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
    lock_path: Path = lifecycle.RUN_LOCK,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Create ledger, manifest, and selection-authorizing receipt in that order."""

    lifecycle.require_run_claim(lock_nonce, lock_path)
    paths.assert_stage_preconditions("fit")
    if not isinstance(source_closure, Mapping) or set(source_closure) != {
        "source_commit", "source_hashes"
    }:
        raise ValueError("fit publication source closure changed")
    lifecycle.verify_source_closure(
        source_closure["source_commit"], source_closure["source_hashes"],
    )
    required_before = (paths.rows_receipt, paths.rows_manifest)
    expected_before = lifecycle.protected_snapshot(required_before)
    if dict(protected_before) != expected_before:
        raise RuntimeError("fit publication protected row snapshot changed")
    ledger = build_fit_ledger(
        fit_records=fit_records, denominator_pass=denominator_pass,
        fit_execution_sha256=fit_execution_sha256,
        fit_role_tensor_sha256=fit_role_tensor_sha256,
    )
    lifecycle.atomic_create_torch(ledger, paths.fit_ledger)
    reloaded = validate_fit_ledger(torch.load(
        paths.fit_ledger, map_location="cpu", weights_only=True,
    ))
    manifest = _manifest(
        ledger=reloaded, manifest_records=manifest_records, paths=paths,
    )
    lifecycle.atomic_create_json(manifest, paths.fit_manifest)
    receipt = _receipt(
        ledger=reloaded, manifest=manifest, source_closure=source_closure,
        protected_before=protected_before, paths=paths,
    )
    lifecycle.atomic_create_json(receipt, paths.fit_receipt)
    lifecycle.require_run_claim(lock_nonce, lock_path)
    return validate_fit_publication(paths=paths)
