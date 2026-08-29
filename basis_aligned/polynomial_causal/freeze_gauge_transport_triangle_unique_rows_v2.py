#!/usr/bin/env python3
"""Receipt-last v2 recovery for the spent unique-row v1 hash convention."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

import freeze_gauge_transport_triangle_unique_rows_v1 as v1


ROOT = v1.ROOT
HERE = v1.HERE
PREREG = HERE / "GAUGE_TRANSPORT_TRIANGLE_UNIQUE_ROWS_V2_RECOVERY_PREREGISTRATION.md"
RUNNER = HERE / "freeze_gauge_transport_triangle_unique_rows_v2.py"
TEST = HERE / "test_freeze_gauge_transport_triangle_unique_rows_v2.py"
V1_SOURCE = HERE / "freeze_gauge_transport_triangle_unique_rows_v1.py"
V1_AUTHORITY = HERE / "gauge_transport_triangle_unique_rows_v1_authority.json"
V1_FAILURE = HERE / "gauge_transport_triangle_unique_rows_v1_failure.json"
V1_AUTHORITY_FILE_SHA256 = "5f7435150561ef385c9a4ee51e2040c4a029e98faefbfe1bc0f92612d820498e"
V1_FAILURE_FILE_SHA256 = "91859b52b55b8be8ac05dc61f26b95fd43cdb92db7b8c39dfa72d226df41eb58"
V1_AUTHORITY_SHA256 = "8901a7446f70358e7e058013bb81c72f477c8636f5a1f76088307eda437025b5"
SELECTION_PLAN_SHA256 = "0d66f060a43959c94afc14691b4a19730147c942da94807f919513fb8c421629"

AUTHORITY = HERE / "gauge_transport_triangle_unique_rows_v2_authority.json"
ROWS = HERE / "gauge_transport_triangle_unique_rows_v2_rows.pt"
MANIFEST = HERE / "gauge_transport_triangle_unique_rows_v2_manifest.json"
RECEIPT = HERE / "gauge_transport_triangle_unique_rows_v2_receipt.json"
FAILURE = HERE / "gauge_transport_triangle_unique_rows_v2_failure.json"
LOCK = Path("/workspace/runs/.gauge_transport_triangle_unique_rows_v2.lock")
SOURCE_FILES = (PREREG, RUNNER, TEST, V1_SOURCE)


def raw_tensor_sha256(value: torch.Tensor) -> str:
    """Parent receipt convention: SHA256 of contiguous CPU tensor bytes only."""
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


@contextlib.contextmanager
def _v2_base_namespace() -> Iterator[None]:
    """Narrowly reuse v1's audited lock/publication/source primitives."""
    replacements = {
        "PREREG": PREREG,
        "RUNNER": RUNNER,
        "TEST": TEST,
        "SOURCE_FILES": SOURCE_FILES,
        "AUTHORITY": AUTHORITY,
        "ROWS": ROWS,
        "MANIFEST": MANIFEST,
        "RECEIPT": RECEIPT,
        "FAILURE": FAILURE,
        "LOCK": LOCK,
    }
    originals = {name: getattr(v1, name) for name in replacements}
    for name, value in replacements.items():
        setattr(v1, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(v1, name, value)


def source_closure() -> dict[str, Any]:
    with _v2_base_namespace():
        return v1.source_closure()


def validate_source_closure(value: Mapping[str, Any]) -> None:
    with _v2_base_namespace():
        v1.validate_source_closure(value)


def acquire_owner_lock() -> dict[str, Any]:
    with _v2_base_namespace():
        return v1.acquire_owner_lock()


def assert_owner_lock(owner: Mapping[str, Any]) -> None:
    with _v2_base_namespace():
        v1.assert_owner_lock(owner)


def release_owner_lock(owner: Mapping[str, Any]) -> None:
    with _v2_base_namespace():
        v1.release_owner_lock(owner)


def publish_json(path: Path, value: Mapping[str, Any], owner: Mapping[str, Any]) -> None:
    with _v2_base_namespace():
        v1.publish_json(path, value, owner=owner)


def publish_bytes(path: Path, value: bytes, owner: Mapping[str, Any]) -> None:
    with _v2_base_namespace():
        v1._publish_create_only(path, value, owner=owner)


def load_v1_parents() -> tuple[dict[str, Any], dict[str, Any]]:
    if v1.file_sha256(V1_AUTHORITY) != V1_AUTHORITY_FILE_SHA256 or v1.file_sha256(
        V1_FAILURE
    ) != V1_FAILURE_FILE_SHA256:
        raise RuntimeError("spent v1 parent bytes changed")
    authority = json.loads(V1_AUTHORITY.read_text())
    failure = json.loads(V1_FAILURE.read_text())
    v1.validate_authority(authority)
    if authority.get("authority_sha256") != V1_AUTHORITY_SHA256 or authority.get(
        "selection_plan", {}
    ).get("selection_plan_sha256") != SELECTION_PLAN_SHA256:
        raise RuntimeError("spent v1 authority identity changed")
    if failure != {
        "exception_message": "unique-row cache tensor changed: n192_skip11000",
        "exception_type": "RuntimeError",
        "manifest_exists": False,
        "receipt_exists": False,
        "rows_exists": False,
        "schema": "gauge_transport_triangle_unique_rows_v1_failure",
        "status": "terminal_failure_no_receipt",
    }:
        raise RuntimeError("spent v1 failure semantics changed")
    for key in ("rows", "manifest", "receipt"):
        if Path(authority["outputs"][key]).exists():
            raise RuntimeError(f"spent v1 unexpectedly has {key}")
    return authority, failure


def load_parent_metadata() -> dict[str, Any]:
    return v1.load_parent_metadata()


def build_authority(
    source: Mapping[str, Any], parent: Mapping[str, Any], v1_authority: Mapping[str, Any],
) -> dict[str, Any]:
    selection = v1.build_selection_plan(parent)
    if selection != v1_authority["selection_plan"]:
        raise RuntimeError("v2 selection is not exactly the spent v1 selection")
    entries = parent["entries"]
    cache_bindings = {}
    for key, frozen in v1.SOURCE_CACHES.items():
        if entries[key]["tensor_raw_sha256"] != frozen["tensor_sha256"]:
            raise RuntimeError(f"parent raw-byte hash changed: {key}")
        cache_bindings[key] = {
            "path": str(frozen["path"]),
            "shape": list(frozen["shape"]),
            "dtype": "torch.int64",
            "file_sha256": frozen["file_sha256"],
            "tensor_raw_sha256": entries[key]["tensor_raw_sha256"],
        }
    body = {
        "schema": "gauge_transport_triangle_unique_rows_v2_authority",
        "status": "frozen_recovery_before_any_v2_row_tensor_model_or_outcome_load",
        "source_closure": dict(source),
        "recovery_parents": {
            "v1_authority_path": str(V1_AUTHORITY),
            "v1_authority_file_sha256": V1_AUTHORITY_FILE_SHA256,
            "v1_authority_sha256": V1_AUTHORITY_SHA256,
            "v1_failure_path": str(V1_FAILURE),
            "v1_failure_file_sha256": V1_FAILURE_FILE_SHA256,
            "v1_rows_manifest_receipt_absent": True,
        },
        "parent_receipt_path": str(v1.PARENT_RECEIPT),
        "parent_receipt_file_sha256": v1.PARENT_RECEIPT_SHA256,
        "selection_plan": selection,
        "cache_bindings": cache_bindings,
        "hash_protocols": {
            "source_cache_tensor": "sha256(contiguous_cpu_tensor_raw_bytes_only)",
            "output_role_tensor": "sha256(dtype_utf8_then_shape_json_then_contiguous_cpu_raw_bytes)",
            "v2_delta_from_v1": "source_cache_validation_uses_parent_raw_byte_hash",
        },
        "permissions": dict(v1_authority["permissions"]),
        "outputs": {
            "authority": str(AUTHORITY), "rows": str(ROWS), "manifest": str(MANIFEST),
            "receipt": str(RECEIPT), "failure": str(FAILURE), "lock": str(LOCK),
        },
    }
    return {**body, "authority_sha256": v1.canonical_sha256(body)}


def validate_authority(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or value.get(
        "schema"
    ) != "gauge_transport_triangle_unique_rows_v2_authority":
        raise RuntimeError("v2 authority schema changed")
    body = {key: value[key] for key in value if key != "authority_sha256"}
    if v1.canonical_sha256(body) != value.get("authority_sha256"):
        raise RuntimeError("v2 authority hash changed")
    validate_source_closure(value["source_closure"])
    parent = load_parent_metadata()
    v1_authority, _ = load_v1_parents()
    if dict(value) != build_authority(value["source_closure"], parent, v1_authority):
        raise RuntimeError("v2 authority differs from exact recovery rebuild")


def load_cache_tensors(authority: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    output = {}
    for key, expected in authority["cache_bindings"].items():
        path = Path(expected["path"])
        before = v1.file_sha256(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        after = v1.file_sha256(path)
        if before != expected["file_sha256"] or after != before or not torch.is_tensor(value) or (
            value.dtype != torch.long or tuple(value.shape) != tuple(expected["shape"])
            or raw_tensor_sha256(value) != expected["tensor_raw_sha256"]
        ):
            raise RuntimeError(f"v2 source-cache replay changed: {key}")
        output[key] = value.contiguous()
    return output


def build_rows_payload(authority: Mapping[str, Any], caches: Mapping[str, torch.Tensor]) -> dict:
    roles, records = {}, {}
    for role, plan in authority["selection_plan"]["roles"].items():
        roles[role] = torch.stack([
            caches[item["source_key"]][item["source_row_index"]].clone() for item in plan
        ]).contiguous()
        records[role] = [dict(item) for item in plan]
    payload = {
        "schema": "gauge_transport_triangle_unique_rows_v2_rows",
        "authority_sha256": authority["authority_sha256"],
        "selection_plan_sha256": SELECTION_PLAN_SHA256,
        "roles": roles,
        "records": records,
    }
    validate_rows_payload(payload, authority)
    return payload


def validate_rows_payload(payload: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    as_v1 = dict(payload)
    as_v1["schema"] = "gauge_transport_triangle_unique_rows_v1_rows"
    v1.validate_rows_payload(as_v1, authority)


def build_manifest(payload: Mapping[str, Any], authority: Mapping[str, Any]) -> dict:
    body = {
        "schema": "gauge_transport_triangle_unique_rows_v2_manifest",
        "authority_sha256": authority["authority_sha256"],
        "selection_plan_sha256": SELECTION_PLAN_SHA256,
        "v1_authority_file_sha256": V1_AUTHORITY_FILE_SHA256,
        "v1_failure_file_sha256": V1_FAILURE_FILE_SHA256,
        "parent_receipt_file_sha256": v1.PARENT_RECEIPT_SHA256,
        "rows_file_sha256": v1.file_sha256(ROWS),
        "role_tensor_composite_sha256s": {
            role: v1.tensor_sha256(payload["roles"][role]) for role in v1.ROLE_SIZES
        },
        "role_record_sha256s": {
            role: v1.canonical_sha256(payload["records"][role]) for role in v1.ROLE_SIZES
        },
        "role_sizes": v1.ROLE_SIZES,
        "source_contributions": v1.EXPECTED_CONTRIBUTIONS,
        "unique_document_count": sum(v1.ROLE_SIZES.values()),
        "hash_protocols": authority["hash_protocols"],
        "triangle_runner_authorized_by_this_manifest": False,
        "authorized_for_training": False,
        "authorized_for_global_ledger_credit": False,
    }
    return {**body, "manifest_sha256": v1.canonical_sha256(body)}


def replay_terminal_state(expected_authority: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    authority_file_sha256 = v1.file_sha256(AUTHORITY)
    authority = json.loads(AUTHORITY.read_text())
    validate_authority(authority)
    if authority != expected_authority or v1.file_sha256(AUTHORITY) != authority_file_sha256:
        raise RuntimeError("v2 authority changed during replay")
    caches = load_cache_tensors(authority)
    rows_file_sha256 = v1.file_sha256(ROWS)
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    validate_rows_payload(payload, authority)
    rebuilt = build_rows_payload(authority, caches)
    if payload["records"] != rebuilt["records"] or any(
        not torch.equal(payload["roles"][role], rebuilt["roles"][role]) for role in v1.ROLE_SIZES
    ) or v1.file_sha256(ROWS) != rows_file_sha256:
        raise RuntimeError("v2 rows differ from exact source-cache replay")
    manifest_file_sha256 = v1.file_sha256(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    if manifest != build_manifest(payload, authority) or v1.file_sha256(
        MANIFEST
    ) != manifest_file_sha256:
        raise RuntimeError("v2 manifest differs from exact rebuild")
    load_v1_parents()
    replay = {
        "authority_file_sha256": authority_file_sha256,
        "rows_file_sha256": rows_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "source_cache_file_sha256s": {
            key: v1.file_sha256(Path(binding["path"]))
            for key, binding in authority["cache_bindings"].items()
        },
    }
    return payload, manifest, replay


def build_receipt(authority: Mapping, payload: Mapping, manifest: Mapping, replay: Mapping) -> dict:
    if FAILURE.exists():
        raise RuntimeError("v2 failure exists before receipt")
    body = {
        "schema": "gauge_transport_triangle_unique_rows_v2_receipt",
        "status": "complete_v2_unique_document_rows_receipt_last",
        **dict(replay),
        "authority_sha256": authority["authority_sha256"],
        "selection_plan_sha256": SELECTION_PLAN_SHA256,
        "manifest_sha256": manifest["manifest_sha256"],
        "role_tensor_composite_sha256s": manifest["role_tensor_composite_sha256s"],
        "v1_authority_file_sha256": V1_AUTHORITY_FILE_SHA256,
        "v1_failure_file_sha256": V1_FAILURE_FILE_SHA256,
        "unique_document_count": sum(v1.ROLE_SIZES.values()),
        "failure_absent": True,
        "triangle_runner_authorized_by_this_receipt": False,
        "global_ledger_credit": False,
    }
    return {**body, "receipt_sha256": v1.canonical_sha256(body)}


def validate_receipt(value: Mapping[str, Any]) -> None:
    if value.get("failure_absent") is not True or FAILURE.exists():
        raise RuntimeError("v2 receipt/failure exclusivity changed")
    authority = json.loads(AUTHORITY.read_text())
    payload, manifest, replay = replay_terminal_state(authority)
    if FAILURE.exists() or dict(value) != build_receipt(authority, payload, manifest, replay):
        raise RuntimeError("v2 receipt differs from exact terminal replay")


def freeze_authority() -> dict:
    if any(path.exists() for path in (AUTHORITY, ROWS, MANIFEST, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("v2 authority namespace is not pristine")
    owner = acquire_owner_lock()
    try:
        v1_authority, _ = load_v1_parents()
        value = build_authority(source_closure(), load_parent_metadata(), v1_authority)
        publish_json(AUTHORITY, value, owner)
        validate_authority(json.loads(AUTHORITY.read_text()))
        return value
    finally:
        release_owner_lock(owner)


def materialize() -> dict:
    if not AUTHORITY.is_file() or any(path.exists() for path in (ROWS, MANIFEST, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("v2 materialization namespace is unavailable")
    owner = acquire_owner_lock()
    try:
        authority = json.loads(AUTHORITY.read_text())
        validate_authority(authority)
        payload = build_rows_payload(authority, load_cache_tensors(authority))
        serialized = io.BytesIO()
        torch.save(payload, serialized)
        publish_bytes(ROWS, serialized.getvalue(), owner)
        manifest = build_manifest(payload, authority)
        publish_json(MANIFEST, manifest, owner)
        payload, manifest, replay = replay_terminal_state(authority)
        receipt = build_receipt(authority, payload, manifest, replay)
        publish_json(RECEIPT, receipt, owner)
        validate_receipt(json.loads(RECEIPT.read_text()))
        return receipt
    except BaseException as error:
        try:
            assert_owner_lock(owner)
            if not RECEIPT.exists() and not FAILURE.exists():
                publish_json(FAILURE, {
                    "schema": "gauge_transport_triangle_unique_rows_v2_failure",
                    "status": "terminal_v2_failure_no_receipt",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "rows_exists": ROWS.exists(), "manifest_exists": MANIFEST.exists(),
                    "receipt_exists": RECEIPT.exists(),
                }, owner)
        except BaseException:
            pass
        raise
    finally:
        release_owner_lock(owner)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-authority", action="store_true")
    group.add_argument("--materialize", action="store_true")
    arguments = parser.parse_args()
    result = freeze_authority() if arguments.freeze_authority else materialize()
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
