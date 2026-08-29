#!/usr/bin/env python3
"""Metadata-first, receipt-last freezer for unique triangle source documents."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/workspace/tensor_language")
HERE = ROOT / "basis_aligned" / "polynomial_causal"
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PREREG = HERE / "GAUGE_TRANSPORT_TRIANGLE_UNIQUE_ROWS_V1_PREREGISTRATION.md"
RUNNER = HERE / "freeze_gauge_transport_triangle_unique_rows_v1.py"
TEST = HERE / "test_freeze_gauge_transport_triangle_unique_rows_v1.py"
PARENT_RECEIPT = BQ / ".rowcache" / "fineweb_oracle_v2_receipt.json"
AUTHORITY = HERE / "gauge_transport_triangle_unique_rows_v1_authority.json"
ROWS = HERE / "gauge_transport_triangle_unique_rows_v1_rows.pt"
MANIFEST = HERE / "gauge_transport_triangle_unique_rows_v1_manifest.json"
RECEIPT = HERE / "gauge_transport_triangle_unique_rows_v1_receipt.json"
FAILURE = HERE / "gauge_transport_triangle_unique_rows_v1_failure.json"
LOCK = Path("/workspace/runs/.gauge_transport_triangle_unique_rows_v1.lock")

PARENT_RECEIPT_SHA256 = "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16"
PINNED_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
PINNED_MANIFEST_SHA256 = "ba5e92b0d157f47cc6f8656eb1c37e46b7aac6957be8be68c1596736b98e6f90"
PINNED_PARQUET_SHA256 = "c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930"
TOKEN_LENGTH = 513
ROLE_SIZES = {"basis": 96, "fit": 96, "evaluation": 192}
ROLE_POOLS = {
    "basis": ("n480_skip80",),
    "fit": ("n480_skip80",),
    "evaluation": ("n192_skip11000", "n192_skip7000", "n96_skip1200"),
}
EXPECTED_CONTRIBUTIONS = {
    "basis": {"n480_skip80": 96},
    "fit": {"n480_skip80": 96},
    "evaluation": {
        "n192_skip11000": 105,
        "n192_skip7000": 79,
        "n96_skip1200": 8,
    },
}
SOURCE_CACHES = {
    "n480_skip80": {
        "path": BQ / ".rowcache" / "fineweb_n480_skip80.pt",
        "shape": (480, TOKEN_LENGTH),
        "file_sha256": "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496",
        "tensor_sha256": "343d92ce07f78572e3233120d3361814c63f69fa76e97e58b62d1d6c8f24497f",
    },
    "n192_skip7000": {
        "path": BQ / ".rowcache" / "fineweb_n192_skip7000.pt",
        "shape": (192, TOKEN_LENGTH),
        "file_sha256": "d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c",
        "tensor_sha256": "10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0",
    },
    "n192_skip11000": {
        "path": BQ / ".rowcache" / "fineweb_n192_skip11000.pt",
        "shape": (192, TOKEN_LENGTH),
        "file_sha256": "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
        "tensor_sha256": "5d6c1697f6d05860e4235c21e6324e3451d47924565d8edb62e06fbe37b3a1fa",
    },
    "n96_skip1200": {
        "path": BQ / ".rowcache" / "fineweb_n96_skip1200.pt",
        "shape": (96, TOKEN_LENGTH),
        "file_sha256": "21707551f35d13818c10ac59e12e9445ef076d0522371fe779691bfab719d34f",
        "tensor_sha256": "d6302f327983e8233509e0ad8a05aa84fad88784861a9f8d10575b325be83dda",
    },
}
SOURCE_FILES = (PREREG, RUNNER, TEST)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        advanced = os.write(descriptor, payload[offset:])
        if advanced <= 0:
            raise OSError("create-only publication made no progress")
        offset += advanced


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acquire_owner_lock() -> dict[str, Any]:
    nonce = secrets.token_hex(32)
    payload = (json.dumps({
        "schema": "gauge_transport_triangle_unique_rows_v1_owner_lock",
        "nonce": nonce,
        "pid": os.getpid(),
    }, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        identity = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(LOCK.parent)
    return {
        "device": identity.st_dev,
        "inode": identity.st_ino,
        "nonce": nonce,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def assert_owner_lock(owner: Mapping[str, Any]) -> None:
    identity = os.lstat(LOCK)
    if not stat.S_ISREG(identity.st_mode) or identity.st_dev != owner.get("device") or (
        identity.st_ino != owner.get("inode")
    ):
        raise RuntimeError("unique-row owner lock was replaced")
    payload = LOCK.read_bytes()
    decoded = json.loads(payload)
    if hashlib.sha256(payload).hexdigest() != owner.get("payload_sha256") or (
        decoded.get("schema") != "gauge_transport_triangle_unique_rows_v1_owner_lock"
        or decoded.get("nonce") != owner.get("nonce")
    ):
        raise RuntimeError("unique-row owner lock contents changed")


def release_owner_lock(owner: Mapping[str, Any]) -> None:
    assert_owner_lock(owner)
    LOCK.unlink()
    _fsync_directory(LOCK.parent)


def _publish_create_only(
    path: Path, payload: bytes, *, owner: Mapping[str, Any] | None = None,
) -> None:
    if owner is not None:
        assert_owner_lock(owner)
    temporary = path.with_name(f".{path.name}.tmp-{secrets.token_hex(16)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    linked = False
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if owner is not None:
            assert_owner_lock(owner)
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    if not linked:
        raise RuntimeError("create-only publication did not link its final path")


def publish_json(
    path: Path, value: Mapping[str, Any], *, owner: Mapping[str, Any] | None = None,
) -> None:
    _publish_create_only(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        owner=owner,
    )


def source_closure() -> dict[str, Any]:
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT).decode().strip()
    hashes = {}
    for path in SOURCE_FILES:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.run(
            ("git", "show", f"{commit}:{relative}"), cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if committed.returncode:
            raise RuntimeError(f"unique-row source is not committed: {relative}")
        digest = hashlib.sha256(committed.stdout).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"live unique-row source differs from commit: {relative}")
        hashes[relative] = digest
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "origin/main"), cwd=ROOT,
    ).returncode:
        raise RuntimeError("unique-row source commit is not pushed")
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": canonical_sha256(body)}


def validate_source_closure(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {"commit", "paths", "sha256"} or not (
        isinstance(value.get("commit"), str) and isinstance(value.get("paths"), Mapping)
    ) or set(value["paths"]) != {str(path.relative_to(ROOT)) for path in SOURCE_FILES} or (
        canonical_sha256({"commit": value["commit"], "paths": value["paths"]})
        != value.get("sha256")
    ):
        raise RuntimeError("unique-row source closure is malformed")
    for relative, digest in value["paths"].items():
        committed = subprocess.run(
            ("git", "show", f"{value['commit']}:{relative}"), cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise RuntimeError(f"unique-row committed source drift: {relative}")
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"unique-row live source drift: {relative}")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", value["commit"], "origin/main"), cwd=ROOT,
    ).returncode:
        raise RuntimeError("unique-row source commit is no longer on pushed origin/main")


def load_parent_metadata() -> dict[str, Any]:
    before = file_sha256(PARENT_RECEIPT)
    serialized = PARENT_RECEIPT.read_bytes()
    after = file_sha256(PARENT_RECEIPT)
    if before != PARENT_RECEIPT_SHA256 or after != before or hashlib.sha256(
        serialized
    ).hexdigest() != before:
        raise RuntimeError("canonical FineWeb parent receipt changed")
    receipt = json.loads(serialized)
    gate = receipt.get("ordered_manifest_local_parquet_identity_gate", {})
    if receipt.get("authorized_for_scored_experiments") is not True or (
        gate.get("revision") != PINNED_REVISION
        or gate.get("ordered_manifest_sha256") != PINNED_MANIFEST_SHA256
        or gate.get("source_sha256") != PINNED_PARQUET_SHA256
    ):
        raise RuntimeError("canonical FineWeb parent authority changed")
    entries = receipt.get("entries")
    provenance = receipt.get("document_provenance", {})
    if not isinstance(entries, dict) or provenance.get("schema_version") != 1 or not isinstance(
        provenance.get("sets"), dict
    ):
        raise RuntimeError("canonical FineWeb parent metadata is incomplete")
    for key, expected in SOURCE_CACHES.items():
        entry = entries.get(key)
        if not isinstance(entry, dict) or tuple(entry.get("shape", ())) != expected["shape"] or (
            entry.get("dtype") != "torch.int64"
            or entry.get("tensor_raw_sha256") != expected["tensor_sha256"]
            or Path(entry.get("cache_path", "")) != expected["path"]
        ):
            raise RuntimeError(f"canonical FineWeb cache entry changed: {key}")
    return receipt


def _record(source: str, index: int, value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"document_id", "dataset_document_index", "chunk_id", "token_start"}
    if not isinstance(value, Mapping) or not required.issubset(value) or not isinstance(
        value["document_id"], str
    ) or not value["document_id"] or any(
        isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 0
        for name in required - {"document_id"}
    ):
        raise RuntimeError(f"malformed provenance record in {source} at {index}")
    return {
        "source_key": source,
        "source_row_index": index,
        "document_id": value["document_id"],
        "dataset_document_index": value["dataset_document_index"],
        "chunk_id": value["chunk_id"],
        "token_start": value["token_start"],
    }


def build_selection_plan(receipt: Mapping[str, Any]) -> dict[str, Any]:
    sets = receipt.get("document_provenance", {}).get("sets", {})
    used: set[str] = set()
    roles: dict[str, list[dict[str, Any]]] = {}
    for role in ("basis", "fit", "evaluation"):
        chosen: list[dict[str, Any]] = []
        for source in ROLE_POOLS[role]:
            rows = sets.get(source)
            if not isinstance(rows, list) or len(rows) != SOURCE_CACHES[source]["shape"][0]:
                raise RuntimeError(f"provenance pool is incomplete: {source}")
            seen_in_pool: set[str] = set()
            for index, raw in enumerate(rows):
                record = _record(source, index, raw)
                document = record["document_id"]
                if document in used or document in seen_in_pool:
                    continue
                seen_in_pool.add(document)
                record["role_index"] = len(chosen)
                chosen.append(record)
                if len(chosen) == ROLE_SIZES[role]:
                    break
            if len(chosen) == ROLE_SIZES[role]:
                break
        if len(chosen) != ROLE_SIZES[role]:
            raise RuntimeError(f"unique-document pool is exhausted for {role}: {len(chosen)}")
        documents = {record["document_id"] for record in chosen}
        if len(documents) != len(chosen) or documents & used:
            raise RuntimeError("selection plan repeats a document")
        used.update(documents)
        roles[role] = chosen
    contributions = {
        role: dict(sorted(Counter(record["source_key"] for record in records).items()))
        for role, records in roles.items()
    }
    if contributions != EXPECTED_CONTRIBUTIONS or len(used) != sum(ROLE_SIZES.values()):
        raise RuntimeError("unique-row selection differs from its frozen known answer")
    if max(record["dataset_document_index"] for record in roles["fit"]) >= min(
        record["dataset_document_index"] for record in roles["evaluation"]
    ):
        raise RuntimeError("evaluation documents are not later than fit documents")
    body = {
        "schema": "gauge_transport_triangle_unique_rows_v1_selection_plan",
        "role_sizes": ROLE_SIZES,
        "role_pool_order": {key: list(value) for key, value in ROLE_POOLS.items()},
        "source_contributions": contributions,
        "roles": roles,
        "unique_document_count": len(used),
    }
    return {**body, "selection_plan_sha256": canonical_sha256(body)}


def build_authority(source: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    plan = build_selection_plan(receipt)
    body = {
        "schema": "gauge_transport_triangle_unique_rows_v1_authority",
        "status": "frozen_before_any_row_tensor_model_or_outcome_load",
        "source_closure": dict(source),
        "parent_receipt_path": str(PARENT_RECEIPT),
        "parent_receipt_file_sha256": PARENT_RECEIPT_SHA256,
        "selection_plan": plan,
        "cache_bindings": {
            key: {
                "path": str(value["path"]),
                "shape": list(value["shape"]),
                "dtype": "torch.int64",
                "file_sha256": value["file_sha256"],
                "tensor_sha256": value["tensor_sha256"],
            }
            for key, value in SOURCE_CACHES.items()
        },
        "permissions": {
            "row_materialization_from_pinned_caches_only": True,
            "triangle_runner_authorized_by_this_authority": False,
            "conditional_future_row_eligibility": (
                "only after receipt-last materialization and a separately frozen, "
                "source-closed triangle runner authority"
            ),
            "model_or_response_computed": False,
            "training_authorized": False,
            "global_ledger_credit": False,
        },
        "outputs": {
            "authority": str(AUTHORITY),
            "rows": str(ROWS), "manifest": str(MANIFEST),
            "receipt": str(RECEIPT), "failure": str(FAILURE), "lock": str(LOCK),
        },
    }
    return {**body, "authority_sha256": canonical_sha256(body)}


def validate_authority(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or value.get(
        "schema"
    ) != "gauge_transport_triangle_unique_rows_v1_authority" or value.get(
        "status"
    ) != "frozen_before_any_row_tensor_model_or_outcome_load":
        raise RuntimeError("unique-row authority header changed")
    if set(value) != {
        "schema", "status", "source_closure", "parent_receipt_path",
        "parent_receipt_file_sha256", "selection_plan", "cache_bindings",
        "permissions", "outputs", "authority_sha256",
    }:
        raise RuntimeError("unique-row authority schema changed")
    body = {key: value[key] for key in value if key != "authority_sha256"}
    if canonical_sha256(body) != value.get("authority_sha256"):
        raise RuntimeError("unique-row authority hash changed")
    validate_source_closure(value["source_closure"])
    parent = load_parent_metadata()
    rebuilt = build_authority(value["source_closure"], parent)
    if dict(value) != rebuilt:
        raise RuntimeError("unique-row authority differs from exact rebuilt authority")


def load_cache_tensors(authority: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    output = {}
    for key, expected in authority["cache_bindings"].items():
        path = Path(expected["path"])
        before = file_sha256(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        if file_sha256(path) != before or before != expected["file_sha256"] or not torch.is_tensor(
            value
        ) or value.dtype != torch.long or tuple(value.shape) != tuple(expected["shape"]) or (
            tensor_sha256(value) != expected["tensor_sha256"]
        ):
            raise RuntimeError(f"unique-row cache tensor changed: {key}")
        output[key] = value.contiguous()
    return output


def build_rows_payload(
    authority: Mapping[str, Any], caches: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    role_tensors = {}
    role_records = {}
    for role, records in authority["selection_plan"]["roles"].items():
        rows = torch.stack([
            caches[record["source_key"]][record["source_row_index"]].clone()
            for record in records
        ]).contiguous()
        role_tensors[role] = rows
        role_records[role] = [dict(record) for record in records]
    payload = {
        "schema": "gauge_transport_triangle_unique_rows_v1_rows",
        "authority_sha256": authority["authority_sha256"],
        "selection_plan_sha256": authority["selection_plan"]["selection_plan_sha256"],
        "roles": role_tensors,
        "records": role_records,
    }
    validate_rows_payload(payload, authority)
    return payload


def validate_rows_payload(payload: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema", "authority_sha256", "selection_plan_sha256", "roles", "records",
    } or payload["schema"] != "gauge_transport_triangle_unique_rows_v1_rows" or (
        payload["authority_sha256"] != authority["authority_sha256"]
        or payload["selection_plan_sha256"] != authority["selection_plan"]["selection_plan_sha256"]
        or set(payload["roles"]) != set(ROLE_SIZES)
        or set(payload["records"]) != set(ROLE_SIZES)
    ):
        raise RuntimeError("unique-row payload header changed")
    documents = []
    for role, size in ROLE_SIZES.items():
        tensor = payload["roles"][role]
        records = payload["records"][role]
        if not torch.is_tensor(tensor) or tensor.dtype != torch.long or tuple(tensor.shape) != (
            size, TOKEN_LENGTH
        ) or records != authority["selection_plan"]["roles"][role]:
            raise RuntimeError(f"unique-row payload role changed: {role}")
        documents.extend(record["document_id"] for record in records)
    if len(documents) != len(set(documents)):
        raise RuntimeError("unique-row payload repeats a source document")


def build_manifest(payload: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": "gauge_transport_triangle_unique_rows_v1_manifest",
        "authority_sha256": authority["authority_sha256"],
        "selection_plan_sha256": payload["selection_plan_sha256"],
        "parent_receipt_file_sha256": PARENT_RECEIPT_SHA256,
        "rows_file_sha256": file_sha256(ROWS),
        "role_tensor_sha256s": {
            role: tensor_sha256(payload["roles"][role]) for role in ROLE_SIZES
        },
        "role_record_sha256s": {
            role: canonical_sha256(payload["records"][role]) for role in ROLE_SIZES
        },
        "role_sizes": ROLE_SIZES,
        "source_contributions": EXPECTED_CONTRIBUTIONS,
        "unique_document_count": sum(ROLE_SIZES.values()),
        "triangle_runner_authorized_by_this_manifest": False,
        "conditional_future_row_eligibility": authority["permissions"][
            "conditional_future_row_eligibility"
        ],
        "authorized_for_training": False,
        "authorized_for_global_ledger_credit": False,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def validate_manifest(
    value: Mapping[str, Any], payload: Mapping[str, Any], authority: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or value.get(
        "schema"
    ) != "gauge_transport_triangle_unique_rows_v1_manifest":
        raise RuntimeError("unique-row manifest header changed")
    body = {key: value[key] for key in value if key != "manifest_sha256"}
    if canonical_sha256(body) != value.get("manifest_sha256") or dict(value) != build_manifest(
        payload, authority
    ):
        raise RuntimeError("unique-row manifest differs from exact rebuilt manifest")


def _payload_equals(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        {key: left[key] for key in left if key != "roles"}
        == {key: right[key] for key in right if key != "roles"}
        and set(left.get("roles", {})) == set(right.get("roles", {})) == set(ROLE_SIZES)
        and all(torch.equal(left["roles"][role], right["roles"][role]) for role in ROLE_SIZES)
    )


def replay_terminal_state(
    expected_authority: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority_bytes_before = file_sha256(AUTHORITY)
    authority = json.loads(AUTHORITY.read_text())
    if file_sha256(AUTHORITY) != authority_bytes_before:
        raise RuntimeError("unique-row authority changed during replay")
    validate_authority(authority)
    if expected_authority is not None and dict(authority) != dict(expected_authority):
        raise RuntimeError("unique-row authority identity changed")

    caches = load_cache_tensors(authority)
    rows_bytes_before = file_sha256(ROWS)
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if file_sha256(ROWS) != rows_bytes_before:
        raise RuntimeError("unique-row artifact changed during replay")
    validate_rows_payload(payload, authority)
    rebuilt_payload = build_rows_payload(authority, caches)
    if not _payload_equals(payload, rebuilt_payload):
        raise RuntimeError("unique-row artifact differs from pinned cache rows")

    manifest_bytes_before = file_sha256(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    if file_sha256(MANIFEST) != manifest_bytes_before:
        raise RuntimeError("unique-row manifest changed during replay")
    validate_manifest(manifest, payload, authority)
    replay = {
        "authority_file_sha256": authority_bytes_before,
        "rows_file_sha256": rows_bytes_before,
        "manifest_file_sha256": manifest_bytes_before,
        "cache_file_sha256s": {
            key: file_sha256(Path(binding["path"]))
            for key, binding in authority["cache_bindings"].items()
        },
    }
    if replay["cache_file_sha256s"] != {
        key: binding["file_sha256"] for key, binding in authority["cache_bindings"].items()
    }:
        raise RuntimeError("unique-row cache bytes changed after tensor replay")
    # Close the replay window by rechecking the source/parent-derived authority and
    # every terminal byte identity once more, after all semantic reconstruction.
    validate_authority(authority)
    if json.loads(AUTHORITY.read_text()) != authority or {
        "authority_file_sha256": file_sha256(AUTHORITY),
        "rows_file_sha256": file_sha256(ROWS),
        "manifest_file_sha256": file_sha256(MANIFEST),
        "cache_file_sha256s": {
            key: file_sha256(Path(binding["path"]))
            for key, binding in authority["cache_bindings"].items()
        },
    } != replay:
        raise RuntimeError("unique-row terminal bytes changed after semantic replay")
    return payload, manifest, replay


def build_receipt(
    authority: Mapping[str, Any], payload: Mapping[str, Any], manifest: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    if FAILURE.exists():
        raise RuntimeError("unique-row failure exists before receipt construction")
    body = {
        "schema": "gauge_transport_triangle_unique_rows_v1_receipt",
        "status": "complete_unique_document_rows_receipt_last",
        "authority_file_sha256": replay["authority_file_sha256"],
        "authority_sha256": authority["authority_sha256"],
        "source_closure_sha256": authority["source_closure"]["sha256"],
        "parent_receipt_file_sha256": PARENT_RECEIPT_SHA256,
        "cache_file_sha256s": dict(replay["cache_file_sha256s"]),
        "rows_file_sha256": replay["rows_file_sha256"],
        "role_tensor_sha256s": {
            role: tensor_sha256(payload["roles"][role]) for role in ROLE_SIZES
        },
        "manifest_file_sha256": replay["manifest_file_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "selection_plan_sha256": payload["selection_plan_sha256"],
        "unique_document_count": sum(ROLE_SIZES.values()),
        "failure_absent": True,
        "triangle_runner_authorized_by_this_receipt": False,
        "global_ledger_credit": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def validate_receipt(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or value.get(
        "schema"
    ) != "gauge_transport_triangle_unique_rows_v1_receipt" or value.get(
        "status"
    ) != "complete_unique_document_rows_receipt_last":
        raise RuntimeError("unique-row receipt header changed")
    if value.get("failure_absent") is not True or FAILURE.exists():
        raise RuntimeError("unique-row receipt/failure exclusivity changed")
    body = {key: value[key] for key in value if key != "receipt_sha256"}
    if canonical_sha256(body) != value.get("receipt_sha256"):
        raise RuntimeError("unique-row receipt hash changed")
    authority = json.loads(AUTHORITY.read_text())
    payload, manifest, replay = replay_terminal_state(authority)
    if dict(value) != build_receipt(authority, payload, manifest, replay):
        raise RuntimeError("unique-row receipt differs from exact terminal replay")


def freeze_authority() -> dict[str, Any]:
    if any(path.exists() for path in (AUTHORITY, ROWS, MANIFEST, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("unique-row authority namespace is not pristine")
    owner = acquire_owner_lock()
    try:
        if any(path.exists() for path in (AUTHORITY, ROWS, MANIFEST, RECEIPT, FAILURE)):
            raise RuntimeError("unique-row authority namespace changed after lock acquisition")
        value = build_authority(source_closure(), load_parent_metadata())
        publish_json(AUTHORITY, value, owner=owner)
        assert_owner_lock(owner)
        reloaded = json.loads(AUTHORITY.read_text())
        validate_authority(reloaded)
        if reloaded != value:
            raise RuntimeError("unique-row authority reload changed")
        return value
    finally:
        release_owner_lock(owner)


def materialize() -> dict[str, Any]:
    if not AUTHORITY.is_file() or any(path.exists() for path in (ROWS, MANIFEST, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("unique-row materialization namespace is unavailable")
    owner = acquire_owner_lock()
    try:
        if any(path.exists() for path in (ROWS, MANIFEST, RECEIPT, FAILURE)):
            raise RuntimeError("unique-row materialization namespace changed after lock acquisition")
        authority = json.loads(AUTHORITY.read_text())
        validate_authority(authority)
        payload = build_rows_payload(authority, load_cache_tensors(authority))
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        _publish_create_only(ROWS, buffer.getvalue(), owner=owner)
        manifest = build_manifest(payload, authority)
        publish_json(MANIFEST, manifest, owner=owner)
        assert_owner_lock(owner)
        replayed_payload, replayed_manifest, replay = replay_terminal_state(authority)
        receipt = build_receipt(authority, replayed_payload, replayed_manifest, replay)
        publish_json(RECEIPT, receipt, owner=owner)
        assert_owner_lock(owner)
        validate_receipt(json.loads(RECEIPT.read_text()))
        return receipt
    except BaseException as error:
        try:
            assert_owner_lock(owner)
            # The receipt is the terminal, receipt-last success marker.  Once its
            # atomic hard link exists, a later validation/fsync/cleanup exception
            # must not create a contradictory terminal failure artifact.
            if not RECEIPT.exists() and not FAILURE.exists():
                publish_json(FAILURE, {
                    "schema": "gauge_transport_triangle_unique_rows_v1_failure",
                    "status": "terminal_failure_no_receipt",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "rows_exists": ROWS.exists(),
                    "manifest_exists": MANIFEST.exists(),
                    "receipt_exists": RECEIPT.exists(),
                }, owner=owner)
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
    print(json.dumps({
        "status": result["status"],
        "path": str(AUTHORITY if arguments.freeze_authority else RECEIPT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
