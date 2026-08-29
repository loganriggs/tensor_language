#!/usr/bin/env python3
"""Create-only, outcome-blind document-role assignment for MLP2 CMR v1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / "MLP2_CMR_V1_ROW_PROTOCOL.json"
PREREG = HERE / "MLP2_CMR_V1_PREREGISTRATION.md"
ORDERED_RECEIPT = HERE.parent / "bilinear_quotient/.rowcache/fineweb_oracle_v2_receipt.json"
PARQUET = Path("/workspace/fineweb_pinned/data/CC-MAIN-2013-20/000_00000.parquet")
MANIFEST = HERE / "mlp2_cmr_v1_document_roles_manifest.json"
RECEIPT = HERE / "mlp2_cmr_v1_document_roles_receipt.json"
LOCK = HERE / ".mlp2_cmr_v1_document_roles.lock"
EXPERIMENT_ID = "bilin18_mlp2_cmr_v1"
ROLES = ("FIT_MEAN", "FIT_SELECTOR", "VALIDATION", "REPLICATION")
DOCUMENTS_PER_ROLE = 192
SOURCE_CLOSURE = (PROTOCOL, PREREG, Path(__file__).resolve(),
                  HERE / "test_freeze_mlp2_cmr_v1_document_roles.py")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def ordering_digest(index: int) -> bytes:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("document index must be a nonnegative integer")
    return hashlib.sha256(
        EXPERIMENT_ID.encode() + b"\0" + str(index).encode("ascii")
    ).digest()


def assign_roles(
    parquet_rows: int, exclusions: Iterable[int],
) -> dict[str, tuple[int, ...]]:
    excluded = set(exclusions)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= parquet_rows
           for x in excluded):
        raise ValueError("registry exclusion is outside the parquet")
    needed = len(ROLES) * DOCUMENTS_PER_ROLE
    ordered = sorted(
        (index for index in range(parquet_rows) if index not in excluded),
        key=lambda index: (ordering_digest(index), index),
    )[:needed]
    if len(ordered) != needed:
        raise RuntimeError("not enough unspent source documents")
    return {
        role: tuple(ordered[offset:offset + DOCUMENTS_PER_ROLE])
        for role, offset in zip(ROLES, range(0, needed, DOCUMENTS_PER_ROLE), strict=True)
    }


def _walk(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk(child)


def registry_snapshot(protocol: Mapping[str, Any]) -> tuple[set[int], dict[str, str]]:
    files = set()
    for path in ROOT.rglob("*.json"):
        if path.resolve() in {MANIFEST.resolve(), RECEIPT.resolve()}:
            continue
        if any(word in path.name.lower() for word in ("receipt", "manifest", "authority")):
            files.add(path.resolve())
    files.add(ORDERED_RECEIPT.resolve())
    exclusions: set[int] = set()
    hashes: dict[str, str] = {}
    source = protocol["fineweb_source"]
    prefix = f"{source['revision']}:{source['relative_path']}:"
    for path in sorted(files):
        before = file_sha256(path)
        payload = json.loads(path.read_text())
        if file_sha256(path) != before:
            raise RuntimeError(f"registry changed while reading: {path}")
        hashes[str(path)] = before
        for key, value in _walk(payload):
            if key in ("dataset_document_index", "source_document_index"):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise RuntimeError(f"malformed document index in {path}")
                exclusions.add(value)
            elif key == "ordered_document_indices" or (
                key in ("document_indices", "source_document_indices",
                        "dataset_document_indices") and isinstance(value, list)
            ):
                if not isinstance(value, list):
                    raise RuntimeError(f"malformed ordered document-index list in {path}")
                for item in value:
                    if isinstance(item, bool) or not isinstance(item, int):
                        raise RuntimeError(f"malformed document-index item in {path}")
                    exclusions.add(item)
            elif key == "document_id" and isinstance(value, str) and value.startswith(prefix):
                suffix = value[len(prefix):]
                if not suffix.isascii() or not suffix.isdecimal() or (
                    suffix.startswith("0") and suffix != "0"
                ):
                    raise RuntimeError(f"malformed canonical document ID in {path}")
                exclusions.add(int(suffix))
    if any(index < 0 or index >= source["parquet_rows"] for index in exclusions):
        raise RuntimeError("registry exclusion exceeds the pinned parquet")
    return exclusions, hashes


def write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def committed_source_identity() -> tuple[str, dict[str, str]]:
    commit = git("rev-parse", "HEAD")
    if commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD and origin/main differ")
    hashes = {}
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != file_sha256(path):
            raise RuntimeError(f"source is not committed: {relative}")
        hashes[str(relative)] = file_sha256(path)
    return commit, hashes


def validate_fixed_inputs(protocol: Mapping[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("protocol identity changed")
    if tuple(protocol["selection"]["roles"]) != ROLES or \
            protocol["selection"]["documents_per_role"] != DOCUMENTS_PER_ROLE:
        raise RuntimeError("protocol role specification changed")
    parent = protocol["ordered_fineweb_receipt"]
    if file_sha256(ROOT / parent["path"]) != parent["sha256"]:
        raise RuntimeError("ordered FineWeb receipt changed")
    source = protocol["fineweb_source"]
    stat = PARQUET.stat()
    if stat.st_size != source["size"] or file_sha256(PARQUET) != source["sha256"]:
        raise RuntimeError("pinned FineWeb bytes changed")


def freeze() -> dict[str, Any]:
    if MANIFEST.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite the document-role namespace")
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        protocol = json.loads(PROTOCOL.read_text())
        validate_fixed_inputs(protocol)
        commit, source_hashes = committed_source_identity()
        exclusions, registry_hashes = registry_snapshot(protocol)
        roles = assign_roles(protocol["fineweb_source"]["parquet_rows"], exclusions)
        flattened = [index for role in ROLES for index in roles[role]]
        manifest = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "document_roles_assigned_outcome_blind",
            "authority": "none_until_receipt",
            "roles": {
                role: {
                    "ordered_document_indices": list(indices),
                    "ordered_document_indices_sha256": canonical_hash(list(indices)),
                    "document_count": len(indices),
                } for role, indices in roles.items()
            },
            "cross_role_document_disjoint": len(flattened) == len(set(flattened)),
            "all_indices_sha256": canonical_hash(flattened),
            "registry_file_hashes": registry_hashes,
            "excluded_document_count": len(exclusions),
            "excluded_indices_sha256": canonical_hash(sorted(exclusions)),
            "protocol_sha256": file_sha256(PROTOCOL),
            "source_commit": commit,
            "source_hashes": source_hashes,
            "authorized_for_tokenization": False,
            "authorized_for_model_forward": False,
            "authorized_for_scientific_outcomes": False
        }
        if not manifest["cross_role_document_disjoint"]:
            raise RuntimeError("role allocation is not document-disjoint")
        write_create_only(MANIFEST, manifest)
        receipt = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "fresh_document_identity_roles_published_receipt_last",
            "authority": "document_role_identity_only",
            "manifest_path": str(MANIFEST.relative_to(ROOT)),
            "manifest_sha256": file_sha256(MANIFEST),
            "protocol_sha256": file_sha256(PROTOCOL),
            "source_commit": commit,
            "source_hashes": source_hashes,
            "registry_file_hashes": registry_hashes,
            "authorized_for_document_identity": True,
            "authorized_for_tokenization": False,
            "authorized_for_model_forward": False,
            "authorized_for_fit_or_evaluation": False,
            "authorized_for_scientific_outcomes": False,
            "next_required_authority": (
                "source-closed token materialization receipt binding ordered "
                "document indices, token bytes, position masks, and short-document handling"
            )
        }
        # Fail if source or any registry identity moved between selection and receipt.
        if git("rev-parse", "HEAD") != commit or git("rev-parse", "origin/main") != commit:
            raise RuntimeError("source commit changed during role assignment")
        current_exclusions, current_hashes = registry_snapshot(protocol)
        if current_exclusions != exclusions or current_hashes != registry_hashes:
            raise RuntimeError("registry snapshot changed during role assignment")
        if file_sha256(MANIFEST) != receipt["manifest_sha256"]:
            raise RuntimeError("manifest changed before receipt publication")
        write_create_only(RECEIPT, receipt)
        return receipt
    finally:
        os.close(lock_fd)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
