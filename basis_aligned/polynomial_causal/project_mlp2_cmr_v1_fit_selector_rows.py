#!/usr/bin/env python3
"""Project the combined MLP2 CMR token container to spent FIT_SELECTOR only."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADDENDUM = HERE / "MLP2_CMR_V1_MARGIN_FREQUENCY_ADDENDUM.md"
MATERIALIZER = HERE / "materialize_mlp2_cmr_v1_token_rows.py"
MATERIALIZER_TEST = HERE / "test_materialize_mlp2_cmr_v1_token_rows.py"
COMBINED = HERE / "mlp2_cmr_v1_token_rows.pt"
COMBINED_MANIFEST = HERE / "mlp2_cmr_v1_token_rows_manifest.json"
COMBINED_RECEIPT = HERE / "mlp2_cmr_v1_token_rows_receipt.json"
OUTPUT = HERE / "mlp2_cmr_v1_fit_selector_rows.pt"
MANIFEST = HERE / "mlp2_cmr_v1_fit_selector_rows_manifest.json"
RECEIPT = HERE / "mlp2_cmr_v1_fit_selector_rows_receipt.json"
LOCK = HERE / ".mlp2_cmr_v1_fit_selector_rows.lock"

COMBINED_SHA256 = "3ed0192993095f7de70ab7f1350d091b6c1d8c4c7d0583fd5f0f6441556e4aa6"
COMBINED_MANIFEST_SHA256 = "8b8f3155a21b73af8b89278b9f09c60bf82fd965a7723e046e191415c5d57bb4"
COMBINED_RECEIPT_SHA256 = "47113c255bf47f9d1c7369639fab39664c71f93134099babadcce9d89a011e85"
EXPECTED_TENSOR_HASHES = {
    "document_indices": "c5216431ed3351a5532bc7316f5660019385a9c05a92a4d98ed9fa762c30fd05",
    "rows": "cfb5403c5e5cf5aabba97ecad0ad5c7915e2f239884152ef3a2a5d44a7a9e465",
    "eligible_mask": "fa21dfe87d37bf61bf2fcb8d3f60d6187208c0197d87a351b9571ac56136a3f3",
    "original_token_counts": "70ec29170505ab0de9d7291c3b46871a02dabedbd21795b4feff8c7a2f8f4925",
    "clipped_token_counts": "eb9af2130ad2ee795d1a46ec585259bf949289b09e7764004bdb169b6caaa363",
}
DOCUMENTS = 192
WIDTH = 257
SEQUENCE = 256
SCORE_START = 64
EOT = 50_256
ELIGIBLE_POSITIONS = 31_505
SUPPORT_DOCUMENTS = 191
ALL_FALSE_ORDINAL = 82

SOURCE_CLOSURE = (
    ADDENDUM, MATERIALIZER, MATERIALIZER_TEST, Path(__file__).resolve(),
    HERE / "test_project_mlp2_cmr_v1_fit_selector_rows.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_create_only(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_torch_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True,
    )
    hashes: dict[str, str] = {}
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = sha256_bytes(blob)
        if file_sha256(path) != digest:
            raise RuntimeError(f"projection source differs from committed bytes: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def validate_role(role: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    expected = {
        "document_indices": ((DOCUMENTS,), torch.long),
        "rows": ((DOCUMENTS, WIDTH), torch.long),
        "eligible_mask": ((DOCUMENTS, SEQUENCE), torch.bool),
        "original_token_counts": ((DOCUMENTS,), torch.long),
        "clipped_token_counts": ((DOCUMENTS,), torch.long),
    }
    if set(role) != set(expected):
        raise RuntimeError("FIT_SELECTOR role keys changed")
    for name, (shape, dtype) in expected.items():
        value = role[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
            raise RuntimeError(f"FIT_SELECTOR tensor is malformed: {name}")
    hashes = {name: tensor_sha256(value) for name, value in role.items()}
    if hashes != EXPECTED_TENSOR_HASHES:
        raise RuntimeError("FIT_SELECTOR tensor identity changed")
    rows = role["rows"]
    mask = role["eligible_mask"]
    original = role["original_token_counts"]
    clipped = role["clipped_token_counts"]
    if len(set(role["document_indices"].tolist())) != DOCUMENTS or int(rows.min()) < 0 or int(
        rows.max()
    ) > EOT or not torch.equal(clipped, original.clamp(max=WIDTH)):
        raise RuntimeError("FIT_SELECTOR token/document census changed")
    positions = torch.arange(SEQUENCE).unsqueeze(0)
    expected_mask = (positions >= SCORE_START) & (
        positions < (clipped - 1).clamp_min(0)[:, None]
    )
    if not torch.equal(mask, expected_mask) or bool(mask[:, :SCORE_START].any()):
        raise RuntimeError("FIT_SELECTOR eligibility semantics changed")
    for ordinal, count in enumerate(clipped.tolist()):
        if count < WIDTH and not bool((rows[ordinal, count:] == EOT).all()):
            raise RuntimeError("FIT_SELECTOR short row padding changed")
    all_false = torch.nonzero(~mask.any(1), as_tuple=False).flatten().tolist()
    if int(mask.sum()) != ELIGIBLE_POSITIONS or int(mask.any(1).sum()) != (
        SUPPORT_DOCUMENTS
    ) or all_false != [ALL_FALSE_ORDINAL]:
        raise RuntimeError("FIT_SELECTOR support census changed")
    return {
        "documents": DOCUMENTS,
        "eligible_positions": ELIGIBLE_POSITIONS,
        "support_documents": SUPPORT_DOCUMENTS,
        "all_false_ordinals": all_false,
        "tensor_hashes": hashes,
    }


def parent_snapshot() -> tuple[dict[str, str], dict[str, bytes]]:
    paths = {
        "combined": COMBINED,
        "combined_manifest": COMBINED_MANIFEST,
        "combined_receipt": COMBINED_RECEIPT,
    }
    expected = {
        "combined": COMBINED_SHA256,
        "combined_manifest": COMBINED_MANIFEST_SHA256,
        "combined_receipt": COMBINED_RECEIPT_SHA256,
    }
    captured = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: sha256_bytes(value) for name, value in captured.items()}
    if hashes != expected:
        raise RuntimeError("combined token parent changed")
    manifest = json.loads(captured["combined_manifest"])
    receipt = json.loads(captured["combined_receipt"])
    if manifest.get("output_file_sha256") != COMBINED_SHA256 or (
        manifest.get("tensor_hashes", {}).get("FIT_SELECTOR") != EXPECTED_TENSOR_HASHES
        or receipt.get("output_file_sha256") != COMBINED_SHA256
        or receipt.get("manifest_sha256") != COMBINED_MANIFEST_SHA256
        or receipt.get("authorized_for_token_inputs") is not True
        or receipt.get("authorized_for_model_forward") is not False
    ):
        raise RuntimeError("combined token receipt/manifest joins changed")
    return hashes, captured


def validate_claim(nonce: str, inode: tuple[int, int]) -> None:
    descriptor = os.open(LOCK, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(descriptor)
        payload = os.read(descriptor, 4096)
    finally:
        os.close(descriptor)
    if (stat.st_dev, stat.st_ino) != inode or json.loads(payload).get("nonce") != nonce:
        raise RuntimeError("FIT_SELECTOR projection claim changed")


def final_guard(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], output_hash: str, manifest_hash: str,
) -> None:
    validate_claim(nonce, inode)
    for relative, expected in source_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError("projection source changed during execution")
    current, _ = parent_snapshot()
    if current != parents or file_sha256(OUTPUT) != output_hash or file_sha256(
        MANIFEST
    ) != manifest_hash or RECEIPT.exists():
        raise RuntimeError("FIT_SELECTOR projection terminal snapshot changed")


def project() -> dict[str, Any]:
    if any(path.exists() for path in (OUTPUT, MANIFEST, RECEIPT, LOCK)):
        raise RuntimeError("FIT_SELECTOR projection namespace already exists")
    commit, source_hashes = committed_source()
    parents, captured = parent_snapshot()
    nonce = secrets.token_hex(32)
    write_create_only(LOCK, canonical_json_bytes({"nonce": nonce, "purpose": "fit_selector_projection"}))
    stat = LOCK.stat(follow_symlinks=False)
    inode = (stat.st_dev, stat.st_ino)
    validate_claim(nonce, inode)
    combined = torch.load(io.BytesIO(captured["combined"]), map_location="cpu", weights_only=True)
    if set(combined) != {"FIT_MEAN", "FIT_SELECTOR", "VALIDATION", "REPLICATION"}:
        raise RuntimeError("combined token role set changed")
    role = {name: value.detach().cpu().clone().contiguous() for name, value in combined[
        "FIT_SELECTOR"
    ].items()}
    del combined, captured
    summary = validate_role(role)
    publish_torch_create_only(OUTPUT, role)
    output_hash = file_sha256(OUTPUT)
    manifest = {
        "schema_version": 1,
        "experiment_id": "bilin18_mlp2_cmr_v1_fit_selector_projection",
        "status": "fit_selector_role_only_published_pending_receipt",
        "source_commit": commit,
        "source_hashes": source_hashes,
        "parents": parents,
        "output_path": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": output_hash,
        "summary": summary,
        "contains_roles": ["FIT_SELECTOR"],
        "combined_container_unavoidably_deserialized_without_model": True,
        "model_loaded": False,
        "scientific_outcome_computed": False,
    }
    write_create_only(MANIFEST, canonical_json_bytes(manifest))
    manifest_hash = file_sha256(MANIFEST)
    replay = torch.load(OUTPUT, map_location="cpu", weights_only=True)
    if validate_role(replay) != summary or set(replay) != set(role):
        raise RuntimeError("FIT_SELECTOR projection semantic replay failed")
    receipt = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": "fit_selector_role_only_projection_complete_receipt_last",
        "source_commit": commit,
        "source_hashes": source_hashes,
        "parents": parents,
        "output_sha256": output_hash,
        "manifest_sha256": manifest_hash,
        "summary": summary,
        "authorized_for_fit_selector_calibration_input": True,
        "authorized_for_model_forward": False,
        "authorized_for_validation": False,
        "authorized_for_replication": False,
    }
    final_guard(source_hashes, parents, nonce, inode, output_hash, manifest_hash)
    validate_claim(nonce, inode)
    write_create_only(RECEIPT, canonical_json_bytes(receipt))
    return receipt


if __name__ == "__main__":
    print(json.dumps(project(), indent=2, sort_keys=True))
