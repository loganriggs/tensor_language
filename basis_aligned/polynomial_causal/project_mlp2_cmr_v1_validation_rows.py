#!/usr/bin/env python3
"""Project the combined MLP2 CMR token container to VALIDATION only, without a model."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Callable, Mapping

import torch

import project_mlp2_cmr_v1_fit_selector_rows as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "MLP2_CMR_V1_PREREGISTRATION.md"
ADDENDUM = HERE / "MLP2_CMR_V1_VALIDATION_ADDENDUM.md"
OUTPUT = HERE / "mlp2_cmr_v1_validation_rows.pt"
MANIFEST = HERE / "mlp2_cmr_v1_validation_rows_manifest.json"
RECEIPT = HERE / "mlp2_cmr_v1_validation_rows_receipt.json"
FAILURE = HERE / "mlp2_cmr_v1_validation_rows_failure.json"
LOCK = HERE / ".mlp2_cmr_v1_validation_rows.lock"

DOCUMENTS = 192
WIDTH = 257
SEQUENCE = 256
SCORE_START = 64
EOT = 50_256
ELIGIBLE_POSITIONS = 29_904
SUPPORT_DOCUMENTS = 191
EXPECTED_TENSOR_HASHES = {
    "document_indices": "9eace2bab6f1a63932cc768db5c78a6909f8eb1f3cef38f499f47e7799375680",
    "rows": "9a8085af575a9c1c41164b4a7daf25e7e7ff9f54d27cc035ce3986f9a0a0f7b8",
    "eligible_mask": "51a06f0382910e318a372896bf0c0501f64a9b6ff0307742cd13b9152bc2d152",
    "original_token_counts": "48e718663ae6178253fe66136468a4bb027bd7773c75617409b5c44fbaccd7a8",
    "clipped_token_counts": "5119f41ab72fb5bab25394bae9027263dde22b6d4a641d06b7445510899f249d",
}

SOURCE_CLOSURE = (
    PREREG, ADDENDUM, Path(__file__).resolve(),
    HERE / "test_project_mlp2_cmr_v1_validation_rows.py",
    *tuple(Path(path).resolve() for path in base.SOURCE_CLOSURE),
)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True,
    )
    hashes: dict[str, str] = {}
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if base.file_sha256(path) != digest:
            raise RuntimeError(f"validation projection source differs from commit: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def validate_role(
    role: Mapping[str, torch.Tensor], *, require_identity: bool = True,
) -> dict[str, Any]:
    expected = {
        "document_indices": ((DOCUMENTS,), torch.long),
        "rows": ((DOCUMENTS, WIDTH), torch.long),
        "eligible_mask": ((DOCUMENTS, SEQUENCE), torch.bool),
        "original_token_counts": ((DOCUMENTS,), torch.long),
        "clipped_token_counts": ((DOCUMENTS,), torch.long),
    }
    if set(role) != set(expected):
        raise RuntimeError("VALIDATION role keys changed")
    for name, (shape, dtype) in expected.items():
        value = role[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
            raise RuntimeError(f"VALIDATION tensor is malformed: {name}")
    hashes = {name: base.tensor_sha256(value) for name, value in role.items()}
    if require_identity and hashes != EXPECTED_TENSOR_HASHES:
        raise RuntimeError("VALIDATION tensor identity changed")
    rows = role["rows"]
    eligible = role["eligible_mask"]
    original = role["original_token_counts"]
    clipped = role["clipped_token_counts"]
    if len(set(role["document_indices"].tolist())) != DOCUMENTS or int(rows.min()) < 0 or int(
        rows.max()
    ) > EOT or not torch.equal(clipped, original.clamp(max=WIDTH)):
        raise RuntimeError("VALIDATION token/document census changed")
    positions = torch.arange(SEQUENCE).unsqueeze(0)
    expected_mask = (positions >= SCORE_START) & (
        positions < (clipped - 1).clamp_min(0)[:, None]
    )
    if not torch.equal(eligible, expected_mask) or bool(eligible[:, :SCORE_START].any()):
        raise RuntimeError("VALIDATION eligibility semantics changed")
    for ordinal, count in enumerate(clipped.tolist()):
        if count < WIDTH and not bool((rows[ordinal, count:] == EOT).all()):
            raise RuntimeError("VALIDATION short-row padding changed")
    all_false = torch.nonzero(~eligible.any(1), as_tuple=False).flatten().tolist()
    if int(eligible.sum()) != ELIGIBLE_POSITIONS or int(eligible.any(1).sum()) != (
        SUPPORT_DOCUMENTS
    ) or len(all_false) != 1:
        raise RuntimeError("VALIDATION support census changed")
    return {
        "documents": DOCUMENTS,
        "eligible_positions": ELIGIBLE_POSITIONS,
        "support_documents": SUPPORT_DOCUMENTS,
        "all_false_ordinals": all_false,
        "tensor_hashes": hashes,
    }


def validate_claim(nonce: str, inode: tuple[int, int]) -> None:
    descriptor = os.open(LOCK, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(descriptor)
        payload = os.read(descriptor, 4096)
    finally:
        os.close(descriptor)
    if (stat.st_dev, stat.st_ino) != inode or json.loads(payload).get("nonce") != nonce:
        raise RuntimeError("VALIDATION projection claim changed")


def write_create_only_guarded(
    path: Path, data: bytes, *, before_link: Callable[[], None],
) -> None:
    """Durably publish once, with the terminal guard adjacent to the hard link."""
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}"
    )
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
        )
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        before_link()
        os.link(temporary, path)
        base.fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def final_guard(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], output_hash: str, manifest_hash: str,
) -> None:
    validate_claim(nonce, inode)
    for relative, expected in source_hashes.items():
        if base.file_sha256(ROOT / relative) != expected:
            raise RuntimeError("VALIDATION projection source changed during execution")
    current, _ = base.parent_snapshot()
    if current != parents or base.file_sha256(OUTPUT) != output_hash or base.file_sha256(
        MANIFEST
    ) != manifest_hash or RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("VALIDATION projection terminal snapshot changed")
    validate_claim(nonce, inode)


def partial_output_snapshot() -> dict[str, str | None]:
    return {
        name: base.file_sha256(path) if path.exists() else None
        for name, path in {"output": OUTPUT, "manifest": MANIFEST}.items()
    }


def failure_guard(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], partial_outputs: dict[str, str | None],
) -> None:
    validate_claim(nonce, inode)
    for relative, expected in source_hashes.items():
        if base.file_sha256(ROOT / relative) != expected:
            raise RuntimeError("VALIDATION projection source changed before failure")
    current, _ = base.parent_snapshot()
    if current != parents or partial_output_snapshot() != partial_outputs or (
        RECEIPT.exists() or FAILURE.exists()
    ):
        raise RuntimeError("VALIDATION projection failure snapshot changed")
    validate_claim(nonce, inode)


def project() -> dict[str, Any]:
    if any(path.exists() for path in (OUTPUT, MANIFEST, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("VALIDATION projection namespace already exists")
    commit, source_hashes = committed_source()
    parents, captured = base.parent_snapshot()
    nonce = secrets.token_hex(32)
    inode: tuple[int, int] | None = None
    try:
        base.write_create_only(LOCK, base.canonical_json_bytes({
            "nonce": nonce, "purpose": "validation_role_projection",
        }))
        stat = LOCK.stat(follow_symlinks=False)
        inode = (stat.st_dev, stat.st_ino)
        validate_claim(nonce, inode)
        combined = torch.load(
            io.BytesIO(captured["combined"]), map_location="cpu", weights_only=True,
        )
        if set(combined) != {"FIT_MEAN", "FIT_SELECTOR", "VALIDATION", "REPLICATION"}:
            raise RuntimeError("combined token role set changed")
        role = {
            name: value.detach().cpu().clone().contiguous()
            for name, value in combined["VALIDATION"].items()
        }
        del combined, captured
        summary = validate_role(role)
        base.publish_torch_create_only(OUTPUT, role)
        output_hash = base.file_sha256(OUTPUT)
        manifest = {
            "schema_version": 1,
            "experiment_id": "bilin18_mlp2_cmr_v1_validation_projection",
            "status": "validation_role_only_published_pending_receipt",
            "source_commit": commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "output_path": str(OUTPUT.relative_to(ROOT)),
            "output_sha256": output_hash,
            "summary": summary,
            "contains_roles": ["VALIDATION"],
            "combined_container_unavoidably_deserialized_without_model": True,
            "model_loaded": False,
            "candidate_constructed": False,
            "scientific_outcome_computed": False,
        }
        base.write_create_only(MANIFEST, base.canonical_json_bytes(manifest))
        manifest_hash = base.file_sha256(MANIFEST)
        replay = torch.load(OUTPUT, map_location="cpu", weights_only=True)
        if validate_role(replay) != summary or set(replay) != set(role):
            raise RuntimeError("VALIDATION projection semantic replay failed")
        receipt = {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "status": "validation_role_only_projection_complete_receipt_last",
            "source_commit": commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "output_sha256": output_hash,
            "manifest_sha256": manifest_hash,
            "summary": summary,
            "authorized_for_validation_model_forward_input": True,
            "authorized_for_replication": False,
            "projection_loaded_model": False,
        }

        def receipt_guard() -> None:
            final_guard(
                source_hashes, parents, nonce, inode, output_hash, manifest_hash,
            )

        write_create_only_guarded(
            RECEIPT, base.canonical_json_bytes(receipt), before_link=receipt_guard,
        )
        return receipt
    except BaseException as error:
        if inode is not None and not RECEIPT.exists() and not FAILURE.exists():
            partial_outputs = partial_output_snapshot()
            failure = {
                "schema_version": 1,
                "experiment_id": "bilin18_mlp2_cmr_v1_validation_projection",
                "status": "validation_projection_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "model_loaded": False,
                "candidate_constructed": False,
                "scientific_outcome_computed": False,
                "partial_outputs": partial_outputs,
            }

            def guarded_failure() -> None:
                failure_guard(
                    source_hashes, parents, nonce, inode, partial_outputs,
                )

            try:
                write_create_only_guarded(
                    FAILURE, base.canonical_json_bytes(failure),
                    before_link=guarded_failure,
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    print(json.dumps(project(), indent=2, sort_keys=True))
