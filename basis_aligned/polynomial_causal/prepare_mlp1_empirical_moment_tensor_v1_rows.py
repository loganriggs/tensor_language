#!/usr/bin/env python3
"""Create-only outcome-blind document-role freezer for MLP1 empirical moments.

Running this program reads only source-control identity, JSON registry identities,
and the pinned parquet's bytes/footer.  It does not read parquet columns, tokenize,
load tensors/checkpoints/models, or execute a model.  It publishes a nonauthorizing
role manifest followed by a receipt that authorizes only those document roles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Iterable, Mapping

import mlp1_empirical_moment_tensor_v1_row_manifest as role_manifest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / "MLP1_EMPIRICAL_MOMENT_TENSOR_ROWS_V1_PROTOCOL.json"
PREREGISTRATION = HERE / "MLP1_EMPIRICAL_MOMENT_TENSOR_DISCRIMINATOR_PREREGISTRATION.md"
ADDENDUM = HERE / "MLP1_EMPIRICAL_MOMENT_TENSOR_EXECUTION_ADDENDUM.md"
ORDERED_FINEWEB_RECEIPT = HERE.parent / "bilinear_quotient" / ".rowcache" / "fineweb_oracle_v2_receipt.json"
MANIFEST = HERE / "mlp1_empirical_moment_tensor_v1_row_roles_manifest.json"
RECEIPT = HERE / "mlp1_empirical_moment_tensor_v1_row_roles_receipt.json"
LOCK = HERE / ".mlp1_empirical_moment_tensor_v1_row_roles.lock"
PARQUET = Path("/workspace/fineweb_pinned/data/CC-MAIN-2013-20/000_00000.parquet")

SOURCE_CLOSURE = (
    PROTOCOL,
    PREREGISTRATION,
    ADDENDUM,
    Path(role_manifest.__file__).resolve(),
    Path(__file__).resolve(),
    HERE / "test_mlp1_empirical_moment_tensor_v1_row_manifest.py",
    HERE / "test_prepare_mlp1_empirical_moment_tensor_v1_rows.py",
)

REGISTRY_TERMS = ("receipt", "manifest", "authority")
EXCLUDED_OUTPUTS = frozenset((MANIFEST.resolve(), RECEIPT.resolve()))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def require_committed_source(path: Path, commit: str) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"source closure escapes repository: {resolved}") from exc
    subprocess.check_call(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != file_sha256(resolved):
        raise RuntimeError(f"source is not byte-identical to {commit}: {relative}")


def discover_registry_files(root: Path = ROOT) -> tuple[Path, ...]:
    """Discover every worktree JSON whose filename declares registry semantics."""
    discovered: set[Path] = {ORDERED_FINEWEB_RECEIPT.resolve()}
    for path in root.rglob("*.json"):
        resolved = path.resolve()
        if resolved in EXCLUDED_OUTPUTS:
            continue
        if not resolved.is_relative_to(root.resolve()):
            raise RuntimeError(f"registry symlink escapes repository: {path}")
        lower = path.name.lower()
        if any(term in lower for term in REGISTRY_TERMS):
            if not path.is_file():
                raise RuntimeError(f"registry candidate is not a regular file: {path}")
            discovered.add(resolved)
    return tuple(sorted(discovered))


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk(child)


def _literal_index(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context} is not a literal nonnegative integer")
    return value


def _index_from_document_id(value: Any, protocol: Mapping[str, Any]) -> int | None:
    if not isinstance(value, str):
        raise RuntimeError("document_id registry field is not a string")
    source = protocol["fineweb_source"]
    prefix = f"{source['revision']}:{source['relative_path']}:"
    if not value.startswith(prefix):
        return None
    suffix = value[len(prefix):]
    if not suffix or (suffix != "0" and suffix.startswith("0")) or not suffix.isascii() \
            or not suffix.isdecimal():
        raise RuntimeError("canonical FineWeb document_id has malformed index")
    return int(suffix)


def load_registry_exclusions(
    files: tuple[Path, ...], protocol: Mapping[str, Any],
) -> tuple[frozenset[int], dict[str, str]]:
    indices: set[int] = set()
    hashes: dict[str, str] = {}
    for path in files:
        before = file_sha256(path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot parse registry JSON: {path}") from exc
        after = file_sha256(path)
        if after != before:
            raise RuntimeError(f"registry changed while reading: {path}")
        hashes[str(path.resolve())] = before
        for key, value in _walk(payload):
            if key in ("dataset_document_index", "source_document_index"):
                indices.add(_literal_index(value, f"{path}:{key}"))
            elif key is not None and key.endswith("document_indices"):
                if not isinstance(value, list):
                    raise RuntimeError(f"{path}:{key} is not an index list")
                indices.update(_literal_index(item, f"{path}:{key}") for item in value)
            elif key == "document_id":
                parsed = _index_from_document_id(value, protocol)
                if parsed is not None:
                    indices.add(parsed)
    row_count = protocol["fineweb_source"]["parquet_rows"]
    outside = sorted(index for index in indices if index >= row_count)
    if outside:
        raise RuntimeError(f"registry FineWeb index exceeds pinned parquet: {outside[:3]}")
    return frozenset(indices), hashes


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 \
            or payload.get("experiment_id") != role_manifest.EXPERIMENT_ID:
        raise RuntimeError("row-role protocol identity changed")
    for parent in payload.get("parents", {}).values():
        parent_path = ROOT / parent["path"]
        if file_sha256(parent_path) != parent["sha256"]:
            raise RuntimeError(f"row-role protocol parent changed: {parent_path}")
    selection = payload.get("selection", {})
    if selection != {
        "ordering_domain": "every_integer_document_index_in_[0,parquet_rows)_minus_recursive_registry_exclusions",
        "ordering_key": "(sha256(utf8('bilin18_mlp1_empirical_moment_v1\\0'+decimal_document_index)).digest(),document_index)",
        "role_order": list(role_manifest.ROLES),
        "documents_per_role": role_manifest.DOCUMENTS_PER_ROLE,
        "window_tokens": role_manifest.WINDOW_TOKENS,
        "eligible_position_start": role_manifest.POSITION_START,
        "eligible_position_stop": role_manifest.POSITION_STOP,
        "rows_per_role": role_manifest.ROWS_PER_ROLE,
        "final_window_position_stop": role_manifest.ROLE_FINAL_POSITION_STOP,
        "fit_prefix_rows": {"FIT100": 100_000, "FIT200": 200_000, "FIT400": 400_000},
    }:
        raise RuntimeError("row-role protocol selection changed")
    return payload


def validate_parquet_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    source = protocol["fineweb_source"]
    if PARQUET.resolve() != Path(source["local_path"]).resolve():
        raise RuntimeError("pinned parquet path changed")
    stat = PARQUET.stat()
    if stat.st_size != source["size"] or file_sha256(PARQUET) != source["sha256"]:
        raise RuntimeError("pinned parquet bytes changed")
    import pyarrow.parquet as parquet
    metadata = parquet.ParquetFile(PARQUET).metadata
    if metadata.num_rows != source["parquet_rows"]:
        raise RuntimeError("pinned parquet row count changed")
    return {
        "path": str(PARQUET.resolve()),
        "size": stat.st_size,
        "sha256": source["sha256"],
        "parquet_rows": metadata.num_rows,
        "parquet_row_groups": metadata.num_row_groups,
    }


@dataclass(frozen=True)
class RunClaim:
    path: Path
    descriptor: int
    inode: int
    nonce: str


def acquire_lock(path: Path = LOCK) -> RunClaim:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    nonce = secrets.token_hex(32)
    payload = json.dumps({"pid": os.getpid(), "nonce": nonce}, sort_keys=True).encode()
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        inode = os.fstat(descriptor).st_ino
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return RunClaim(path, descriptor, inode, nonce)


def require_run_claim(claim: RunClaim) -> None:
    if not claim.path.is_file() or os.stat(claim.path).st_ino != claim.inode \
            or os.fstat(claim.descriptor).st_ino != claim.inode:
        raise RuntimeError("row-role lock ownership changed")
    payload = json.loads(claim.path.read_text())
    if payload != {"nonce": claim.nonce, "pid": os.getpid()}:
        raise RuntimeError("row-role lock claim changed")


def release_lock(claim: RunClaim) -> None:
    try:
        if claim.path.exists() and os.stat(claim.path).st_ino == claim.inode:
            claim.path.unlink()
    finally:
        os.close(claim.descriptor)


def write_json_create_only(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def source_identity() -> tuple[str, dict[str, str]]:
    commit = git("rev-parse", "HEAD")
    if commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    hashes: dict[str, str] = {}
    for path in SOURCE_CLOSURE:
        require_committed_source(path, commit)
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return commit, hashes


def verify_snapshot(
    *, claim: RunClaim, commit: str, source_hashes: Mapping[str, str],
    protocol: Mapping[str, Any], registry_files: tuple[Path, ...],
    registry_hashes: Mapping[str, str], exclusions: frozenset[int],
    parquet_identity: Mapping[str, Any], require_manifest: bool,
) -> None:
    require_run_claim(claim)
    if git("rev-parse", "HEAD") != commit or git("rev-parse", "origin/main") != commit:
        raise RuntimeError("source commit changed during role freeze")
    for path in SOURCE_CLOSURE:
        relative = str(path.relative_to(ROOT))
        if file_sha256(path) != source_hashes.get(relative):
            raise RuntimeError(f"source closure changed during role freeze: {relative}")
        require_committed_source(path, commit)
    current_files = discover_registry_files()
    if current_files != registry_files:
        raise RuntimeError("recursive registry membership changed during role freeze")
    current_exclusions, current_hashes = load_registry_exclusions(current_files, protocol)
    if current_hashes != dict(registry_hashes) or current_exclusions != exclusions:
        raise RuntimeError("recursive registry snapshot changed during role freeze")
    if validate_parquet_identity(protocol) != dict(parquet_identity):
        raise RuntimeError("pinned parquet identity changed during role freeze")
    if require_manifest:
        if not MANIFEST.is_file() or RECEIPT.exists():
            raise RuntimeError("row-role publication namespace changed")
    elif MANIFEST.exists() or RECEIPT.exists():
        raise RuntimeError("row-role publication namespace is not empty")


def freeze_locked(claim: RunClaim) -> dict[str, Any]:
    require_run_claim(claim)
    if MANIFEST.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite row-role namespace")
    protocol = load_protocol()
    commit, source_hashes = source_identity()
    parquet_identity = validate_parquet_identity(protocol)
    registry_files = discover_registry_files()
    exclusions, registry_hashes = load_registry_exclusions(registry_files, protocol)
    census = {
        "discovery_rule": "recursive_repo_json_filename_contains_receipt_manifest_or_authority",
        "registry_file_count": len(registry_files),
        "registry_files": registry_hashes,
        "excluded_document_index_count": len(exclusions),
        "excluded_document_indices_sha256": role_manifest.sha256_json(sorted(exclusions)),
    }
    manifest = role_manifest.build_role_manifest(
        parquet_rows=parquet_identity["parquet_rows"],
        excluded_indices=exclusions,
        registry_census=census,
    )
    manifest.update({
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_commit": commit,
        "source_hashes": source_hashes,
        "parquet_identity": parquet_identity,
    })
    verify_snapshot(
        claim=claim, commit=commit, source_hashes=source_hashes, protocol=protocol,
        registry_files=registry_files, registry_hashes=registry_hashes,
        exclusions=exclusions, parquet_identity=parquet_identity,
        require_manifest=False,
    )
    write_json_create_only(manifest, MANIFEST)
    manifest_hash = file_sha256(MANIFEST)
    receipt = {
        "schema_version": 1,
        "receipt_kind": "mlp1_empirical_moment_tensor_v1_row_roles",
        "status": "document_roles_frozen_before_tokenization_or_model_access",
        "authority": "document_role_identity_only",
        "authorized_for_document_role_identity": True,
        "authorized_for_tokenization": False,
        "authorized_for_activation_capture": False,
        "authorized_for_model_forward": False,
        "authorized_for_fit_or_validation": False,
        "authorized_for_scientific_outcomes": False,
        "manifest_path": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "manifest_bytes": MANIFEST.stat().st_size,
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_commit": commit,
        "source_hashes": source_hashes,
        "parquet_identity": parquet_identity,
        "registry_snapshot": census,
        "downstream_rule": (
            "tokenization requires a distinct source-closed receipt binding ordered "
            "(source_document_index,position_index,token_id) triples; this receipt "
            "cannot authorize checkpoint/model/activation/outcome access"
        ),
    }
    verify_snapshot(
        claim=claim, commit=commit, source_hashes=source_hashes, protocol=protocol,
        registry_files=registry_files, registry_hashes=registry_hashes,
        exclusions=exclusions, parquet_identity=parquet_identity,
        require_manifest=True,
    )
    if file_sha256(MANIFEST) != manifest_hash or MANIFEST.stat().st_size != receipt["manifest_bytes"]:
        raise RuntimeError("role manifest changed before receipt publication")
    require_run_claim(claim)
    write_json_create_only(receipt, RECEIPT)
    return receipt


def freeze() -> dict[str, Any]:
    claim = acquire_lock()
    try:
        return freeze_locked(claim)
    finally:
        release_lock(claim)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
