#!/usr/bin/env python3
"""Freeze fresh source-document rows for the C512/MLP2 physical factorial.

This program is outcome blind and performs no model forward.  It extends the tested
native-Down row harvester with a registry-wide exclusion census: every prior receipt,
manifest, and authority in the experiment directory is hashed, all embedded document
identities are excluded, and every referenced long rank-2 row tensor contributes full
rows and 32-token prefixes to the exclusion set.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
BASE_PATH = HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py"
BASE_SPEC = importlib.util.spec_from_file_location("native_down_rows_v1", BASE_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE)

START_DOCUMENT_INDEX = 47_000
N_SOURCE_DOCUMENTS = 384
WAVE_DOCUMENTS = 192
MAX_CHUNKS_PER_DOCUMENT = 3
TOKEN_LENGTH = 513
CACHE = BQ / ".rowcache_mlp0_c512_mlp2_compensation_v1"
RECEIPT = BQ / "mlp0_c512_mlp2_compensation_v1_rows_receipt.json"
LOCK = BQ / ".mlp0_c512_mlp2_compensation_v1_rows.lock"
SPECIFICATION = HERE / "MLP0_C512_MLP2_COMPENSATION_SPEC.md"
REFERENCE_ROWS = BQ / "bilin18_eval_tokens_large.pt"

GENERIC_FILE_DIGEST_KEYS = ("file_sha256", "cache_file_sha256")
FILE_DIGEST_KEYS = GENERIC_FILE_DIGEST_KEYS + (
    "corpus_sha256", "final_cache_sha256", "sha256",
)
FULL_TENSOR_DIGEST_KEYS = (
    "tensor_full_raw_sha256", "tensor_raw_sha256", "raw_tensor_sha256",
)
PREFIX257_DIGEST_KEYS = ("tensor_prefix257_raw_sha256",)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()


def write_json_create_only(payload: dict[str, Any], path: Path) -> None:
    """Atomically publish JSON without ever replacing an existing authority."""
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            descriptor = None
            handle.write(json.dumps(payload, indent=2) + "\n")
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


def acquire_lock(path: Path = LOCK) -> int:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"row-freezing namespace is already locked: {path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
    except BaseException:
        release_lock(descriptor, path)
        raise
    return descriptor


def release_lock(descriptor: int, path: Path = LOCK) -> None:
    try:
        # Do not unlink a replacement lock if an external actor removed ours.
        if path.exists() and os.stat(path).st_ino == os.fstat(descriptor).st_ino:
            path.unlink()
    finally:
        os.close(descriptor)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed_source(path: Path, source_commit: str) -> None:
    relative = str(path.relative_to(ROOT))
    subprocess.check_call(
        ["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    blob = subprocess.check_output(["git", "show", f"{source_commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != file_sha256(path):
        raise RuntimeError(
            f"row-freezing source is not byte-identical to {source_commit}: {relative}"
        )


def discover_prior_registry_files() -> tuple[Path, ...]:
    """Return every prior JSON receipt/manifest/authority that can bind row roles."""
    paths = {BASE.CANONICAL_RECEIPT}
    for pattern in ("*receipt*.json", "*manifest*.json", "*authority*.json"):
        paths.update(BQ.rglob(pattern))
    paths.discard(RECEIPT)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"registered prior identity file is missing: {missing}")
    return tuple(sorted(paths))


def walk_json(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def row_like_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.endswith((".pt", ".pth")):
        return None
    path = Path(value)
    name = path.name.lower()
    parent_is_rowcache = any(part.startswith(".rowcache") for part in path.parts)
    if not parent_is_rowcache and not any(
        term in name for term in (
            "row", "fineweb", "eval_token", "oracle_corpus", "source_document",
        )
    ):
        return None
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise RuntimeError(f"registered row-like tensor is missing: {path}")
    return path.resolve()


def referenced_row_specs(payload: Any) -> dict[Path, list[dict[str, str]]]:
    """Find row tensors and digest declarations colocated with each path."""
    specifications: dict[Path, list[dict[str, str]]] = {}

    def add_schema_pair(value: dict[str, Any], path_key: str, digest_key: str) -> None:
        if path_key not in value and digest_key not in value:
            return
        if path_key not in value or digest_key not in value:
            raise RuntimeError(f"incomplete declared row-file pair {path_key}/{digest_key}")
        path = row_like_path(value[path_key])
        if path is None:
            raise RuntimeError(f"declared {path_key} is not a row-like tensor path")
        digest = value[digest_key]
        if not is_sha256(digest):
            raise RuntimeError(f"invalid declared SHA-256 in field {digest_key}")
        specifications.setdefault(path, []).append({digest_key: digest})

    def visit(value: Any, role: str | None = None) -> None:
        if isinstance(value, dict):
            digest_keys = (
                GENERIC_FILE_DIGEST_KEYS + FULL_TENSOR_DIGEST_KEYS
                + PREFIX257_DIGEST_KEYS
            )
            declarations: dict[str, str] = {}
            for key, item in value.items():
                if key not in digest_keys:
                    continue
                if not is_sha256(item):
                    raise RuntimeError(f"invalid declared SHA-256 in field {key}")
                declarations[key] = item
            for item in value.values():
                path = row_like_path(item)
                if path is not None:
                    specifications.setdefault(path, []).append(declarations)
            add_schema_pair(value, "corpus_path", "corpus_sha256")
            add_schema_pair(value, "final_cache_path", "final_cache_sha256")
            if role == "corpus":
                add_schema_pair(value, "path", "sha256")
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    visit(item, key)
        elif isinstance(value, list):
            for item in value:
                visit(item, role)

    visit(payload)
    return specifications


def long_row_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.long and value.ndim == 2 and value.shape[1] >= 32:
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from long_row_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from long_row_tensors(child)


def verify_declared_tensor_digests(
    path: Path,
    tensors: list[torch.Tensor],
    specifications: Iterable[Mapping[str, str]],
    observed_file_sha256: str,
) -> None:
    declarations: dict[str, set[str]] = {}
    for specification in specifications:
        for key, expected in specification.items():
            declarations.setdefault(key, set()).add(expected.lower())
    for key in FILE_DIGEST_KEYS:
        for expected in declarations.get(key, set()):
            if observed_file_sha256 != expected:
                raise RuntimeError(f"declared {key} mismatch for {path}")
    tensor_declarations = set(FULL_TENSOR_DIGEST_KEYS + PREFIX257_DIGEST_KEYS)
    if any(key in declarations for key in tensor_declarations) and len(tensors) != 1:
        raise RuntimeError(
            f"cannot bind declared tensor digest to {len(tensors)} row tensors in {path}"
        )
    if not tensors:
        return
    tensor = tensors[0]
    full_digest = tensor_sha256(tensor)
    prefix_digest = tensor_sha256(tensor[:, :257])
    for key in FULL_TENSOR_DIGEST_KEYS:
        for expected in declarations.get(key, set()):
            if full_digest != expected:
                raise RuntimeError(f"declared {key} mismatch for {path}")
    for key in PREFIX257_DIGEST_KEYS:
        for expected in declarations.get(key, set()):
            if prefix_digest != expected:
                raise RuntimeError(f"declared {key} mismatch for {path}")


def load_verified_row_tensor(
    path: Path, specifications: Iterable[Mapping[str, str]],
) -> tuple[list[torch.Tensor], str]:
    before = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    after = file_sha256(path)
    if after != before:
        raise RuntimeError(f"registered row-like tensor changed while loading: {path}")
    tensors = list(long_row_tensors(payload))
    if not tensors:
        raise RuntimeError(f"registered row-like artifact contains no long rank-2 rows: {path}")
    verify_declared_tensor_digests(path, tensors, specifications, before)
    return tensors, before


def load_registry_exclusions(
    registry_files: tuple[Path, ...],
) -> tuple[tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]], dict[str, str], dict[str, str]]:
    documents: set[str] = set()
    indices: set[int] = set()
    full_rows: set[tuple[int, ...]] = set()
    prefixes: set[tuple[int, ...]] = set()
    registry_hashes: dict[str, str] = {}
    tensor_specifications: dict[Path, list[dict[str, str]]] = {
        REFERENCE_ROWS.resolve(): []
    }

    for path in registry_files:
        registry_hashes[str(path.resolve())] = file_sha256(path)
        payload = json.loads(path.read_text())
        for tensor_path, specifications in referenced_row_specs(payload).items():
            tensor_specifications.setdefault(tensor_path, []).extend(specifications)
        for value in walk_json(payload):
            if not isinstance(value, dict):
                continue
            document = value.get("document_id")
            index = value.get("dataset_document_index")
            if isinstance(document, str) and document:
                documents.add(document)
            if isinstance(index, int):
                indices.add(index)

    tensor_hashes: dict[str, str] = {}
    for path in sorted(tensor_specifications):
        tensors, digest = load_verified_row_tensor(path, tensor_specifications[path])
        tensor_hashes[str(path)] = digest
        for tensor in tensors:
            for row in tensor:
                values = tuple(int(item) for item in row.tolist())
                full_rows.add(values)
                prefixes.add(values[:32])

    return (documents, indices, full_rows, prefixes), registry_hashes, tensor_hashes


def verify_frozen_snapshot(
    *,
    source_commit: str,
    source_closure: tuple[Path, ...],
    implementation_hashes: Mapping[str, str],
    registry_files: tuple[Path, ...],
    registry_hashes: Mapping[str, str],
    prior_tensor_hashes: Mapping[str, str],
    prior: tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    source: Path,
) -> None:
    if git("rev-parse", "HEAD") != source_commit or git("rev-parse", "origin/main") != source_commit:
        raise RuntimeError("git identity changed during row selection")
    for path in source_closure:
        relative = str(path.relative_to(ROOT))
        if file_sha256(path) != implementation_hashes.get(relative):
            raise RuntimeError(f"implementation source changed during row selection: {relative}")
        require_committed_source(path, source_commit)
    current_registry = discover_prior_registry_files()
    if current_registry != registry_files:
        raise RuntimeError("prior registry membership changed during row selection")
    current_prior, current_registry_hashes, current_tensor_hashes = load_registry_exclusions(
        current_registry
    )
    if current_registry_hashes != dict(registry_hashes):
        raise RuntimeError("prior registry contents changed during row selection")
    if current_tensor_hashes != dict(prior_tensor_hashes):
        raise RuntimeError("prior row tensor contents changed during row selection")
    if current_prior != prior:
        raise RuntimeError("prior exclusion identity changed during row selection")
    if source.stat().st_size != BASE.local.PINNED_SIZE or file_sha256(source) != BASE.local.PINNED_SHA256:
        raise RuntimeError("pinned FineWeb parquet changed during row selection")


def install_cache_create_only(staging: Path, cache: Path) -> Path:
    """Claim the final cache namespace, then install the staged tensor inside it."""
    cache.mkdir(parents=False, exist_ok=False)
    staged_files = tuple(staging.iterdir())
    if len(staged_files) != 1 or not staged_files[0].is_file():
        raise RuntimeError("row-cache staging must contain exactly one file")
    final_path = cache / staged_files[0].name
    os.replace(staged_files[0], final_path)
    return final_path


def freeze_locked() -> dict[str, Any]:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite C512/MLP2 row authority")
    source_commit = git("rev-parse", "HEAD")
    if source_commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    source_closure = (
        Path(__file__), SPECIFICATION, BASE_PATH,
        HERE / "local_fineweb_harvest.py",
    )
    for path in source_closure:
        require_committed_source(path, source_commit)
    implementation_hashes = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in source_closure
    }

    canonical, source = BASE.validate_ordered_source()
    registry_files = discover_prior_registry_files()
    prior, registry_hashes, prior_tensor_hashes = load_registry_exclusions(registry_files)

    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    rows, records = BASE.harvest_balanced_documents(
        BASE.local.parquet_texts([source]),
        encoding.encode_ordinary,
        set(prior[3]),
        set(prior[0]),
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=N_SOURCE_DOCUMENTS,
        wave_documents=WAVE_DOCUMENTS,
        max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
        token_length=TOKEN_LENGTH,
    )
    summary = BASE.summarize_records(records)
    disjointness = BASE.validate_eval_disjointness(rows, records, prior)

    staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        staged = staging / "eval_384_source_documents.pt"
        torch.save(rows, staged)
        final_path = CACHE / staged.name
        entry = {
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
            "tensor_raw_sha256": tensor_sha256(rows),
            "file_sha256": file_sha256(staged),
            "cache_path": str(final_path.resolve()),
        }
        verify_frozen_snapshot(
            source_commit=source_commit,
            source_closure=source_closure,
            implementation_hashes=implementation_hashes,
            registry_files=registry_files,
            registry_hashes=registry_hashes,
            prior_tensor_hashes=prior_tensor_hashes,
            prior=prior,
            source=source,
        )
        installed = install_cache_create_only(staging, CACHE)
        if installed != final_path or file_sha256(installed) != entry["file_sha256"]:
            raise RuntimeError("installed row cache differs from staged artifact")
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    # Close the small install-to-publication window as well. A failure here leaves
    # cache bytes but never creates an authority receipt.
    verify_frozen_snapshot(
        source_commit=source_commit,
        source_closure=source_closure,
        implementation_hashes=implementation_hashes,
        registry_files=registry_files,
        registry_hashes=registry_hashes,
        prior_tensor_hashes=prior_tensor_hashes,
        prior=prior,
        source=source,
    )

    receipt = {
        "schema_version": 1,
        "receipt_kind": "mlp0_c512_mlp2_compensation_v1_rows",
        "status": "frozen_before_any_c512_mlp2_model_forward",
        "authority": "pinned_local_ordered_manifest_source_document_balanced_registry_wide_exclusion",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
        "source_commit": source_commit,
        "selection": {
            "start_dataset_document_index": START_DOCUMENT_INDEX,
            "n_source_documents": N_SOURCE_DOCUMENTS,
            "wave_source_documents": WAVE_DOCUMENTS,
            "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT,
            "token_length": TOKEN_LENGTH,
        },
        "sample_summary": summary,
        "entries": {"eval": entry},
        "document_provenance": {"schema_version": 1, "sets": {"eval": records}},
        "disjointness_gates": disjointness,
        "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
        "source_receipt_path": str(BASE.CANONICAL_RECEIPT.resolve()),
        "source_receipt_sha256": registry_hashes[str(BASE.CANONICAL_RECEIPT.resolve())],
        "prior_registry_files": registry_hashes,
        "prior_row_tensors": prior_tensor_hashes,
        "exclusion_counts": {
            "source_documents": len(prior[0]),
            "dataset_document_indices": len(prior[1]),
            "full_rows": len(prior[2]),
            "prefix32": len(prior[3]),
            "registry_files": len(registry_hashes),
            "row_tensor_files": len(prior_tensor_hashes),
        },
        "implementation_hashes": implementation_hashes,
    }
    # Assemble the complete receipt first, then make one final closure check before
    # its create-only publication. No live filesystem identity is read into the
    # payload after this check.
    verify_frozen_snapshot(
        source_commit=source_commit,
        source_closure=source_closure,
        implementation_hashes=implementation_hashes,
        registry_files=registry_files,
        registry_hashes=registry_hashes,
        prior_tensor_hashes=prior_tensor_hashes,
        prior=prior,
        source=source,
    )
    write_json_create_only(receipt, RECEIPT)
    return receipt


def freeze() -> dict[str, Any]:
    descriptor = acquire_lock()
    try:
        return freeze_locked()
    finally:
        release_lock(descriptor)


def load_frozen_rows(path: Path = RECEIPT) -> tuple[dict[str, Any], torch.Tensor]:
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "frozen_before_any_c512_mlp2_model_forward"
            or receipt.get("authorized_for_scored_experiments") is not True
            or receipt.get("authorized_for_training") is not False
            or not all(receipt.get("disjointness_gates", {}).values())):
        raise RuntimeError("C512/MLP2 row receipt is not authoritative")
    entry = receipt.get("entries", {}).get("eval", {})
    row_path = Path(entry.get("cache_path", ""))
    rows = torch.load(row_path, map_location="cpu", weights_only=True)
    if (not isinstance(rows, torch.Tensor) or rows.dtype != torch.long
            or rows.ndim != 2 or rows.shape[1] != TOKEN_LENGTH
            or tensor_sha256(rows) != entry.get("tensor_raw_sha256")
            or file_sha256(row_path) != entry.get("file_sha256")):
        raise RuntimeError("frozen C512/MLP2 rows changed")
    BASE.summarize_records(receipt["document_provenance"]["sets"]["eval"])
    return receipt, rows


if __name__ == "__main__":
    frozen = freeze()
    print(json.dumps({
        "receipt": str(RECEIPT),
        "summary": frozen["sample_summary"],
        "disjointness": frozen["disjointness_gates"],
        "exclusion_counts": frozen["exclusion_counts"],
        "tensor_sha256": frozen["entries"]["eval"]["tensor_raw_sha256"],
    }, indent=2))
