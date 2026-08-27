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
from typing import Any, Iterable

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
SPECIFICATION = HERE / "MLP0_C512_MLP2_COMPENSATION_SPEC.md"
REFERENCE_ROWS = BQ / "bilin18_eval_tokens_large.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed_source(path: Path) -> None:
    relative = str(path.relative_to(ROOT))
    subprocess.check_call(
        ["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    blob = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != file_sha256(path):
        raise RuntimeError(f"row-freezing source is not byte-identical to HEAD: {relative}")


def discover_prior_registry_files() -> tuple[Path, ...]:
    """Return every prior JSON receipt/manifest/authority that can bind row roles."""
    paths = {BASE.CANONICAL_RECEIPT}
    for pattern in ("*receipt*.json", "*manifest*.json", "*authority*.json"):
        paths.update(BQ.glob(pattern))
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


def referenced_row_paths(payload: Any) -> set[Path]:
    """Find existing tensor paths whose filenames identify token/row artifacts."""
    paths: set[Path] = set()
    for value in walk_json(payload):
        if not isinstance(value, str) or not value.endswith((".pt", ".pth")):
            continue
        path = Path(value)
        name = path.name.lower()
        if not any(term in name for term in ("row", "fineweb", "eval_token", "oracle_corpus")):
            continue
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise RuntimeError(f"registered row-like tensor is missing: {path}")
        paths.add(path.resolve())
    return paths


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


def load_registry_exclusions(
    registry_files: tuple[Path, ...],
) -> tuple[tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]], dict[str, str], dict[str, str]]:
    documents: set[str] = set()
    indices: set[int] = set()
    full_rows: set[tuple[int, ...]] = set()
    prefixes: set[tuple[int, ...]] = set()
    registry_hashes: dict[str, str] = {}
    tensor_paths: set[Path] = {REFERENCE_ROWS.resolve()}

    for path in registry_files:
        registry_hashes[str(path.resolve())] = file_sha256(path)
        payload = json.loads(path.read_text())
        tensor_paths.update(referenced_row_paths(payload))
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
    for path in sorted(tensor_paths):
        tensor_hashes[str(path)] = file_sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tensors = list(long_row_tensors(payload))
        if not tensors:
            raise RuntimeError(f"registered row-like artifact contains no long rank-2 rows: {path}")
        for tensor in tensors:
            for row in tensor:
                values = tuple(int(item) for item in row.tolist())
                full_rows.add(values)
                prefixes.add(values[:32])

    return (documents, indices, full_rows, prefixes), registry_hashes, tensor_hashes


def freeze() -> dict[str, Any]:
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
        require_committed_source(path)

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
    if source.stat().st_size != BASE.local.PINNED_SIZE or file_sha256(source) != BASE.local.PINNED_SHA256:
        raise RuntimeError("pinned FineWeb parquet changed during row selection")

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
        os.replace(staging, CACHE)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

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
        "source_receipt_sha256": file_sha256(BASE.CANONICAL_RECEIPT),
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
        "implementation_hashes": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in source_closure
        },
    }
    write_json_atomic(receipt, RECEIPT)
    return receipt


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
