#!/usr/bin/env python3
"""Freeze registry-wide fresh FineWeb rows for the MLP1 global-gate assay.

This is outcome blind and performs no model forward.  Every prior receipt, manifest,
authority, referenced long token tensor, source document, dataset index, full row, and
32-token prefix in the experiment registry is excluded before selecting 32 new source
documents beginning at dataset document index 50,000.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

import prepare_mlp0_c512_mlp2_compensation_v1_rows as registry


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
START_DOCUMENT_INDEX = 50_000
N_SOURCE_DOCUMENTS = 32
WAVE_DOCUMENTS = 16
MAX_CHUNKS_PER_DOCUMENT = 1
TOKEN_LENGTH = 513
CACHE = BQ / ".rowcache_mlp1_global_gate_v1"
RECEIPT = BQ / "mlp1_global_gate_v1_rows_receipt.json"
LOCK = BQ / ".mlp1_global_gate_v1_rows.lock"
SPECIFICATION = HERE / "MLP1_GLOBAL_GATE_RESPONSE_PREREGISTRATION.md"


def discover_prior_registry_files() -> tuple[Path, ...]:
    paths = {registry.BASE.CANONICAL_RECEIPT}
    for pattern in ("*receipt*.json", "*manifest*.json", "*authority*.json"):
        paths.update(BQ.rglob(pattern))
    paths.discard(RECEIPT)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"registered prior identity file is missing: {missing}")
    return tuple(sorted(paths))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_document: dict[str, int] = {}
    by_wave = {"A": set(), "B": set()}
    chunks_by_wave = {"A": 0, "B": 0}
    for record in records:
        document, wave = record.get("document_id"), record.get("wave")
        if not isinstance(document, str) or not document or wave not in by_wave:
            raise RuntimeError("MLP1 row record identity or wave is malformed")
        by_document[document] = by_document.get(document, 0) + 1
        by_wave[wave].add(document)
        chunks_by_wave[wave] += 1
    counts = list(by_document.values())
    summary = {
        "n_source_documents": len(by_document),
        "n_chunks": len(records),
        "raw_prediction_positions": len(records) * (TOKEN_LENGTH - 1),
        "min_chunks_per_document": min(counts) if counts else 0,
        "max_chunks_per_document": max(counts) if counts else 0,
        "chunks_per_document_histogram": {
            str(value): counts.count(value) for value in sorted(set(counts))
        },
        "waves": {
            wave: {
                "n_source_documents": len(by_wave[wave]),
                "n_chunks": chunks_by_wave[wave],
                "raw_prediction_positions": chunks_by_wave[wave] * (TOKEN_LENGTH - 1),
            }
            for wave in ("A", "B")
        },
    }
    if (
        summary["n_source_documents"] != N_SOURCE_DOCUMENTS
        or summary["n_chunks"] != N_SOURCE_DOCUMENTS
        or summary["min_chunks_per_document"] != 1
        or summary["max_chunks_per_document"] != MAX_CHUNKS_PER_DOCUMENT
        or any(summary["waves"][wave]["n_source_documents"] != WAVE_DOCUMENTS
               for wave in ("A", "B"))
    ):
        raise RuntimeError(f"MLP1 balanced-document receipt invariant failed: {summary}")
    return summary


def verify_frozen_snapshot(
    *, source_commit: str, source_closure: tuple[Path, ...],
    implementation_hashes: Mapping[str, str], registry_files: tuple[Path, ...],
    registry_hashes: Mapping[str, str], prior_tensor_hashes: Mapping[str, str],
    prior: tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    source: Path,
) -> None:
    if registry.git("rev-parse", "HEAD") != source_commit or registry.git(
        "rev-parse", "origin/main"
    ) != source_commit:
        raise RuntimeError("git identity changed during MLP1 row selection")
    for path in source_closure:
        relative = str(path.relative_to(ROOT))
        if registry.file_sha256(path) != implementation_hashes.get(relative):
            raise RuntimeError(f"MLP1 row source changed during selection: {relative}")
        registry.require_committed_source(path, source_commit)
    current_files = discover_prior_registry_files()
    if current_files != registry_files:
        raise RuntimeError("prior registry membership changed during MLP1 row selection")
    current_prior, current_hashes, current_tensors = registry.load_registry_exclusions(
        current_files
    )
    if current_hashes != dict(registry_hashes):
        raise RuntimeError("prior registry contents changed during MLP1 row selection")
    if current_tensors != dict(prior_tensor_hashes):
        raise RuntimeError("prior row tensors changed during MLP1 row selection")
    if current_prior != prior:
        raise RuntimeError("prior row exclusions changed during MLP1 row selection")
    if source.stat().st_size != registry.BASE.local.PINNED_SIZE or registry.file_sha256(
        source
    ) != registry.BASE.local.PINNED_SHA256:
        raise RuntimeError("pinned FineWeb parquet changed during MLP1 row selection")


def freeze_locked() -> dict[str, Any]:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite MLP1 global-gate row authority")
    source_commit = registry.git("rev-parse", "HEAD")
    if source_commit != registry.git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    source_closure = (
        Path(__file__), SPECIFICATION,
        HERE / "prepare_mlp0_c512_mlp2_compensation_v1_rows.py",
        HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py",
        HERE / "local_fineweb_harvest.py",
    )
    for path in source_closure:
        registry.require_committed_source(path, source_commit)
    implementation_hashes = {
        str(path.relative_to(ROOT)): registry.file_sha256(path) for path in source_closure
    }

    canonical, source = registry.BASE.validate_ordered_source()
    registry_files = discover_prior_registry_files()
    prior, registry_hashes, prior_tensor_hashes = registry.load_registry_exclusions(
        registry_files
    )
    if prior[1] and max(prior[1]) >= START_DOCUMENT_INDEX:
        raise RuntimeError(
            "registered dataset-document indices reach the prospective MLP1 start"
        )

    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    rows, records = registry.BASE.harvest_balanced_documents(
        registry.BASE.local.parquet_texts([source]), encoding.encode_ordinary,
        set(prior[3]), set(prior[0]),
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=N_SOURCE_DOCUMENTS,
        wave_documents=WAVE_DOCUMENTS,
        max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
        token_length=TOKEN_LENGTH,
    )
    summary = summarize_records(records)
    disjointness = registry.BASE.validate_eval_disjointness(rows, records, prior)
    if tuple(rows.shape) != (N_SOURCE_DOCUMENTS, TOKEN_LENGTH) or len(records) != (
        N_SOURCE_DOCUMENTS
    ) or len({record["document_id"] for record in records}) != N_SOURCE_DOCUMENTS:
        raise RuntimeError("MLP1 global-gate row harvest is not one row per source document")

    staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"MLP1 row staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        staged = staging / "fineweb_32_source_documents.pt"
        torch.save(rows, staged)
        final_path = CACHE / staged.name
        entry = {
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
            "tensor_raw_sha256": registry.tensor_sha256(rows),
            "file_sha256": registry.file_sha256(staged),
            "cache_path": str(final_path.resolve()),
        }
        verify_frozen_snapshot(
            source_commit=source_commit, source_closure=source_closure,
            implementation_hashes=implementation_hashes,
            registry_files=registry_files, registry_hashes=registry_hashes,
            prior_tensor_hashes=prior_tensor_hashes, prior=prior, source=source,
        )
        installed = registry.install_cache_create_only(staging, CACHE)
        if installed != final_path or registry.file_sha256(installed) != entry["file_sha256"]:
            raise RuntimeError("installed MLP1 row cache differs from staged artifact")
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    verify_frozen_snapshot(
        source_commit=source_commit, source_closure=source_closure,
        implementation_hashes=implementation_hashes,
        registry_files=registry_files, registry_hashes=registry_hashes,
        prior_tensor_hashes=prior_tensor_hashes, prior=prior, source=source,
    )
    receipt = {
        "schema_version": 1,
        "receipt_kind": "mlp1_global_gate_v1_rows",
        "status": "frozen_before_any_global_gate_model_forward",
        "authority": (
            "pinned_local_ordered_manifest_source_document_balanced_"
            "registry_wide_exclusion"
        ),
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
        "source_commit": source_commit,
        "selection": {
            "start_dataset_document_index": START_DOCUMENT_INDEX,
            "n_source_documents": N_SOURCE_DOCUMENTS,
            "wave_source_documents": WAVE_DOCUMENTS,
            "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT,
            "token_length": TOKEN_LENGTH,
            "cohort_assignment": "first 16 ordered harvested documents fit; next 16 validation",
        },
        "sample_summary": summary,
        "entries": {"all": entry},
        "document_provenance": {"schema_version": 1, "sets": {"all": records}},
        "disjointness_gates": disjointness,
        "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
        "source_receipt_path": str(registry.BASE.CANONICAL_RECEIPT.resolve()),
        "source_receipt_sha256": registry_hashes[
            str(registry.BASE.CANONICAL_RECEIPT.resolve())
        ],
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
    verify_frozen_snapshot(
        source_commit=source_commit, source_closure=source_closure,
        implementation_hashes=implementation_hashes,
        registry_files=registry_files, registry_hashes=registry_hashes,
        prior_tensor_hashes=prior_tensor_hashes, prior=prior, source=source,
    )
    registry.write_json_create_only(receipt, RECEIPT)
    return receipt


def freeze() -> dict[str, Any]:
    descriptor = registry.acquire_lock(LOCK)
    try:
        return freeze_locked()
    finally:
        registry.release_lock(descriptor, LOCK)


def load_frozen_rows() -> tuple[dict[str, Any], torch.Tensor]:
    receipt = json.loads(RECEIPT.read_text())
    if (
        receipt.get("status") != "frozen_before_any_global_gate_model_forward"
        or receipt.get("authorized_for_scored_experiments") is not True
        or receipt.get("authorized_for_training") is not False
        or not all(receipt.get("disjointness_gates", {}).values())
    ):
        raise RuntimeError("MLP1 global-gate row receipt is not authoritative")
    entry = receipt["entries"]["all"]
    path = Path(entry["cache_path"])
    rows = torch.load(path, map_location="cpu", weights_only=True)
    records = receipt["document_provenance"]["sets"]["all"]
    if (
        tuple(rows.shape) != (N_SOURCE_DOCUMENTS, TOKEN_LENGTH)
        or rows.dtype != torch.long or len(records) != N_SOURCE_DOCUMENTS
        or len({record["document_id"] for record in records}) != N_SOURCE_DOCUMENTS
        or registry.tensor_sha256(rows) != entry["tensor_raw_sha256"]
        or registry.file_sha256(path) != entry["file_sha256"]
    ):
        raise RuntimeError("frozen MLP1 global-gate rows changed")
    return receipt, rows


if __name__ == "__main__":
    result = freeze()
    print(json.dumps({
        "receipt": str(RECEIPT),
        "summary": result["sample_summary"],
        "disjointness": result["disjointness_gates"],
        "exclusion_counts": result["exclusion_counts"],
        "tensor_sha256": result["entries"]["all"]["tensor_raw_sha256"],
    }, indent=2))
