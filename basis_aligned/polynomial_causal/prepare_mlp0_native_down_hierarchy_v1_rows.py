#!/usr/bin/env python3
"""Freeze source-document-balanced rows for the MLP0 native-Down screen.

This module is outcome blind.  It reads the already pinned FineWeb parquet, takes
exactly 384 new source documents in dataset order at ``skip=25000``, and retains at
most three 513-token chunks from each.  The two 192-document waves are fixed before
any candidate or model outcome is evaluated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Iterable

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(HERE))
import local_fineweb_harvest as local  # noqa: E402


START_DOCUMENT_INDEX = 25_000
N_SOURCE_DOCUMENTS = 384
WAVE_DOCUMENTS = 192
MAX_CHUNKS_PER_DOCUMENT = 3
TOKEN_LENGTH = 513
CANONICAL_RECEIPT = BQ / ".rowcache" / "fineweb_oracle_v2_receipt.json"
STAGE0_RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"
CACHE = BQ / ".rowcache_mlp0_native_down_hierarchy_v1"
RECEIPT = BQ / "mlp0_native_down_hierarchy_v1_rows_receipt.json"


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


def harvest_balanced_documents(
    texts: Iterable[tuple[str, str]],
    encode: Callable[[str], list[int]],
    seen_prefixes: set[tuple[int, ...]],
    excluded_documents: set[str],
    *,
    start_document_index: int = START_DOCUMENT_INDEX,
    n_source_documents: int = N_SOURCE_DOCUMENTS,
    wave_documents: int = WAVE_DOCUMENTS,
    max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
    token_length: int = TOKEN_LENGTH,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Select 1--K chunks from exactly N eligible, independent source documents."""
    if not (0 < wave_documents < n_source_documents):
        raise ValueError("wave_documents must split the positive source-document count")
    if min(start_document_index, max_chunks_per_document, token_length) < 1:
        raise ValueError("start, chunk cap, and token length must be positive")
    rows: list[list[int]] = []
    records: list[dict[str, Any]] = []
    selected_documents = 0
    selected_prefixes = set(seen_prefixes)
    for document_index, (document_id, text) in enumerate(texts):
        if document_index < start_document_index:
            continue
        if selected_documents >= n_source_documents:
            break
        if document_id in excluded_documents:
            continue
        if not isinstance(document_id, str) or not document_id or not isinstance(text, str):
            raise ValueError("texts must provide nonempty string document ids and text")
        tokens = encode(text)
        document_rows: list[tuple[list[int], int]] = []
        for start in range(0, len(tokens) - token_length, token_length):
            row = tokens[start:start + token_length]
            prefix = tuple(row[:32])
            if prefix in selected_prefixes:
                continue
            document_rows.append((row, start))
            selected_prefixes.add(prefix)
            if len(document_rows) >= max_chunks_per_document:
                break
        if not document_rows:
            continue
        wave = "A" if selected_documents < wave_documents else "B"
        ordinal = selected_documents
        for row, start in document_rows:
            rows.append(row)
            records.append({
                "document_id": document_id,
                "dataset_document_index": document_index,
                "source_document_ordinal": ordinal,
                "wave": wave,
                "chunk_id": start // token_length,
                "token_start": start,
            })
        selected_documents += 1
    tensor = torch.tensor(rows, dtype=torch.long)
    if selected_documents != n_source_documents or tensor.ndim != 2:
        raise RuntimeError(
            f"source ended after {selected_documents}/{n_source_documents} eligible documents"
        )
    if tensor.shape[1] != token_length:
        raise RuntimeError(f"wrong frozen token length: {tuple(tensor.shape)}")
    return tensor, records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_document: dict[str, int] = {}
    by_wave = {"A": set(), "B": set()}
    chunks_by_wave = {"A": 0, "B": 0}
    for record in records:
        document = record["document_id"]
        wave = record["wave"]
        by_document[document] = by_document.get(document, 0) + 1
        by_wave[wave].add(document)
        chunks_by_wave[wave] += 1
    counts = list(by_document.values())
    summary = {
        "n_source_documents": len(by_document),
        "n_chunks": len(records),
        "raw_prediction_positions": len(records) * (TOKEN_LENGTH - 1),
        "min_chunks_per_document": min(counts),
        "max_chunks_per_document": max(counts),
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
    expected = {
        "n_source_documents": N_SOURCE_DOCUMENTS,
        "min_chunks_per_document": 1,
        "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT,
    }
    if any(summary[key] != value for key, value in expected.items()):
        raise RuntimeError(f"balanced-document receipt invariant failed: {summary}")
    if any(summary["waves"][wave]["n_source_documents"] != WAVE_DOCUMENTS
           for wave in ("A", "B")):
        raise RuntimeError("replication waves are not 192 source documents each")
    return summary


def discover_prior_receipts() -> tuple[Path, ...]:
    paths = {CANONICAL_RECEIPT, STAGE0_RECEIPT}
    paths.update(BQ.glob("*rows_receipt.json"))
    paths.discard(RECEIPT)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing prior row receipts: {missing}")
    return tuple(sorted(paths))


def load_prior_identities(paths: tuple[Path, ...]) -> tuple[set[str], set[int], set, set]:
    documents: set[str] = set()
    indices: set[int] = set()
    full_rows: set[tuple[int, ...]] = set()
    prefixes: set[tuple[int, ...]] = set()
    seen_paths: set[Path] = set()
    for receipt_path in paths:
        receipt = json.loads(receipt_path.read_text())
        for records in receipt.get("document_provenance", {}).get("sets", {}).values():
            for record in records:
                if isinstance(record.get("document_id"), str):
                    documents.add(record["document_id"])
                if isinstance(record.get("dataset_document_index"), int):
                    indices.add(record["dataset_document_index"])
        for entry in receipt.get("entries", {}).values():
            raw_path = entry.get("cache_path") or entry.get("path")
            if not raw_path:
                continue
            path = Path(raw_path)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            rows = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
                raise RuntimeError(f"invalid prior row tensor: {path}")
            for row in rows:
                values = tuple(int(value) for value in row.tolist())
                full_rows.add(values)
                prefixes.add(values[:32])
    return documents, indices, full_rows, prefixes


def validate_eval_disjointness(
    rows: torch.Tensor,
    records: list[dict[str, Any]],
    prior: tuple[set[str], set[int], set, set],
) -> dict[str, bool]:
    prior_docs, prior_indices, prior_rows, prior_prefixes = prior
    documents = {record["document_id"] for record in records}
    indices = {record["dataset_document_index"] for record in records}
    full = {tuple(int(value) for value in row.tolist()) for row in rows}
    prefixes = {row[:32] for row in full}
    wave_a = {record["document_id"] for record in records if record["wave"] == "A"}
    wave_b = {record["document_id"] for record in records if record["wave"] == "B"}
    gates = {
        "source_documents_disjoint_from_prior": documents.isdisjoint(prior_docs),
        "dataset_document_indices_disjoint_from_prior": indices.isdisjoint(prior_indices),
        "full_rows_disjoint_from_prior": full.isdisjoint(prior_rows),
        "prefix32_disjoint_from_prior": prefixes.isdisjoint(prior_prefixes),
        "replication_waves_document_disjoint": wave_a.isdisjoint(wave_b),
    }
    if not all(gates.values()):
        raise RuntimeError(f"evaluation disjointness failed: {[k for k,v in gates.items() if not v]}")
    return gates


def validate_ordered_source() -> tuple[dict[str, Any], Path]:
    canonical = json.loads(CANONICAL_RECEIPT.read_text())
    gate = canonical.get("ordered_manifest_local_parquet_identity_gate", {})
    required = {
        "passed": True,
        "revision": local.PINNED_REVISION,
        "config": "default",
        "first_relative_path": local.PINNED_RELATIVE_PATH,
        "source_size": local.PINNED_SIZE,
        "source_sha256": local.PINNED_SHA256,
    }
    if canonical.get("authorized_for_scored_experiments") is not True:
        raise RuntimeError("canonical FineWeb receipt is not authoritative")
    if any(gate.get(key) != value for key, value in required.items()):
        raise RuntimeError("ordered FineWeb manifest identity changed")
    source = Path(gate["source_local_path"])
    if (not source.is_file() or source.stat().st_size != local.PINNED_SIZE
            or file_sha256(source) != local.PINNED_SHA256):
        raise RuntimeError("pinned FineWeb parquet identity failed")
    return canonical, source


def freeze() -> dict[str, Any]:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite native-Down row authority")
    canonical, source = validate_ordered_source()
    prior_paths = discover_prior_receipts()
    prior = load_prior_identities(prior_paths)

    import tiktoken
    reference = torch.load(BQ / "bilin18_eval_tokens_large.pt", map_location="cpu",
                           weights_only=True)
    seen_prefixes = {tuple(int(value) for value in row[:32].tolist()) for row in reference}
    seen_prefixes.update(prior[3])
    encoding = tiktoken.get_encoding("gpt2")
    rows, records = harvest_balanced_documents(
        local.parquet_texts([source]), encoding.encode_ordinary, seen_prefixes, prior[0]
    )
    summary = summarize_records(records)
    disjointness = validate_eval_disjointness(rows, records, prior)
    if source.stat().st_size != local.PINNED_SIZE or file_sha256(source) != local.PINNED_SHA256:
        raise RuntimeError("pinned FineWeb parquet changed during selection")

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
            "cache_path": str(final_path.resolve()),
        }
        os.replace(staging, CACHE)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    receipt = {
        "schema_version": 1,
        "receipt_kind": "mlp0_native_down_hierarchy_v1_rows",
        "status": "frozen_before_any_native_down_model_forward",
        "authority": "pinned_local_ordered_manifest_source_document_balanced",
        "authorized_for_scored_experiments": True,
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
        "source_receipt_path": str(CANONICAL_RECEIPT.resolve()),
        "source_receipt_sha256": file_sha256(CANONICAL_RECEIPT),
        "prior_role_receipts": {str(path.resolve()): file_sha256(path) for path in prior_paths},
        "implementation_hashes": {
            "row_freezer": file_sha256(Path(__file__)),
            "local_harvester": file_sha256(Path(local.__file__)),
        },
    }
    write_json_atomic(receipt, RECEIPT)
    return receipt


def load_frozen_rows(path: Path = RECEIPT) -> tuple[dict[str, Any], torch.Tensor]:
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "frozen_before_any_native_down_model_forward"
            or receipt.get("authorized_for_scored_experiments") is not True
            or not all(receipt.get("disjointness_gates", {}).values())):
        raise RuntimeError("native-Down row receipt is not authoritative")
    entry = receipt.get("entries", {}).get("eval", {})
    rows = torch.load(entry.get("cache_path", ""), map_location="cpu", weights_only=True)
    if (not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.shape[1] != TOKEN_LENGTH
            or rows.dtype != torch.long or tensor_sha256(rows) != entry.get("tensor_raw_sha256")):
        raise RuntimeError("frozen native-Down rows changed")
    summarize_records(receipt["document_provenance"]["sets"]["eval"])
    return receipt, rows


def main() -> None:
    receipt = freeze()
    _, rows = load_frozen_rows()
    print(json.dumps({
        "status": receipt["status"],
        "shape": list(rows.shape),
        "tensor_raw_sha256": receipt["entries"]["eval"]["tensor_raw_sha256"],
        "sample_summary": receipt["sample_summary"],
        "disjointness_gates": receipt["disjointness_gates"],
    }, indent=2))
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
