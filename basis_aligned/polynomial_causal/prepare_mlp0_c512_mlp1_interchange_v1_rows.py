#!/usr/bin/env python3
"""Freeze new source-document rows for the C512/MLP1 physical 2x2.

This file is outcome blind and performs no model forward. It deliberately reuses
the already-tested local FineWeb identity and balanced-document primitives from the
native-Down v1 freezer while writing a distinct, append-only authority.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
BASE_PATH = HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py"
BASE_SPEC = importlib.util.spec_from_file_location("native_down_rows_v1", BASE_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE)

START_DOCUMENT_INDEX = 43_000
N_SOURCE_DOCUMENTS = 384
WAVE_DOCUMENTS = 192
MAX_CHUNKS_PER_DOCUMENT = 3
TOKEN_LENGTH = 513
CACHE = BQ / ".rowcache_mlp0_c512_mlp1_interchange_v1"
RECEIPT = BQ / "mlp0_c512_mlp1_interchange_v1_rows_receipt.json"
SPECIFICATION = HERE / "MLP0_C512_MLP1_INTERCHANGE_SPEC.md"
CODE_REGISTER = HERE / "code_oracle_corpus_v2.pt"


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


def discover_prior_receipts() -> tuple[Path, ...]:
    paths = {BASE.CANONICAL_RECEIPT}
    paths.update(BQ.glob("*rows_receipt.json"))
    paths.discard(RECEIPT)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing prior row receipts: {missing}")
    return tuple(sorted(paths))


def validate_code_register() -> dict[str, Any]:
    artifact = torch.load(CODE_REGISTER, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or set(artifact) != {"rows", "manifest"}:
        raise RuntimeError("code register is not the frozen v2 dictionary")
    rows, manifest = artifact["rows"], artifact["manifest"]
    if (not isinstance(rows, torch.Tensor) or rows.dtype != torch.long
            or tuple(rows.shape) != (480, 257)):
        raise RuntimeError("code register tensor identity changed")
    expected = {
        "schema_version": 2,
        "shape": [480, 257],
        "splits": {"basis": [0, 96], "discovery": [96, 288], "heldout": [288, 480]},
        "file_disjoint_splits": True,
        "no_row_crosses_file_boundary": True,
        "tensor_raw_sha256": "62adc15486397152102eba6d0fa8b6b77553271a5bd5fb5a0ff73930a1a82d88",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("code register manifest identity changed")
    heldout_files = manifest.get("files", {}).get("heldout", [])
    if len(heldout_files) != 48 or manifest.get("split_cluster_counts", {}).get("heldout") != 48:
        raise RuntimeError("code register heldout file count changed")
    return {
        "path": str(CODE_REGISTER.resolve()),
        "file_sha256": file_sha256(CODE_REGISTER),
        "tensor_raw_sha256": tensor_sha256(rows),
        "heldout_row_interval": [288, 480],
        "heldout_rows": 192,
        "heldout_source_files": 48,
        "resampling_unit": "source_file",
    }


def freeze() -> dict[str, Any]:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite C512/MLP1 interchange row authority")
    if not SPECIFICATION.is_file():
        raise RuntimeError("missing interchange specification")

    canonical, source = BASE.validate_ordered_source()
    prior_paths = discover_prior_receipts()
    prior = BASE.load_prior_identities(prior_paths)
    code_register = validate_code_register()

    import tiktoken
    reference = torch.load(
        BQ / "bilin18_eval_tokens_large.pt", map_location="cpu", weights_only=True
    )
    seen_prefixes = {tuple(int(value) for value in row[:32].tolist()) for row in reference}
    seen_prefixes.update(prior[3])
    encoding = tiktoken.get_encoding("gpt2")
    rows, records = BASE.harvest_balanced_documents(
        BASE.local.parquet_texts([source]),
        encoding.encode_ordinary,
        seen_prefixes,
        prior[0],
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=N_SOURCE_DOCUMENTS,
        wave_documents=WAVE_DOCUMENTS,
        max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
        token_length=TOKEN_LENGTH,
    )
    summary = BASE.summarize_records(records)
    disjointness = BASE.validate_eval_disjointness(rows, records, prior)
    if source.stat().st_size != BASE.local.PINNED_SIZE or file_sha256(source) != BASE.local.PINNED_SHA256:
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
        "receipt_kind": "mlp0_c512_mlp1_interchange_v1_rows",
        "status": "frozen_before_any_c512_interchange_model_forward",
        "authority": "pinned_local_ordered_manifest_source_document_balanced",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
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
        "prior_role_receipts": {str(path.resolve()): file_sha256(path) for path in prior_paths},
        "code_ood_register": code_register,
        "implementation_hashes": {
            "specification": file_sha256(SPECIFICATION),
            "row_freezer": file_sha256(Path(__file__)),
            "base_row_freezer": file_sha256(BASE_PATH),
            "local_harvester": file_sha256(Path(BASE.local.__file__)),
        },
    }
    write_json_atomic(receipt, RECEIPT)
    return receipt


def load_frozen_rows(path: Path = RECEIPT) -> tuple[dict[str, Any], torch.Tensor]:
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "frozen_before_any_c512_interchange_model_forward"
            or receipt.get("authorized_for_scored_experiments") is not True
            or receipt.get("authorized_for_training") is not False
            or not all(receipt.get("disjointness_gates", {}).values())):
        raise RuntimeError("C512/MLP1 interchange row receipt is not authoritative")
    entry = receipt.get("entries", {}).get("eval", {})
    rows = torch.load(entry.get("cache_path", ""), map_location="cpu", weights_only=True)
    if (not isinstance(rows, torch.Tensor) or rows.dtype != torch.long
            or tuple(rows.shape[1:]) != (TOKEN_LENGTH,)
            or tensor_sha256(rows) != entry.get("tensor_raw_sha256")):
        raise RuntimeError("frozen C512/MLP1 interchange rows changed")
    BASE.summarize_records(receipt["document_provenance"]["sets"]["eval"])
    return receipt, rows


if __name__ == "__main__":
    frozen = freeze()
    print(json.dumps({
        "receipt": str(RECEIPT),
        "summary": frozen["sample_summary"],
        "disjointness": frozen["disjointness_gates"],
        "tensor_sha256": frozen["entries"]["eval"]["tensor_raw_sha256"],
    }, indent=2))
