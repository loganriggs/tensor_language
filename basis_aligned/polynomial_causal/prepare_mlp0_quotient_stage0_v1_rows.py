#!/usr/bin/env python3
"""Freeze network-independent rows for the corrected MLP0 Stage-0 screen.

The abandoned development collector exposed ``skip=17000`` without producing an
outcome.  V1 therefore uses a new prospective evaluation window at ``skip=21000``.
Rows are rebuilt from the pinned first FineWeb parquet whose ordered-manifest
identity is already certified by ``fineweb_oracle_v2_receipt.json``.  Only the
evaluation role must be new: the fit role deliberately reproduces the historical
``mlp0_downstream_clusters.py`` construction at ``n=960, skip=80``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(HERE))
import local_fineweb_harvest as local  # noqa: E402


FIT_SPEC = (960, 80)
EVAL_SPEC = (192, 21000)
SPECS = (FIT_SPEC, EVAL_SPEC)
CANONICAL_RECEIPT = BQ / ".rowcache" / "fineweb_oracle_v2_receipt.json"
PRIOR_RECEIPTS = (
    CANONICAL_RECEIPT,
    BQ / "early_mlp_affine_compiler_v1_rows_receipt.json",
    BQ / "early_mlp_state_complete_compiler_v2_rows_receipt.json",
    BQ / "early_mlp_state_complete_compiler_v21_rows_receipt.json",
)
CACHE = BQ / ".rowcache_mlp0_quotient_stage0_v1"
RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"


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


def _validate_ordered_source() -> tuple[dict[str, Any], Path]:
    receipt = json.loads(CANONICAL_RECEIPT.read_text())
    gate = receipt.get("ordered_manifest_local_parquet_identity_gate", {})
    required = {
        "passed": True,
        "revision": local.PINNED_REVISION,
        "config": "default",
        "first_relative_path": local.PINNED_RELATIVE_PATH,
        "source_size": local.PINNED_SIZE,
        "source_sha256": local.PINNED_SHA256,
    }
    if receipt.get("authorized_for_scored_experiments") is not True:
        raise RuntimeError("canonical FineWeb receipt is not authoritative")
    if any(gate.get(key) != value for key, value in required.items()):
        raise RuntimeError("canonical FineWeb ordered-manifest gate changed")
    source = Path(gate["source_local_path"])
    if (not source.is_file() or source.stat().st_size != local.PINNED_SIZE
            or file_sha256(source) != local.PINNED_SHA256):
        raise RuntimeError("pinned FineWeb parquet identity failed")
    return receipt, source


def _load_prior_roles(paths: tuple[Path, ...] = PRIOR_RECEIPTS) -> tuple[set, set, set, set]:
    documents: set[str] = set()
    document_indices: set[int] = set()
    full_rows: set[tuple[int, ...]] = set()
    prefixes: set[tuple[int, ...]] = set()
    seen_cache_paths: set[Path] = set()
    for receipt_path in paths:
        receipt = json.loads(receipt_path.read_text())
        records_by_role = receipt.get("document_provenance", {}).get("sets", {})
        for records in records_by_role.values():
            for record in records:
                if isinstance(record.get("document_id"), str):
                    documents.add(record["document_id"])
                if isinstance(record.get("dataset_document_index"), int):
                    document_indices.add(record["dataset_document_index"])
        for entry in receipt.get("entries", {}).values():
            raw = entry.get("cache_path") or entry.get("path")
            if not raw:
                continue
            cache_path = Path(raw)
            if cache_path in seen_cache_paths:
                continue
            seen_cache_paths.add(cache_path)
            rows = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
                raise RuntimeError(f"invalid prior row artifact: {cache_path}")
            for row in rows:
                values = tuple(int(value) for value in row.tolist())
                full_rows.add(values)
                prefixes.add(values[:32])
    return documents, document_indices, full_rows, prefixes


def validate_new_eval(
    fit_rows: torch.Tensor,
    eval_rows: torch.Tensor,
    fit_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    prior: tuple[set, set, set, set],
) -> dict[str, bool]:
    prior_docs, prior_indices, prior_rows, prior_prefixes = prior
    fit_docs = {row["document_id"] for row in fit_records}
    eval_docs = {row["document_id"] for row in eval_records}
    eval_indices = {row["dataset_document_index"] for row in eval_records}
    fit_full = {tuple(int(x) for x in row.tolist()) for row in fit_rows}
    eval_full = {tuple(int(x) for x in row.tolist()) for row in eval_rows}
    fit_prefix = {row[:32] for row in fit_full}
    eval_prefix = {row[:32] for row in eval_full}
    gates = {
        "eval_document_disjoint_from_fit": eval_docs.isdisjoint(fit_docs),
        "eval_full_row_disjoint_from_fit": eval_full.isdisjoint(fit_full),
        "eval_prefix32_disjoint_from_fit": eval_prefix.isdisjoint(fit_prefix),
        "eval_document_disjoint_from_prior_roles": eval_docs.isdisjoint(prior_docs),
        "eval_document_index_disjoint_from_prior_roles": eval_indices.isdisjoint(prior_indices),
        "eval_full_row_disjoint_from_prior_roles": eval_full.isdisjoint(prior_rows),
        "eval_prefix32_disjoint_from_prior_roles": eval_prefix.isdisjoint(prior_prefixes),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"prospective evaluation disjointness failed: {failed}")
    return gates


def freeze() -> dict[str, Any]:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite the v1 MLP0 row authority namespace")
    canonical, source = _validate_ordered_source()
    import tiktoken

    reference = torch.load(BQ / "bilin18_eval_tokens_large.pt", map_location="cpu",
                           weights_only=True)
    seen = {tuple(row[:32].tolist()) for row in reference}
    encoding = tiktoken.get_encoding("gpt2")
    rows, provenance = local.harvest_texts(
        local.parquet_texts([source]), SPECS, encoding.encode_ordinary, seen
    )
    prior = _load_prior_roles()
    gates = validate_new_eval(
        rows[FIT_SPEC], rows[EVAL_SPEC], provenance[FIT_SPEC], provenance[EVAL_SPEC], prior
    )
    staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        entries = {}
        records = {}
        for role, spec in (("fit", FIT_SPEC), ("eval", EVAL_SPEC)):
            path = staging / f"{role}_n{spec[0]}_skip{spec[1]}.pt"
            torch.save(rows[spec], path)
            entries[role] = {
                "n": spec[0], "skip": spec[1], "shape": list(rows[spec].shape),
                "dtype": str(rows[spec].dtype), "tensor_raw_sha256": tensor_sha256(rows[spec]),
                "cache_path": str((CACHE / path.name).resolve()),
            }
            records[role] = provenance[spec]
        os.replace(staging, CACHE)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    receipt = {
        "schema_version": 1,
        "receipt_kind": "mlp0_quotient_stage0_v1_rows",
        "status": "frozen_before_any_v1_model_forward",
        "authority": "pinned_local_ordered_manifest",
        "authorized_for_scored_experiments": True,
        "role_designation": {
            "fit": "historical construction role; overlap permitted",
            "eval": "new prospective confirmatory role; outcome-unexposed at freeze",
        },
        "abandoned_development_window": {"n": 192, "skip": 17000,
            "reason": "collector/source/cell/KL defects; no result artifact"},
        "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
        "source_receipt_path": str(CANONICAL_RECEIPT.resolve()),
        "source_receipt_sha256": file_sha256(CANONICAL_RECEIPT),
        "implementation_hashes": {
            "row_freezer": file_sha256(Path(__file__)),
            "local_harvester": file_sha256(Path(local.__file__)),
        },
        "entries": entries,
        "document_provenance": {"schema_version": 1, "sets": records},
        "disjointness_gates": gates,
        "prior_role_receipts": {
            str(path.resolve()): file_sha256(path) for path in PRIOR_RECEIPTS
        },
    }
    write_json_atomic(receipt, RECEIPT)
    return receipt


def load_frozen_rows(path: Path = RECEIPT) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "frozen_before_any_v1_model_forward"
            or receipt.get("authorized_for_scored_experiments") is not True
            or not all(receipt.get("disjointness_gates", {}).values())):
        raise RuntimeError("MLP0 v1 row receipt is not authoritative")
    loaded = {}
    for role, spec in (("fit", FIT_SPEC), ("eval", EVAL_SPEC)):
        entry = receipt.get("entries", {}).get(role, {})
        rows = torch.load(entry.get("cache_path", ""), map_location="cpu", weights_only=True)
        if (tuple(rows.shape) != (spec[0], 513) or rows.dtype != torch.long
                or tensor_sha256(rows) != entry.get("tensor_raw_sha256")):
            raise RuntimeError(f"frozen MLP0 {role} rows changed")
        loaded[role] = rows
    return receipt, loaded


def main() -> None:
    receipt = freeze()
    load_frozen_rows()
    print(json.dumps({
        "status": receipt["status"],
        "entries": {key: value["tensor_raw_sha256"] for key, value in receipt["entries"].items()},
        "disjointness_gates": receipt["disjointness_gates"],
    }, indent=2))
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
