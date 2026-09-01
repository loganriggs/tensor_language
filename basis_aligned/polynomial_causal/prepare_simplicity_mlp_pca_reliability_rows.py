#!/usr/bin/env python3
"""Rung451: create-only independent rows for MLP-PCA reliability."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
SOURCE = Path(__file__).resolve()
BANK = HERE / "prospective_consequence_candidate_bank_v1.json"
PRIOR = BQ / "simplicity_consequence_v1_rows_receipt.json"
PARENT = BQ / "mlp0_c512_mlp2_full512_composition_v2_rows_receipt.json"
CACHE = BQ / ".rowcache_simplicity_mlp_pca_reliability_v1"
ROWS = CACHE / "reliability_192.pt"
RECEIPT = BQ / "simplicity_mlp_pca_reliability_v1_rows_receipt.json"
HASHES = {
    BANK: "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be",
    PRIOR: "1611c5bd60491a6b600950874ae55cd5925afad12096a48de3426e88e9cfc5d8",
    PARENT: "3b0de23f0d9f3ef781e25a6f6173071abacf4aab8965faa2bd6ab6b3a747bbb0",
}
PARENT_FILE = "c340a75c27afa0c37f4404c8ddc865275ed626542260c7f84ae14dba7c1ccbbc"
PARENT_TENSOR = "ff34039b5b6691eeee18942d7a31d46b8347ec00e1126956a8849f6c9765d1dc"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def committed_source() -> tuple[str, str]:
    relative = str(SOURCE.relative_to(ROOT))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    digest = hashlib.sha256(blob).hexdigest()
    if sha256(SOURCE) != digest:
        raise RuntimeError("reliability row freezer is not the committed HEAD blob")
    return commit, digest


def sets(rows: torch.Tensor, provenance: list[dict[str, Any]]) -> dict[str, set[Any]]:
    full = {tuple(int(x) for x in row.tolist()) for row in rows}
    return {
        "full": full, "prefix": {row[:32] for row in full},
        "document": {str(item["document_id"]) for item in provenance},
        "dataset_index": {int(item["dataset_document_index"]) for item in provenance},
    }


def fit_sets(bank: dict[str, Any]) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
    names = sorted({artifact["path"] for row in bank["rows"] for artifact in row["required_artifacts"]
                    if artifact["path"].startswith(".rowcache/")})
    full: set[tuple[int, ...]] = set(); prefix: set[tuple[int, ...]] = set()
    for name in names:
        value = torch.load(BQ / name, map_location="cpu", weights_only=True)
        value = value["rows"] if isinstance(value, dict) else value
        for row in value:
            item = tuple(int(x) for x in row[:257].tolist())
            full.add(item); prefix.add(item[:32])
    return full, prefix


def save_create_only(value: torch.Tensor, path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as sink:
        torch.save(value, sink); sink.flush(); os.fsync(sink.fileno())


def main() -> None:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("rung451 output namespace already exists")
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    commit, source_hash = committed_source()
    bank = json.loads(BANK.read_text())
    parent = json.loads(PARENT.read_text())
    entry = parent["entries"]["EVALUATION"]
    parent_path = Path(entry["path"])
    if entry["file_sha256"] != PARENT_FILE or sha256(parent_path) != PARENT_FILE:
        raise RuntimeError("parent file hash changed")
    rows = torch.load(parent_path, map_location="cpu", weights_only=True).long()
    if entry["tensor_sha256"] != PARENT_TENSOR or tensor_sha256(rows) != PARENT_TENSOR:
        raise RuntimeError("parent semantic tensor hash changed")
    provenance = parent["provenance"]["EVALUATION"]
    prior = json.loads(PRIOR.read_text())
    role_sets = {name: sets(torch.load(Path(item["path"]), map_location="cpu", weights_only=True),
                            prior["provenance"][name]) for name, item in prior["entries"].items()}
    current = sets(rows, provenance)
    fit_full, fit_prefix = fit_sets(bank)
    checks = {
        "shape_dtype": tuple(rows.shape) == (192, 257) and rows.dtype == torch.long,
        "unique": all(len(value) == 192 for value in current.values()),
        "teaching_full": current["full"].isdisjoint(role_sets["TEACHING"]["full"]),
        "teaching_prefix": current["prefix"].isdisjoint(role_sets["TEACHING"]["prefix"]),
        "teaching_documents": current["document"].isdisjoint(role_sets["TEACHING"]["document"]),
        "teaching_indices": current["dataset_index"].isdisjoint(role_sets["TEACHING"]["dataset_index"]),
        "sealed_full": current["full"].isdisjoint(role_sets["SEALED_CONFIRMATION"]["full"]),
        "sealed_prefix": current["prefix"].isdisjoint(role_sets["SEALED_CONFIRMATION"]["prefix"]),
        "sealed_documents": current["document"].isdisjoint(role_sets["SEALED_CONFIRMATION"]["document"]),
        "sealed_indices": current["dataset_index"].isdisjoint(role_sets["SEALED_CONFIRMATION"]["dataset_index"]),
        "fit_full": current["full"].isdisjoint(fit_full),
        "fit_prefix": current["prefix"].isdisjoint(fit_prefix),
    }
    if not all(checks.values()):
        raise RuntimeError(f"rung451 separation failure: {checks}")
    CACHE.mkdir(mode=0o755)
    try:
        save_create_only(rows, ROWS)
        payload = {
            "schema": "simplicity_mlp_pca_reliability_v1_rows", "status": "frozen_before_reliability_outcomes",
            "source_commit": commit, "source_sha256": source_hash,
            "bank_sha256": HASHES[BANK], "prior_roles_receipt_sha256": HASHES[PRIOR],
            "parent_receipt_sha256": HASHES[PARENT], "parent_role": "EVALUATION",
            "entry": {"path": str(ROWS.resolve()), "shape": list(rows.shape), "dtype": str(rows.dtype),
                      "file_sha256": sha256(ROWS), "tensor_sha256": tensor_sha256(rows),
                      "waves": [[0, 96], [96, 192]]},
            "provenance": provenance, "checks": checks,
            "outcome_access": {"model_loaded": False, "candidate_consequences_loaded": False},
        }
        descriptor = os.open(RECEIPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True); sink.write("\n")
            sink.flush(); os.fsync(sink.fileno())
    except BaseException:
        if not RECEIPT.exists():
            for path in CACHE.glob("*"):
                path.unlink()
            CACHE.rmdir()
        raise
    print(json.dumps({
        "status": "complete", "rung": 451, "receipt_sha256": sha256(RECEIPT),
        "tensor_sha256": tensor_sha256(rows), "checks": checks,
        "pred_a_authority_and_closure": True, "pred_b_unique_role": True,
        "pred_c_disjoint": True, "pred_d_create_only": True,
        "strong_null_row_authority_invalid": False,
        "next_step": "preregister_tie_aware_mlp_pca_reliability",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
