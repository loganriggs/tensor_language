#!/usr/bin/env python3
"""Rung447: create-only teaching and sealed rows for candidate consequences."""

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
BANK = HERE / "prospective_consequence_candidate_bank_v1.json"
BANK_SHA = "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be"
SOURCE = Path(__file__).resolve()
CACHE = BQ / ".rowcache_simplicity_consequence_v1"
RECEIPT = BQ / "simplicity_consequence_v1_rows_receipt.json"
SLICE = slice(96, 192)

PARENTS = {
    "TEACHING": {
        "receipt": BQ / "mlp2_rank512_refit_v1_rows_receipt.json",
        "receipt_sha256": "e631c4ccf75ca561a4267c85330b7d2774717bad1920c6d5921b1258ddab10a9",
        "file_sha256": "774d34bda702bf78783639a6fb0c23986155b0be2da0adfcd31b77201f0f4a6e",
        "tensor_sha256": "580075ad9530b1ed832a56748badf3eb2416ca0c01c79c21b6026e84be41f9b7",
    },
    "SEALED_CONFIRMATION": {
        "receipt": BQ / "mlp2_trajectory_robust_r512_v2_physical_eval_rows_receipt.json",
        "receipt_sha256": "efe44941388878cfac1467508e2763b6a5db9b881be224c9db86314deaa61c2a",
        "file_sha256": "3e0bdab49a3413423cb1bae71fbcb8ab627bc2ace561e81ea6fec35fa94e02d9",
        "tensor_sha256": "1ea7decce7694afd84fdb0e375d9913bfeebb811163a9524184771af3dfba548",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def committed_source() -> tuple[str, str]:
    relative = str(SOURCE.relative_to(ROOT))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    digest = hashlib.sha256(blob).hexdigest()
    if sha256(SOURCE) != digest:
        raise RuntimeError("row freezer source is not the committed HEAD blob")
    return commit, digest


def load_parent(role: str, spec: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    if sha256(spec["receipt"]) != spec["receipt_sha256"]:
        raise RuntimeError(f"{role} parent receipt hash mismatch")
    receipt = json.loads(spec["receipt"].read_text())
    entry = receipt["entries"]["EVALUATION"]
    path = Path(entry["path"])
    if entry["file_sha256"] != spec["file_sha256"] or sha256(path) != spec["file_sha256"]:
        raise RuntimeError(f"{role} parent file hash mismatch")
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if entry["tensor_sha256"] != spec["tensor_sha256"] or tensor_sha256(rows) != spec["tensor_sha256"]:
        raise RuntimeError(f"{role} parent tensor hash mismatch")
    provenance = receipt["provenance"]["EVALUATION"]
    if len(rows) != 192 or len(provenance) != 192:
        raise RuntimeError(f"{role} parent length changed")
    return rows[SLICE].clone(), provenance[SLICE], receipt


def fit_rows(bank: dict[str, Any]) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]], dict[str, str]]:
    paths = sorted({
        artifact["path"]
        for row in bank["rows"]
        for artifact in row["required_artifacts"]
        if artifact["path"].startswith(".rowcache/")
    })
    full: set[tuple[int, ...]] = set()
    prefix: set[tuple[int, ...]] = set()
    hashes: dict[str, str] = {}
    for name in paths:
        path = BQ / name
        hashes[name] = sha256(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        value = value["rows"] if isinstance(value, dict) else value
        for row in value:
            item = tuple(int(x) for x in row[:257].tolist())
            full.add(item); prefix.add(item[:32])
    return full, prefix, hashes


def role_sets(rows: torch.Tensor, provenance: list[dict[str, Any]]) -> dict[str, set[Any]]:
    full = {tuple(int(x) for x in row.tolist()) for row in rows}
    return {
        "full": full,
        "prefix": {row[:32] for row in full},
        "document": {str(item["document_id"]) for item in provenance},
        "dataset_index": {int(item["dataset_document_index"]) for item in provenance},
    }


def save_create_only(value: torch.Tensor, path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as sink:
        torch.save(value, sink); sink.flush(); os.fsync(sink.fileno())


def main() -> None:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("rung447 output namespace already exists")
    if sha256(BANK) != BANK_SHA:
        raise RuntimeError("rung445 bank hash mismatch")
    source_commit, source_hash = committed_source()
    bank = json.loads(BANK.read_text())
    roles: dict[str, torch.Tensor] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    parent_receipts: dict[str, str] = {}
    for role, spec in PARENTS.items():
        roles[role], provenance[role], _receipt = load_parent(role, spec)
        parent_receipts[role] = spec["receipt_sha256"]
    sets = {role: role_sets(roles[role], provenance[role]) for role in roles}
    fit_full, fit_prefix, fit_hashes = fit_rows(bank)
    checks = {
        "shape_dtype": all(tuple(value.shape) == (96, 257) and value.dtype == torch.long
                           for value in roles.values()),
        "unique_within_roles": all(all(len(group) == 96 for group in sets[role].values()) for role in roles),
        "cross_role_full": sets["TEACHING"]["full"].isdisjoint(sets["SEALED_CONFIRMATION"]["full"]),
        "cross_role_prefix": sets["TEACHING"]["prefix"].isdisjoint(sets["SEALED_CONFIRMATION"]["prefix"]),
        "cross_role_documents": sets["TEACHING"]["document"].isdisjoint(sets["SEALED_CONFIRMATION"]["document"]),
        "cross_role_dataset_indices": sets["TEACHING"]["dataset_index"].isdisjoint(sets["SEALED_CONFIRMATION"]["dataset_index"]),
        "teaching_fit_full": sets["TEACHING"]["full"].isdisjoint(fit_full),
        "teaching_fit_prefix": sets["TEACHING"]["prefix"].isdisjoint(fit_prefix),
        "sealed_fit_full": sets["SEALED_CONFIRMATION"]["full"].isdisjoint(fit_full),
        "sealed_fit_prefix": sets["SEALED_CONFIRMATION"]["prefix"].isdisjoint(fit_prefix),
    }
    if not all(checks.values()):
        raise RuntimeError(f"rung447 separation failed: {checks}")
    CACHE.mkdir(mode=0o755)
    entries: dict[str, dict[str, Any]] = {}
    try:
        for role, value in roles.items():
            path = CACHE / f"{role.lower()}_96.pt"
            save_create_only(value, path)
            entries[role] = {
                "path": str(path.resolve()), "shape": list(value.shape), "dtype": str(value.dtype),
                "file_sha256": sha256(path), "tensor_sha256": tensor_sha256(value),
                "waves": [[0, 48], [48, 96]],
            }
        receipt = {
            "schema": "simplicity_consequence_v1_rows",
            "status": "roles_frozen_before_any_candidate_consequence",
            "source_commit": source_commit, "source_sha256": source_hash,
            "bank_sha256": BANK_SHA, "parent_receipt_sha256": parent_receipts,
            "parent_slice": [SLICE.start, SLICE.stop], "entries": entries,
            "provenance": provenance, "candidate_fit_cache_sha256": fit_hashes,
            "checks": checks,
            "outcome_access": {"model_loaded": False, "candidate_consequences_loaded": False},
        }
        descriptor = os.open(RECEIPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w") as sink:
            json.dump(receipt, sink, indent=2, sort_keys=True); sink.write("\n")
            sink.flush(); os.fsync(sink.fileno())
    except BaseException:
        if not RECEIPT.exists():
            for path in CACHE.glob("*"):
                path.unlink()
            CACHE.rmdir()
        raise
    print(json.dumps({
        "status": "complete", "rung": 447, "receipt": str(RECEIPT),
        "receipt_sha256": sha256(RECEIPT), "checks": checks,
        "teaching_tensor_sha256": entries["TEACHING"]["tensor_sha256"],
        "sealed_tensor_sha256": entries["SEALED_CONFIRMATION"]["tensor_sha256"],
        "pred_a_authority_and_closure": True,
        "pred_b_role_separation": True,
        "pred_c_candidate_fit_separation": True,
        "pred_d_deterministic_publication": True,
        "strong_null_row_authority_invalid": False,
        "next_step": "preregister_exact_teaching_consequence_harness",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

