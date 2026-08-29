#!/usr/bin/env python3
"""Outcome-blind fresh TRAIN/EVALUATION rows for MLP2 rank-512 refitting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
BASE_PATH = HERE / "prepare_block3_native_down_behavioral_port_v1_rows.py"
SPEC = importlib.util.spec_from_file_location("mlp2_refit_row_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

PREREG = HERE / "MLP2_RANK512_REFIT_V1_PREREGISTRATION.md"
RUNNER = HERE / "run_mlp2_rank512_refit_v1.py"
TEST = HERE / "test_mlp2_rank512_refit_v1.py"
FREEZER = Path(__file__).resolve()
SOURCE_PATHS = (PREREG, FREEZER, RUNNER, TEST)

START_DOCUMENT_INDEX = 100_000
DOCUMENTS_PER_ROLE = 192
TOTAL_DOCUMENTS = 384
TOKEN_LENGTH = 257
PREFIX_LENGTH = 32
CACHE = BQ / ".rowcache_mlp2_rank512_refit_v1"
RECEIPT = BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
LOCK = Path("/workspace/runs/.mlp2_rank512_refit_v1_rows.lock")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes(commit: str) -> dict[str, str]:
    if git("rev-parse", "HEAD") != commit or git("rev-parse", "origin/main") != commit:
        raise RuntimeError("row freezer requires synchronized HEAD/origin")
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted row-freezer source: {relative}")
        output[relative] = digest
    return output


def write_json_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def split_rows(rows: torch.Tensor, records: list[dict[str, Any]]):
    if rows.shape != (TOTAL_DOCUMENTS, TOKEN_LENGTH) or len(records) != TOTAL_DOCUMENTS:
        raise RuntimeError("fresh row family changed")
    return {
        "TRAIN": (rows[:DOCUMENTS_PER_ROLE].clone(), records[:DOCUMENTS_PER_ROLE]),
        "EVALUATION": (rows[DOCUMENTS_PER_ROLE:].clone(), records[DOCUMENTS_PER_ROLE:]),
    }


def validate_selected(
    rows: torch.Tensor, records: list[dict[str, Any]], prior,
) -> dict[str, bool]:
    documents = [record["document_id"] for record in records]
    indices = [record["dataset_document_index"] for record in records]
    full = {tuple(int(v) for v in row.tolist()) for row in rows}
    prefixes = {row[:PREFIX_LENGTH] for row in full}
    prior_rows = {row[:TOKEN_LENGTH] for row in prior[2] if len(row) >= TOKEN_LENGTH}
    gates = {
        "unique_source_documents": len(set(documents)) == TOTAL_DOCUMENTS,
        "unique_dataset_indices": len(set(indices)) == TOTAL_DOCUMENTS,
        "unique_full_rows": len(full) == TOTAL_DOCUMENTS,
        "unique_prefix32": len(prefixes) == TOTAL_DOCUMENTS,
        "source_documents_disjoint_from_registry": set(documents).isdisjoint(prior[0]),
        "dataset_indices_disjoint_from_registry": set(indices).isdisjoint(prior[1]),
        "full_rows_disjoint_from_registry": full.isdisjoint(prior_rows),
        "prefix32_disjoint_from_registry": prefixes.isdisjoint(prior[3]),
    }
    if not all(gates.values()):
        raise RuntimeError(f"fresh MLP2 refit rows failed: {gates}")
    return gates


def freeze() -> dict[str, Any]:
    if CACHE.exists() or RECEIPT.exists() or LOCK.exists():
        raise RuntimeError("MLP2 refit row namespace already exists")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        commit = git("rev-parse", "HEAD")
        sources = source_hashes(commit)
        canonical, parquet = BASE.BASE.validate_ordered_source()
        registry_files = BASE.discover_registry_files()
        prior, registry_hashes, tensor_hashes, waiver_proofs, nonrow_proofs = (
            BASE.load_registry_exclusions(registry_files)
        )
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        rows, records = BASE.harvest_fresh_documents(
            BASE.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
            start_document_index=START_DOCUMENT_INDEX,
            n_source_documents=TOTAL_DOCUMENTS, token_length=TOKEN_LENGTH,
        )
        gates = validate_selected(rows, records, prior)
        roles = split_rows(rows, records)
        staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
        staging.mkdir(parents=True, exist_ok=False)
        entries = {}
        try:
            for role, (tensor, _) in roles.items():
                staged = staging / f"{role.lower()}_192.pt"
                torch.save(tensor, staged)
                entries[role] = {
                    "path": str((CACHE / staged.name).resolve()),
                    "file_sha256": file_sha256(staged),
                    "tensor_sha256": tensor_sha256(tensor),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            if source_hashes(commit) != sources:
                raise RuntimeError("row-freezer source changed during harvest")
            CACHE.mkdir(parents=False, exist_ok=False)
            for staged in staging.iterdir():
                os.replace(staged, CACHE / staged.name)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        for role, (tensor, _) in roles.items():
            entry = entries[role]
            path = Path(entry["path"])
            replay = torch.load(path, map_location="cpu", weights_only=True)
            if file_sha256(path) != entry["file_sha256"] or not torch.equal(replay, tensor):
                raise RuntimeError("installed MLP2 refit rows changed")
        receipt = {
            "schema": "mlp2_rank512_refit_v1_rows",
            "status": "fresh_roles_frozen_before_any_model_or_training_access",
            "source_commit": commit,
            "source_hashes": sources,
            "selection": {
                "start_document_index": START_DOCUMENT_INDEX,
                "documents_per_role": DOCUMENTS_PER_ROLE,
                "token_length": TOKEN_LENGTH,
                "scored_slice": [64, 256],
            },
            "roles": {
                "TRAIN": {"authorized_for_training": True, "authorized_for_evaluation": False},
                "EVALUATION": {"authorized_for_training": False, "authorized_for_evaluation": True},
            },
            "entries": entries,
            "provenance": {role: records for role, (_, records) in roles.items()},
            "disjointness": gates,
            "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
            "registry_hashes": registry_hashes,
            "prior_tensor_hashes": tensor_hashes,
            "waiver_proofs": waiver_proofs,
            "nonrow_proofs": nonrow_proofs,
            "outcome_access": {"model_loaded": False, "training_run": False},
        }
        write_json_create_only(RECEIPT, receipt)
        return receipt
    finally:
        os.close(fd)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True, indent=2))
