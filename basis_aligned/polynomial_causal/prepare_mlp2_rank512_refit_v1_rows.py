#!/usr/bin/env python3
"""Outcome-blind, create-only fresh rows for MLP2 rank-512 refitting.

This module delegates registry census and pinned FineWeb traversal to the tested
Block-3 freezer. It cannot import or load the model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Any, Mapping

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
AUDIT = HERE / "mlp2_rank512_refit_v1_independent_audit.json"
SOURCE_PATHS = tuple(dict.fromkeys((
    PREREG, FREEZER, RUNNER, TEST, *BASE.SOURCE_PATHS,
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_bilin18_observed_model_facade.py",
    HERE / "mlp2_cmr_v1_physical_program.py",
    HERE / "test_mlp2_cmr_v1_physical_program.py",
    ROOT / "jacclust/__init__.py",
    ROOT / "jacclust/tt_model.py",
)))

START_DOCUMENT_INDEX = 100_000
DOCUMENTS_PER_ROLE = 192
TOTAL_DOCUMENTS = 384
TOKEN_LENGTH = 257
PREFIX_LENGTH = 32
CACHE = BQ / ".rowcache_mlp2_rank512_refit_v1"
RECEIPT = BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
FAILURE = BQ / "mlp2_rank512_refit_v1_rows_failure.json"
LOCK = Path("/workspace/runs/.mlp2_rank512_refit_v1_rows.lock")

RunClaim = BASE.RunClaim
acquire_claim = BASE.acquire_claim
require_claim = BASE.require_claim
release_claim = BASE.release_claim
file_sha256 = BASE.file_sha256
tensor_sha256 = BASE.tensor_sha256


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes(commit: str) -> dict[str, str]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    output: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted row-freezer source: {relative}")
        output[relative] = digest
    return output


def validate_independent_audit(
    sources: Mapping[str, str], path: Path = AUDIT,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError("independent MLP2 refit audit is absent")
    before = file_sha256(path)
    raw = path.read_bytes()
    if file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError("independent MLP2 refit audit changed while reading")
    value = json.loads(raw)
    required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if set(value) != required or value.get("schema") != (
        "mlp2_rank512_refit_v1_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False or (
        not isinstance(value.get("tests_passed"), int) or value["tests_passed"] < 1
    ) or not isinstance(value.get("reviewer"), str) or not value["reviewer"]:
        raise RuntimeError("independent MLP2 refit audit is not an exact GO")
    if value.get("audited_source_hashes") != dict(sources):
        raise RuntimeError("independent MLP2 refit audit source binding changed")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("independent MLP2 refit audit commit binding changed")
    return value, before


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_create_only(
    path: Path, value: Mapping[str, Any], *, pre_link_check=None,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as sink:
            descriptor = None
            json.dump(value, sink, sort_keys=True, indent=2, allow_nan=False)
            sink.write("\n"); sink.flush(); os.fsync(sink.fileno())
        if pre_link_check is not None:
            pre_link_check()
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def split_rows(rows: torch.Tensor, records: list[dict[str, Any]]):
    if tuple(rows.shape) != (TOTAL_DOCUMENTS, TOKEN_LENGTH) or rows.dtype != torch.long \
            or len(records) != TOTAL_DOCUMENTS:
        raise RuntimeError("fresh row family changed")
    return {
        "TRAIN": (rows[:DOCUMENTS_PER_ROLE].clone(), records[:DOCUMENTS_PER_ROLE]),
        "EVALUATION": (rows[DOCUMENTS_PER_ROLE:].clone(), records[DOCUMENTS_PER_ROLE:]),
    }


def validate_selected(rows: torch.Tensor, records: list[dict[str, Any]], prior):
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


def verify_cache(path: Path, entry: Mapping[str, Any]) -> torch.Tensor:
    before = file_sha256(path)
    if before != entry.get("file_sha256"):
        raise RuntimeError("installed MLP2 refit row bytes changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before or not isinstance(value, torch.Tensor) or (
        value.dtype != torch.long or tuple(value.shape) != (DOCUMENTS_PER_ROLE, TOKEN_LENGTH)
    ) or tensor_sha256(value) != entry.get("tensor_sha256"):
        raise RuntimeError("installed MLP2 refit row tensor changed")
    return value


def verify_snapshot(snapshot: Mapping[str, Any]) -> None:
    if source_hashes(snapshot["commit"]) != snapshot["sources"]:
        raise RuntimeError("MLP2 refit source closure changed")
    current_files = BASE.discover_registry_files()
    if current_files != snapshot["registry_files"]:
        raise RuntimeError("MLP2 refit registry membership changed")
    current = BASE.load_registry_exclusions(current_files)
    if BASE.discover_registry_files() != current_files or current != snapshot["registry"]:
        raise RuntimeError("MLP2 refit registry contents changed")
    parquet = snapshot["parquet"]
    if parquet.stat().st_size != BASE.BASE.local.PINNED_SIZE or (
        file_sha256(parquet) != BASE.BASE.local.PINNED_SHA256
    ):
        raise RuntimeError("pinned FineWeb source changed")
    validate_independent_audit(snapshot["sources"])


def freeze_locked(claim: RunClaim) -> dict[str, Any]:
    require_claim(claim, LOCK)
    if CACHE.exists() or RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("MLP2 refit row namespace already exists")
    commit = git("rev-parse", "HEAD")
    sources = source_hashes(commit)
    audit, audit_hash = validate_independent_audit(sources)
    canonical, parquet = BASE.BASE.validate_ordered_source()
    registry_files = BASE.discover_registry_files()
    registry = BASE.load_registry_exclusions(registry_files)
    prior, registry_hashes, tensor_hashes, waiver_proofs, nonrow_proofs = registry
    snapshot = {
        "commit": commit, "sources": sources, "registry_files": registry_files,
        "registry": registry, "parquet": parquet,
    }
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
    entries: dict[str, dict[str, Any]] = {}
    try:
        for role, (tensor, _) in roles.items():
            staged = staging / f"{role.lower()}_192.pt"
            with staged.open("xb") as sink:
                torch.save(tensor, sink); sink.flush(); os.fsync(sink.fileno())
            entries[role] = {
                "path": str((CACHE / staged.name).resolve()),
                "file_sha256": file_sha256(staged),
                "tensor_sha256": tensor_sha256(tensor),
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            }
        _fsync_directory(staging)
        verify_snapshot(snapshot); require_claim(claim, LOCK)
        # One atomic directory rename publishes both roles together.  A crash can
        # leave either the private staging directory or the complete cache, never a
        # cache containing only one licensed role.
        os.rename(staging, CACHE)
        _fsync_directory(CACHE); _fsync_directory(CACHE.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    installed = {role: verify_cache(Path(entry["path"]), entry)
                 for role, entry in entries.items()}
    verify_snapshot(snapshot); require_claim(claim, LOCK)
    replay_rows, replay_records = BASE.harvest_fresh_documents(
        BASE.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=TOTAL_DOCUMENTS, token_length=TOKEN_LENGTH,
    )
    replay_roles = split_rows(replay_rows, replay_records)
    if any(not torch.equal(installed[role], replay_roles[role][0])
           or roles[role][1] != replay_roles[role][1] for role in roles):
        raise RuntimeError("MLP2 refit canonical row replay changed")
    receipt = {
        "schema": "mlp2_rank512_refit_v1_rows",
        "status": "fresh_roles_frozen_before_any_model_or_training_access",
        "source_commit": commit, "source_hashes": sources,
        "independent_audit": {
            "path": str(AUDIT.resolve()), "file_sha256": audit_hash,
            "audited_source_commit": audit["audited_source_commit"],
            "reviewer": audit["reviewer"], "tests_passed": audit["tests_passed"],
        },
        "selection": {"start_document_index": START_DOCUMENT_INDEX,
                      "documents_per_role": DOCUMENTS_PER_ROLE,
                      "token_length": TOKEN_LENGTH, "scored_slice": [64, 256]},
        "roles": {
            "TRAIN": {"authorized_for_training": True, "authorized_for_evaluation": False},
            "EVALUATION": {"authorized_for_training": False, "authorized_for_evaluation": True},
        },
        "entries": entries,
        "provenance": {role: value[1] for role, value in roles.items()},
        "disjointness": gates,
        "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
        "registry_hashes": registry_hashes, "prior_tensor_hashes": tensor_hashes,
        "waiver_proofs": waiver_proofs, "nonrow_proofs": nonrow_proofs,
        "outcome_access": {"model_loaded": False, "training_run": False},
    }

    def final_guard() -> None:
        require_claim(claim, LOCK)
        if RECEIPT.exists() or FAILURE.exists():
            raise RuntimeError("MLP2 refit row terminal artifact appeared")
        verify_snapshot(snapshot)
        for role, entry in entries.items():
            verify_cache(Path(entry["path"]), entry)
        # The registry/cache replay above is intentionally expensive.  Recheck the
        # claim and opposite terminal after it, immediately before the hard link.
        require_claim(claim, LOCK)
        if RECEIPT.exists() or FAILURE.exists():
            raise RuntimeError("MLP2 refit row terminal artifact raced publication")

    write_json_create_only(RECEIPT, receipt, pre_link_check=final_guard)
    return receipt


def freeze() -> dict[str, Any]:
    claim = acquire_claim(LOCK)
    try:
        return freeze_locked(claim)
    except BaseException as exc:
        failure = {
            "schema": "mlp2_rank512_refit_v1_rows_failure",
            "status": "terminal_failure_no_receipt",
            "error": repr(exc), "receipt_exists": RECEIPT.exists(),
            "cache_exists": CACHE.exists(), "cache_preserved": CACHE.exists(),
            "lock_inode": claim.inode, "lock_nonce_sha256": hashlib.sha256(
                claim.nonce.encode()).hexdigest(),
        }
        if not RECEIPT.exists() and not FAILURE.exists():
            def failure_guard() -> None:
                require_claim(claim, LOCK)
                if RECEIPT.exists() or FAILURE.exists():
                    raise RuntimeError("MLP2 refit row terminal artifact raced failure")
            write_json_create_only(FAILURE, failure, pre_link_check=failure_guard)
        raise
    finally:
        release_claim(claim, LOCK)


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True, indent=2))
