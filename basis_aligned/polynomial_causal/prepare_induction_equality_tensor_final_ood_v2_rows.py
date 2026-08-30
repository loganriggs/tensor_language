#!/usr/bin/env python3
"""Fresh, metadata-disjoint row freezer for equality-tensor FINAL/OOD v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Any, Callable, Mapping

import torch

import prepare_block3_native_down_behavioral_port_v1_rows as natural
import prepare_terminal_copy_induction_v1_rows as base
import terminal_copy_induction_v1 as contract


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = ROOT / "basis_aligned/bilinear_quotient"
PREREG = HERE / "INDUCTION_EQUALITY_TENSOR_FINAL_OOD_V2_PREREGISTRATION.md"
V1_PREREG = HERE / "INDUCTION_EQUALITY_TENSOR_FINAL_OOD_PREREGISTRATION.md"
V1_AUDIT = HERE / "induction_equality_tensor_final_ood_independent_audit.json"
DISCOVERY = HERE / "induction_equality_tensor_discovery.json"
AUDIT = HERE / "induction_equality_tensor_final_ood_v2_rows_independent_audit.json"
CACHE = BQ / ".rowcache_induction_equality_tensor_final_ood_v2"
RECEIPT = BQ / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
FAILURE = BQ / "induction_equality_tensor_final_ood_v2_rows_failure.json"
LOCK = Path("/workspace/runs/.induction_equality_tensor_final_ood_v2_rows.lock")
ROLES = ("label_fit", "final_natural", "ood_code")
N = 192
START_DOCUMENT_INDEX = 180_000
DISCOVERY_SHA256 = "0b826952d227c6f2c9e8b0fadf19aeb28edcd4153a52e4b67777a587733e184b"
V1_AUDIT_SHA256 = "3fae8d163a367c2af600fbe584f457ace7537a9688e3b091c379f7ebc9b043da"
DIRECT_SOURCES = (
    Path(__file__).resolve(), HERE / "test_prepare_induction_equality_tensor_final_ood_v2_rows.py",
    PREREG, V1_PREREG, V1_AUDIT, DISCOVERY,
)
SOURCE_PATHS = tuple(dict.fromkeys((*DIRECT_SOURCES, *base.SOURCE_PATHS)))
FORBIDDEN_V1_ROLE_NAMES = {"final_natural.pt", "ood_code.pt"}


def file_sha256(path: Path) -> str:
    return natural.file_sha256(path)


def source_closure(commit: str) -> dict[str, str]:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"v2 row source drift: {relative}")
        output[relative] = digest
    return output


def validate_audit(commit: str, sources: Mapping[str, str]) -> dict[str, Any]:
    before = file_sha256(AUDIT)
    payload = json.loads(AUDIT.read_bytes())
    if file_sha256(AUDIT) != before or set(payload) != {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    } or payload.get("schema") != "induction_equality_tensor_final_ood_v2_rows_independent_audit" \
            or payload.get("status") != "GO" or payload.get("outcome_access") is not False \
            or payload.get("audited_source_commit") != commit \
            or payload.get("audited_source_hashes") != dict(sources) \
            or type(payload.get("tests_passed")) is not int or payload["tests_passed"] < 1 \
            or not isinstance(payload.get("reviewer"), str) or not payload["reviewer"]:
        raise RuntimeError("v2 row audit is not an exact source-bound GO")
    return payload


def audited_source_binding() -> tuple[str, dict[str, str], dict[str, Any]]:
    """Select immutable source identity from the audit, not moving shared HEAD."""
    before = file_sha256(AUDIT)
    raw = AUDIT.read_bytes()
    if file_sha256(AUDIT) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError("v2 row audit changed while selecting its source commit")
    candidate = json.loads(raw)
    commit = candidate.get("audited_source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("v2 row audit source commit is malformed")
    sources = source_closure(commit)
    return commit, sources, validate_audit(commit, sources)


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def metadata_registry_snapshot(registry_files: tuple[Path, ...]) -> tuple[dict[str, Any], dict[str, str]]:
    """Collect exclusion identities without opening any registered tensor."""
    documents, indices, code_paths, normalized, code_sources = set(), set(), set(), set(), set()
    hashes = {}
    forbidden_references = set()
    for path in registry_files:
        before = file_sha256(path)
        raw = path.read_bytes()
        if file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
            raise RuntimeError("registry changed during metadata-only read")
        hashes[str(path.resolve())] = before
        payload = json.loads(raw)
        registry_commit = payload.get("source_commit")
        for value in _walk(payload):
            if isinstance(value, str) and Path(value).name in FORBIDDEN_V1_ROLE_NAMES:
                forbidden_references.add(str(Path(value).resolve()))
            if not isinstance(value, Mapping):
                continue
            if isinstance(value.get("document_id"), str):
                documents.add(value["document_id"])
            if type(value.get("dataset_document_index")) is int:
                indices.add(value["dataset_document_index"])
            candidate = value.get("path")
            if isinstance(candidate, str) and candidate.endswith(".py") and isinstance(
                value.get("blob_sha256"), str
            ):
                code_paths.add(candidate)
                if isinstance(value.get("normalized_python_sha256"), str):
                    normalized.add(value["normalized_python_sha256"])
                else:
                    if not isinstance(registry_commit, str) or len(registry_commit) != 40:
                        raise RuntimeError(
                            "prior code record lacks both normalized hash and source commit"
                        )
                    code_sources.add((registry_commit, candidate, value["blob_sha256"]))
    return ({
        "documents": documents, "indices": indices, "code_paths": code_paths,
        "normalized": normalized, "forbidden_v1_role_references": forbidden_references,
        "code_sources_missing_normalized": code_sources,
    }, hashes)


def recover_prior_normalized_hashes(
    code_sources: set[tuple[str, str, str]],
) -> set[str]:
    """Recover historical normalized hashes from authority-bound commit/path/blob triples."""

    if type(code_sources) is not set:
        raise TypeError("historical code sources must be one exact set")
    output = set()
    for item in sorted(code_sources):
        if type(item) is not tuple or len(item) != 3:
            raise ValueError("historical code source triple is malformed")
        commit, path, expected_blob_sha256 = item
        if len(commit) != 40 or not path.endswith(".py") or len(expected_blob_sha256) != 64:
            raise ValueError("historical code source identity is malformed")
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0 or hashlib.sha256(completed.stdout).hexdigest() != (
            expected_blob_sha256
        ):
            raise RuntimeError("historical code blob cannot be recovered exactly")
        output.add(base.normalized_python_sha256(completed.stdout))
    return output


def _registry_replay(files, expected, hashes):
    current_files = natural.discover_registry_files()
    current, current_hashes = metadata_registry_snapshot(current_files)
    if current_files != files or current != expected or current_hashes != hashes:
        raise RuntimeError("metadata registry changed")


def _payload(role, rows, records, cells):
    return {
        "schema": "induction_equality_tensor_final_ood_v2_role",
        "role": role, "rows": rows, "records": records,
        "copy_cells": base.serialize_copy_cells(cells),
    }


def _validate_payload(path: Path, expected: Mapping[str, Any], entry: Mapping[str, Any]):
    before = file_sha256(path)
    if before != entry["file_sha256"]:
        raise RuntimeError("v2 role file hash changed")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before or not base._semantic_equal(loaded, expected) \
            or natural.tensor_sha256(loaded["rows"]) != entry["rows_tensor_sha256"] \
            or natural.tensor_sha256(loaded["copy_cells"]["positive"]) != entry["positive_sha256"] \
            or natural.tensor_sha256(loaded["copy_cells"]["matched_negative"]) != entry["matched_sha256"]:
        raise RuntimeError("v2 role semantic replay failed")


def scored_support_census(
    masks: Mapping[str, torch.Tensor], *, min_tokens: int = 200, min_documents: int = 30,
) -> dict[str, dict[str, int]]:
    """Bind and gate every scorer-exposed cell, including collateral and all positions."""

    if tuple(masks) != ("positive", "matched_negative", "off_target", "all"):
        raise ValueError("v2 support masks must exactly follow the scorer cell order")
    shape = masks["positive"].shape
    if any(
        not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.device.type != "cpu"
        or mask.shape != shape for mask in masks.values()
    ) or len(shape) != 2:
        raise ValueError("v2 support masks must share one CPU boolean matrix currency")
    if type(min_tokens) is not int or type(min_documents) is not int or (
        min_tokens <= 0 or min_documents <= 0
    ):
        raise ValueError("v2 support thresholds must be positive integers")
    output = {
        name: {
            "tokens": int(mask.sum()),
            "documents": int(mask.any(1).sum()),
        }
        for name, mask in masks.items()
    }
    if any(
        value["tokens"] < min_tokens or value["documents"] < min_documents
        for value in output.values()
    ):
        raise RuntimeError("fresh v2 role support is below preregistered minimum")
    return output


def write_receipt_create_only(
    payload: Mapping[str, Any], path: Path, *, pre_link_check: Callable[[], None],
) -> None:
    """Publish a semantically replayed receipt; the hard link is the terminal action."""

    normalized = base.json_normalize(payload)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as sink:
            descriptor = None
            sink.write(json.dumps(normalized, indent=2, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        replay = json.loads(temporary.read_bytes())
        if replay != normalized or base.json_normalize(replay) != normalized:
            raise RuntimeError("v2 row receipt temporary semantic replay failed")
        pre_link_check()
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


def failure_is_still_publishable() -> bool:
    return not RECEIPT.exists() and not FAILURE.exists()


def freeze() -> dict[str, Any]:
    claim = natural.acquire_claim(LOCK)
    try:
        if any(path.exists() for path in (CACHE, RECEIPT, FAILURE)):
            raise RuntimeError("v2 row namespace is spent")
        if file_sha256(DISCOVERY) != DISCOVERY_SHA256 or file_sha256(V1_AUDIT) != V1_AUDIT_SHA256:
            raise RuntimeError("fixed candidate or preserved NO-GO audit changed")
        commit, sources, audit = audited_source_binding()
        registry_files = natural.discover_registry_files()
        prior, registry_hashes = metadata_registry_snapshot(registry_files)
        canonical, parquet = natural.BASE.validate_ordered_source()
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        frozen_source_identity = base.source_identity(
            canonical, parquet, registry_hashes, encoding,
        )
        prior_normalized = set(prior["normalized"]) | recover_prior_normalized_hashes(
            prior["code_sources_missing_normalized"],
        )
        if not prior_normalized:
            raise RuntimeError("historical normalized-code exclusion is unexpectedly empty")
        empty_rows: set[tuple[int, ...]] = set()
        natural_rows, natural_records = natural.harvest_fresh_documents(
            natural.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary,
            (prior["documents"], prior["indices"], empty_rows, empty_rows),
            start_document_index=START_DOCUMENT_INDEX, n_source_documents=2 * N,
            token_length=contract.ROW_WIDTH,
        )
        role_rows = {"label_fit": natural_rows[:N].contiguous(), "final_natural": natural_rows[N:].contiguous()}
        role_records = {"label_fit": natural_records[:N], "final_natural": natural_records[N:]}
        natural_full = {tuple(int(x) for x in row) for row in natural_rows.tolist()}
        code_rows, code_records = base.allocate_code_rows(
            base.ordered_code_blobs(commit), encoding.encode_ordinary,
            (set(prior["documents"]), set(prior["indices"]), natural_full,
             {row[:natural.PREFIX_LENGTH] for row in natural_full}),
            set(prior["code_paths"]), excluded_normalized=prior_normalized, n_rows=N,
        )
        for record in code_records:
            record["role"] = "ood_code"
        role_rows["ood_code"], role_records["ood_code"] = code_rows, code_records
        frequencies = contract.FitTokenFrequencies.from_rows(role_rows["label_fit"], vocab_size=50257)
        cells = {}
        for role in ROLES:
            ids = tuple(str(record.get("document_id") or f"code:{record['path']}") for record in role_records[role])
            cells[role] = contract.build_copy_cells(role_rows[role], frequencies, ids)
        support = {}
        for role in ("final_natural", "ood_code"):
            all_scored = torch.zeros_like(cells[role].positive)
            all_scored[:, contract.SCORE_START:contract.SCORE_STOP] = True
            support[role] = scored_support_census({
                "positive": cells[role].positive,
                "matched_negative": cells[role].matched_negative,
                "off_target": cells[role].off_target,
                "all": all_scored,
            })
        _registry_replay(registry_files, prior, registry_hashes)
        natural.require_claim(claim, LOCK)
        staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
        staging.mkdir(parents=False, exist_ok=False)
        payloads, entries = {}, {}
        try:
            for role in ROLES:
                payloads[role] = _payload(role, role_rows[role], role_records[role], cells[role])
                target = staging / f"{role}.pt"
                torch.save(payloads[role], target)
                with target.open("rb") as source:
                    os.fsync(source.fileno())
                entries[role] = {
                    "path": str((CACHE / target.name).resolve()), "file_sha256": file_sha256(target),
                    "rows_tensor_sha256": natural.tensor_sha256(role_rows[role]),
                    "positive_sha256": natural.tensor_sha256(cells[role].positive),
                    "matched_sha256": natural.tensor_sha256(cells[role].matched_negative),
                }
                _validate_payload(target, payloads[role], entries[role])
            natural.require_claim(claim, LOCK)
            os.replace(staging, CACHE)
            base._fsync_directory(CACHE.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        for role in ROLES:
            _validate_payload(Path(entries[role]["path"]), payloads[role], entries[role])
        receipt = base.json_normalize({
            "schema": "induction_equality_tensor_final_ood_v2_rows_receipt",
            "status": "frozen_before_any_v2_model_forward", "source_commit": commit,
            "source_hashes": sources, "audit": audit, "entries": entries,
            "roles": {"label_fit": "label_construction_only_no_model", "final_natural": "one_shot_final", "ood_code": "one_shot_code_ood"},
            "candidate_parent_sha256": DISCOVERY_SHA256, "preserved_v1_no_go_audit_sha256": V1_AUDIT_SHA256,
            "metadata_registry_hashes": registry_hashes,
            "metadata_exclusion_counts": {key: len(value) for key, value in prior.items()},
            "prior_normalized_code_hashes_sha256": hashlib.sha256(json.dumps(
                sorted(prior_normalized), separators=(",", ":"),
            ).encode()).hexdigest(),
            "source_identity": frozen_source_identity,
            "support_census": support,
            "old_v1_role_tensors_deserialized": False,
            "outcome_access": False,
        })
        def guard():
            _registry_replay(registry_files, prior, registry_hashes)
            if base.source_identity(canonical, parquet, registry_hashes, encoding) != (
                frozen_source_identity
            ) or hashlib.sha256(json.dumps(
                sorted(set(prior["normalized"]) | recover_prior_normalized_hashes(
                    prior["code_sources_missing_normalized"],
                )), separators=(",", ":"),
            ).encode()).hexdigest() != receipt["prior_normalized_code_hashes_sha256"]:
                raise RuntimeError("source identity or historical code exclusion changed")
            source_closure(commit); validate_audit(commit, sources)
            for role in ROLES:
                _validate_payload(Path(entries[role]["path"]), payloads[role], entries[role])
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("v2 row terminal appeared")
            natural.require_claim(claim, LOCK)
        write_receipt_create_only(receipt, RECEIPT, pre_link_check=guard)
        return receipt
    except BaseException as error:
        if failure_is_still_publishable():
            failure = {"schema": "induction_equality_tensor_final_ood_v2_rows_failure", "status": "terminal_failure_no_receipt", "error_type": type(error).__name__, "error": str(error), "cache_exists": CACHE.exists(), "outcome_access": False}
            try:
                def failure_guard():
                    if RECEIPT.exists() or FAILURE.exists():
                        raise RuntimeError("v2 row rival terminal appeared")
                    natural.require_claim(claim, LOCK)
                natural.write_json_create_only(failure, FAILURE, pre_link_check=failure_guard)
            except BaseException:
                pass
        raise
    finally:
        natural.release_claim(claim, LOCK)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
