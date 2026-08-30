#!/usr/bin/env python3
"""Receipt-last, model-free publisher for bracket FIT/SELECT/OOD rows.

The module cannot mint its source authority or independent audit. Import is I/O-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Any, Mapping

import torch

import bracket_closure_rows_v1 as contract
from bracket_closure_masks_v1 import (
    BracketDomain, DelimiterFamily, DelimiterRegistry, build_bracket_masks,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_CLOSURE = (
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_CANARY_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_ROWS_V1_AMENDMENT.md",
    "basis_aligned/polynomial_causal/bracket_closure_masks_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_rows_v1.py",
    "basis_aligned/polynomial_causal/freeze_bracket_closure_rows_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_masks_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_rows_v1.py",
    "basis_aligned/polynomial_causal/test_freeze_bracket_closure_rows_v1.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_ok(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def source_closure(commit: str) -> dict[str, str]:
    if not _sha_ok(commit, 40):
        raise RuntimeError("row authority source commit is malformed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    output = {}
    for relative in SOURCE_CLOSURE:
        path = ROOT / relative
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"bracket row source drift: {relative}")
        output[relative] = digest
    return output


def _read_stable_json(path: Path) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    if file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"JSON changed during read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value, before


def validate_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "schema", "source_commit", "source_hashes", "candidate_path",
        "candidate_sha256", "candidate_source_identity_sha256",
        "delimiter_registry_sha256", "historical_registries", "allocation_seed",
        "cache_path", "receipt_path", "failure_path", "lock_path", "outcome_access",
    }
    if set(payload) != keys or payload.get("schema") != "bracket_closure_rows_v1_authority" \
            or payload.get("outcome_access") is not False:
        raise RuntimeError("bracket row authority schema changed")
    if not _sha_ok(payload.get("source_commit"), 40) or payload.get("source_hashes") != (
        source_closure(payload["source_commit"])
    ):
        raise RuntimeError("bracket row authority source binding changed")
    for key in (
        "candidate_sha256", "candidate_source_identity_sha256",
        "delimiter_registry_sha256",
    ):
        if not _sha_ok(payload.get(key)):
            raise RuntimeError(f"authority {key} is malformed")
    registries = payload.get("historical_registries")
    if not isinstance(registries, list) or not registries or any(
        not isinstance(item, dict) or set(item) != {"path", "sha256"}
        or not isinstance(item["path"], str) or not _sha_ok(item["sha256"])
        for item in registries
    ):
        raise RuntimeError("authority historical registry is malformed")
    if not isinstance(payload.get("allocation_seed"), str) or not payload["allocation_seed"]:
        raise RuntimeError("authority allocation seed is malformed")
    paths = [payload[key] for key in (
        "candidate_path", "cache_path", "receipt_path", "failure_path", "lock_path",
    )]
    if any(not isinstance(value, str) or not Path(value).is_absolute() for value in paths) \
            or len(set(paths)) != len(paths):
        raise RuntimeError("authority namespace paths are malformed")
    return dict(payload)


def validate_independent_audit(
    payload: Mapping[str, Any], *, authority_sha256: str, authority: Mapping[str, Any],
) -> None:
    if set(payload) != {
        "schema", "status", "outcome_access", "authority_sha256",
        "audited_source_commit", "audited_source_hashes", "tests_passed", "reviewer",
    } or payload.get("schema") != "bracket_closure_rows_v1_independent_audit" \
            or payload.get("status") != "GO" or payload.get("outcome_access") is not False \
            or payload.get("authority_sha256") != authority_sha256 \
            or payload.get("audited_source_commit") != authority["source_commit"] \
            or payload.get("audited_source_hashes") != authority["source_hashes"] \
            or type(payload.get("tests_passed")) is not int or payload["tests_passed"] < 1 \
            or not isinstance(payload.get("reviewer"), str) or not payload["reviewer"]:
        raise RuntimeError("independent bracket row audit is not an exact source-bound GO")


def _registry(payload: Mapping[str, Any]) -> DelimiterRegistry:
    if set(payload) != {"families", "quote_control_ids", "punctuation_control_ids"}:
        raise RuntimeError("candidate delimiter registry schema changed")
    families = tuple(DelimiterFamily(
        item["name"], tuple(item["opener_ids"]), tuple(item["closer_ids"]),
    ) for item in payload["families"])
    return DelimiterRegistry(
        families, tuple(payload["quote_control_ids"]), tuple(payload["punctuation_control_ids"]),
    )


def _load_candidates(path: Path, expected_sha: str):
    before = file_sha256(path)
    if before != expected_sha:
        raise RuntimeError("candidate bundle hash changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before or not isinstance(value, dict) or set(value) != {
        "schema", "rows", "records", "delimiter_registry", "source_identity",
    } or value["schema"] != "bracket_closure_rows_v1_candidates":
        raise RuntimeError("candidate bundle schema changed")
    rows = value["rows"]
    records = tuple(contract.CandidateRecord(
        item["document_id"], item["source_document_index"], item["source_file"],
        item["source_revision"], item["source_blob_sha256"], BracketDomain(item["domain"]),
        item["license_id"], item.get("normalized_python_sha256"),
    ) for item in value["records"])
    registry = _registry(value["delimiter_registry"])
    required_source_identity = {
        "tokenizer_name", "tokenizer_sha256", "prose_source", "prose_revision",
        "prose_blob_sha256", "prose_license", "code_repository", "code_commit",
        "code_license", "builder_source_commit", "builder_source_hashes",
    }
    if not isinstance(value["source_identity"], dict) or set(
        value["source_identity"]
    ) != required_source_identity or any(
        not isinstance(item, (str, dict)) or not item
        for item in value["source_identity"].values()
    ):
        raise RuntimeError("candidate source/tokenizer/license identity is incomplete")
    source_identity_sha = hashlib.sha256(json.dumps(
        value["source_identity"], sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return value, rows, records, registry, source_identity_sha


def _record_dict(record: contract.CandidateRecord) -> dict[str, Any]:
    return {
        "document_id": record.document_id,
        "source_document_index": record.source_document_index,
        "source_file": record.source_file,
        "source_revision": record.source_revision,
        "source_blob_sha256": record.source_blob_sha256,
        "domain": record.domain.value,
        "license_id": record.license_id,
        "normalized_python_sha256": record.normalized_python_sha256,
    }


def _role_payload(role: contract.FrozenRole) -> dict[str, Any]:
    masks = role.masks
    return {
        "schema": "bracket_closure_rows_v1_role", "role": role.role.value,
        "rows": role.rows, "records": [_record_dict(record) for record in role.records],
        "masks": {**dict(masks.named_cells()), "family_index": masks.family_index,
                  "depth": masks.depth, "distance": masks.distance,
                  "domain_index": masks.domain_index},
        "support": {key: dict(value) for key, value in role.support.items()},
    }


def _payload_summary(
    path: Path, expected_hash: str, registry: DelimiterRegistry,
) -> dict[str, Any]:
    before = file_sha256(path)
    if before != expected_hash:
        raise RuntimeError("installed bracket role hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before or not isinstance(payload, dict) or payload.get(
        "schema"
    ) != "bracket_closure_rows_v1_role" or payload.get("role") not in {
        role.value for role in contract.RowRole
    }:
        raise RuntimeError("installed bracket role semantic replay failed")
    if set(payload) != {"schema", "role", "rows", "records", "masks", "support"}:
        raise RuntimeError("installed bracket role schema has extra/missing fields")
    rows = payload.get("rows")
    if not torch.is_tensor(rows) or rows.dtype != torch.long or rows.shape != (320, 257):
        raise RuntimeError("installed bracket role row currency changed")
    records = tuple(contract.CandidateRecord(
        item["document_id"], item["source_document_index"], item["source_file"],
        item["source_revision"], item["source_blob_sha256"], BracketDomain(item["domain"]),
        item["license_id"], item.get("normalized_python_sha256"),
    ) for item in payload["records"])
    if len(records) != 320:
        raise RuntimeError("installed bracket provenance count changed")
    masks = build_bracket_masks(
        rows, registry, tuple(record.domain for record in records),
        first_prediction=contract.SCORE_START,
    )
    expected_masks = {
        **dict(masks.named_cells()), "family_index": masks.family_index,
        "depth": masks.depth, "distance": masks.distance, "domain_index": masks.domain_index,
    }
    if set(payload["masks"]) != set(expected_masks) or any(
        not torch.equal(payload["masks"][name], value)
        for name, value in expected_masks.items()
    ):
        raise RuntimeError("installed bracket masks do not replay from rows")
    support = {domain.value: contract.support_census(masks, domain) for domain in BracketDomain}
    if payload["support"] != support:
        raise RuntimeError("installed bracket support does not replay from masks")
    records_sha = hashlib.sha256(json.dumps(
        payload["records"], sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "file_sha256": before, "rows_sha256": contract.tensor_sha256(rows),
        "records_sha256": records_sha,
    }


def _claim(path: Path) -> tuple[int, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(16)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, nonce.encode()); os.fsync(descriptor); os.close(descriptor)
    return path.stat().st_dev, path.stat().st_ino, nonce


def _require_claim(path: Path, claim: tuple[int, int, str]) -> None:
    stat = path.stat()
    if (stat.st_dev, stat.st_ino, path.read_text()) != claim:
        raise RuntimeError("bracket row lock ownership changed")


def _publish_json_last(payload: Mapping[str, Any], path: Path, guard) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as sink:
            sink.write(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n")
            sink.flush(); os.fsync(sink.fileno())
        if json.loads(temporary.read_bytes()) != payload:
            raise RuntimeError("temporary receipt semantic replay failed")
        guard(); os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(authority_path: Path, audit_path: Path) -> dict[str, Any]:
    authority_raw, authority_sha = _read_stable_json(authority_path)
    authority = validate_authority(authority_raw)
    audit, _ = _read_stable_json(audit_path)
    validate_independent_audit(audit, authority_sha256=authority_sha, authority=authority)
    cache, receipt, failure, lock = (Path(authority[key]) for key in (
        "cache_path", "receipt_path", "failure_path", "lock_path",
    ))
    claim = _claim(lock)
    try:
        if any(path.exists() for path in (cache, receipt, failure)):
            raise RuntimeError("bracket row namespace is already spent")
        candidate_path = Path(authority["candidate_path"])
        _, rows, records, registry, source_identity_sha = _load_candidates(
            candidate_path, authority["candidate_sha256"],
        )
        if source_identity_sha != authority["candidate_source_identity_sha256"] or (
            contract.registry_sha256(registry) != authority["delimiter_registry_sha256"]
        ):
            raise RuntimeError("candidate source/registry identity changed")
        historical_payloads, registry_hashes = [], {}
        for item in authority["historical_registries"]:
            path = Path(item["path"]); payload, observed = _read_stable_json(path)
            if observed != item["sha256"]:
                raise RuntimeError("historical registry hash changed")
            historical_payloads.append(payload); registry_hashes[str(path)] = observed
        prior = contract.historical_exclusions(tuple(historical_payloads))
        if not prior.documents or not (prior.row_sha256 or prior.prefix32_sha256) or (
            not prior.source_files or not prior.normalized_python
        ):
            raise RuntimeError("historical document/row/code exclusion census is empty")
        roles = contract.allocate_roles(
            rows, records, registry, prior, seed=authority["allocation_seed"],
        )
        staging = cache.with_name(f".{cache.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
        staging.mkdir(parents=True, exist_ok=False)
        entries = {}
        try:
            for role in roles:
                path = staging / f"{role.role.value}.pt"
                with path.open("xb") as sink:
                    torch.save(_role_payload(role), sink); sink.flush(); os.fsync(sink.fileno())
                digest = file_sha256(path)
                entries[role.role.value] = {
                    "filename": path.name, **_payload_summary(path, digest, registry),
                }
            _require_claim(lock, claim)
            cache.mkdir(parents=False, exist_ok=False)
            for entry in entries.values():
                os.link(staging / entry["filename"], cache / entry["filename"])
            directory = os.open(cache, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        for entry in entries.values():
            _payload_summary(cache / entry["filename"], entry["file_sha256"], registry)
        result = {
            "schema": "bracket_closure_rows_v1_receipt",
            "status": "frozen_before_any_model_forward_receipt_last",
            "authority_sha256": authority_sha, "audit_sha256": file_sha256(audit_path),
            "source_commit": authority["source_commit"],
            "source_hashes": authority["source_hashes"],
            "candidate_sha256": authority["candidate_sha256"],
            "candidate_source_identity_sha256": source_identity_sha,
            "delimiter_registry_sha256": contract.registry_sha256(registry),
            "historical_registry_hashes": registry_hashes,
            "historical_exclusion_counts": {
                name: len(getattr(prior, name)) for name in (
                    "documents", "source_files", "source_blobs", "normalized_python",
                    "row_sha256", "prefix32_sha256",
                )
            },
            "entries": entries, "outcome_access": False,
        }
        def guard():
            current_authority, current_sha = _read_stable_json(authority_path)
            if current_sha != authority_sha or validate_authority(current_authority) != authority:
                raise RuntimeError("authority changed before receipt")
            current_audit, _ = _read_stable_json(audit_path)
            validate_independent_audit(
                current_audit, authority_sha256=authority_sha, authority=authority,
            )
            _load_candidates(candidate_path, authority["candidate_sha256"])
            for item in authority["historical_registries"]:
                if file_sha256(Path(item["path"])) != item["sha256"]:
                    raise RuntimeError("history changed before receipt")
            for entry in entries.values():
                _payload_summary(cache / entry["filename"], entry["file_sha256"], registry)
            if receipt.exists() or failure.exists():
                raise RuntimeError("rival bracket row terminal appeared")
            _require_claim(lock, claim)
        _publish_json_last(result, receipt, guard)
        return result
    except BaseException as error:
        if not receipt.exists() and not failure.exists():
            try:
                _publish_json_last({
                    "schema": "bracket_closure_rows_v1_failure",
                    "status": "terminal_failure_no_success_receipt",
                    "error_type": type(error).__name__, "error": str(error),
                    "outcome_access": False,
                }, failure, lambda: _require_claim(lock, claim))
            except BaseException:
                pass
        raise
    finally:
        try:
            _require_claim(lock, claim); lock.unlink()
        except (FileNotFoundError, RuntimeError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(freeze(arguments.authority, arguments.audit), indent=2))
