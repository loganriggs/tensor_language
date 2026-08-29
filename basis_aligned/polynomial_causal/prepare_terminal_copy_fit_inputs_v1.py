#!/usr/bin/env python3
"""Create-only, outcome-blind projection of terminal-copy fit inputs.

The parent container includes fields that the fit collector is forbidden to consume.
This transaction, run before any E4 fit model forward, publishes only CPU long
``tokens[192,256]`` and 192 ordered document IDs.  It explicitly binds the disclosed
pre-authority engineering exposure in the accompanying erratum.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Mapping, NamedTuple

import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PARENT_RECEIPT = BQ / "terminal_copy_induction_v2_rows_receipt.json"
PARENT_RECEIPT_SHA256 = "aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8"
ERRATUM = HERE / "TERMINAL_COPY_FIT_INPUT_EXPOSURE_ERRATUM.md"
RUNNER = Path(__file__).resolve()
TEST = HERE / "test_prepare_terminal_copy_fit_inputs_v1.py"
AUDIT = HERE / "terminal_copy_fit_inputs_v1_independent_audit.json"
AUTHORITY = BQ / "terminal_copy_fit_inputs_v1_authority.json"
CACHE = BQ / ".rowcache_terminal_copy_fit_inputs_v1"
INPUTS = CACHE / "fit_inputs.pt"
MANIFEST = BQ / "terminal_copy_fit_inputs_v1_manifest.json"
RECEIPT = BQ / "terminal_copy_fit_inputs_v1_receipt.json"
FAILURE = BQ / "terminal_copy_fit_inputs_v1_failure.json"
LOCK = Path("/workspace/runs/.terminal_copy_fit_inputs_v1.lock")

SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/TERMINAL_COPY_FIT_INPUT_EXPOSURE_ERRATUM.md",
    "basis_aligned/polynomial_causal/prepare_terminal_copy_fit_inputs_v1.py",
    "basis_aligned/polynomial_causal/test_prepare_terminal_copy_fit_inputs_v1.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def document_digest(document_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for document_id in document_ids:
        encoded = document_id.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def stable_json(path: Path) -> dict[str, Any]:
    before = file_sha256(path)
    raw = path.read_bytes()
    if file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"JSON changed while reading: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON is not an object: {path}")
    return value


def source_closure() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"fit-input source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def verify_source(binding: Mapping[str, Any]) -> None:
    body = {"commit": binding.get("commit"), "paths": binding.get("paths")}
    if set(binding) != {"commit", "paths", "sha256"} or logical_sha256(body) != binding.get(
        "sha256"
    ) or set(body["paths"]) != set(SOURCE_PATHS):
        raise RuntimeError("fit-input source closure is malformed")
    for relative, digest in body["paths"].items():
        blob = subprocess.check_output(["git", "show", f"{body['commit']}:{relative}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != digest or file_sha256(ROOT / relative) != digest:
            raise RuntimeError("fit-input source closure drift")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(body["commit"]), "origin/main"],
        cwd=ROOT, check=True,
    )


def parent_binding() -> dict[str, Any]:
    if file_sha256(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256:
        raise RuntimeError("terminal-copy parent receipt changed")
    receipt = stable_json(PARENT_RECEIPT)
    entry = receipt.get("entries", {}).get("fit_natural", {})
    license_ = receipt.get("role_licenses", {}).get("fit_natural", {})
    if (
        receipt.get("status") != "frozen_before_any_terminal_copy_model_forward"
        or receipt.get("summary", {}).get("roles", {}).get("fit_natural") != 192
        or license_.get("authorized_use") != "fit_per_position_head_write_means_only"
        or not isinstance(entry.get("path"), str)
        or not isinstance(entry.get("file_sha256"), str)
        or not isinstance(entry.get("rows_tensor_sha256"), str)
    ):
        raise RuntimeError("terminal-copy parent fit license changed")
    return {
        "receipt_path": str(PARENT_RECEIPT),
        "receipt_sha256": PARENT_RECEIPT_SHA256,
        "container_path": entry["path"],
        "container_sha256": entry["file_sha256"],
        "rows_tensor_sha256": entry["rows_tensor_sha256"],
        "authorized_use": license_["authorized_use"],
    }


def validate_audit(path: Path = AUDIT) -> dict[str, Any]:
    if path.resolve() != AUDIT.resolve():
        raise RuntimeError("fit-input audit path changed")
    value = stable_json(path)
    reviewed = {
        str(RUNNER.relative_to(ROOT)): file_sha256(RUNNER),
        str(TEST.relative_to(ROOT)): file_sha256(TEST),
        str(ERRATUM.relative_to(ROOT)): file_sha256(ERRATUM),
    }
    if (
        set(value) != {
            "schema", "status", "approved", "outcome_access", "reviewer",
            "reviewed_source_sha256s", "focused_tests", "remaining_blockers",
        }
        or
        value.get("schema") != "terminal_copy_fit_inputs_v1_independent_audit"
        or value.get("status") != "approved_outcome_blind_projection"
        or value.get("approved") is not True
        or value.get("outcome_access") is not False
        or value.get("reviewer") != "independent_artifact_audit_agent"
        or value.get("reviewed_source_sha256s") != reviewed
        or not isinstance(value.get("focused_tests"), Mapping)
        or value["focused_tests"].get("passed") is not True
        or type(value["focused_tests"].get("count")) is not int
        or value["focused_tests"]["count"] <= 0
        or not isinstance(value.get("remaining_blockers"), list)
    ):
        raise RuntimeError("fit-input audit semantics or reviewed bytes changed")
    return value


class Claim(NamedTuple):
    descriptor: int
    inode: int
    nonce: str


def acquire_claim(path: Path = LOCK) -> Claim:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (nonce + "\n").encode())
        os.fsync(descriptor)
        return Claim(descriptor, os.fstat(descriptor).st_ino, nonce)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def require_claim(claim: Claim, path: Path = LOCK) -> None:
    if (
        not path.is_file() or path.stat().st_ino != claim.inode
        or path.read_text() != claim.nonce + "\n"
    ):
        raise RuntimeError("fit-input lock ownership changed")


def release_claim(claim: Claim, path: Path = LOCK) -> None:
    try:
        if path.exists() and path.stat().st_ino == claim.inode:
            path.unlink()
    finally:
        os.close(claim.descriptor)


def create_only_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        with temporary.open("xb") as sink:
            sink.write(data)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    create_only_bytes(path, (json.dumps(
        dict(value), sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode())


def create_only_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        with temporary.open("xb") as sink:
            torch.save(dict(value), sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_projection(path: Path, authority_sha256: str) -> dict[str, Any]:
    before = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before or not isinstance(payload, dict) or set(payload) != {
        "schema", "authority_sha256", "tokens", "ordered_document_ids",
        "tokens_sha256", "ordered_document_ids_sha256",
    }:
        raise RuntimeError("fit-input projection schema changed")
    tokens = payload["tokens"]
    documents = payload["ordered_document_ids"]
    if (
        payload["schema"] != "terminal_copy_fit_inputs_v1_payload"
        or payload["authority_sha256"] != authority_sha256
        or not torch.is_tensor(tokens) or tokens.dtype != torch.long
        or tokens.device.type != "cpu" or tuple(tokens.shape) != (192, 256)
        or not isinstance(documents, tuple) or len(documents) != 192
        or len(set(documents)) != 192 or any(not isinstance(x, str) or not x for x in documents)
        or tensor_sha256(tokens) != payload["tokens_sha256"]
        or document_digest(documents) != payload["ordered_document_ids_sha256"]
    ):
        raise RuntimeError("fit-input projection tensor semantics changed")
    return payload


def execute() -> dict[str, Any]:
    outputs = (AUTHORITY, INPUTS, MANIFEST, RECEIPT, FAILURE, LOCK)
    spent = [str(path) for path in outputs if path.exists()]
    if spent:
        raise RuntimeError(f"fit-input namespace is spent: {spent}")
    source = source_closure()
    parent = parent_binding()
    audit = validate_audit()
    body = {
        "schema": "terminal_copy_fit_inputs_v1_authority",
        "status": "frozen_before_any_E4_fit_model_forward_after_disclosed_parent_container_access",
        "source_closure": source,
        "parent_binding": parent,
        "erratum_path": str(ERRATUM),
        "erratum_sha256": file_sha256(ERRATUM),
        "audit_path": str(AUDIT),
        "audit_sha256": file_sha256(AUDIT),
        "outputs": {"authority": str(AUTHORITY), "inputs": str(INPUTS),
                    "manifest": str(MANIFEST), "receipt": str(RECEIPT),
                    "failure": str(FAILURE), "lock": str(LOCK)},
        "model_forward_calls_before_authority": 0,
        "authorized_for_fit_input_projection": True,
        "authorized_for_model_forward": False,
        "authorized_for_candidate_selection": False,
    }
    authority = {**body, "authority_sha256": logical_sha256(body)}
    create_only_json(AUTHORITY, authority)
    claim = acquire_claim()
    try:
        require_claim(claim)
        container = Path(parent["container_path"])
        before = file_sha256(container)
        if before != parent["container_sha256"]:
            raise RuntimeError("parent fit container changed before projection")
        payload = torch.load(container, map_location="cpu", weights_only=True)
        if file_sha256(container) != before or not isinstance(payload, dict):
            raise RuntimeError("parent fit container changed during projection")
        rows = payload.get("rows")
        records = payload.get("records")
        if (
            not torch.is_tensor(rows) or rows.dtype != torch.long
            or rows.device.type != "cpu" or tuple(rows.shape) != (192, 257)
            or tensor_sha256(rows) != parent["rows_tensor_sha256"]
            or not isinstance(records, list) or len(records) != 192
        ):
            raise RuntimeError("parent fit rows changed")
        documents = tuple(record.get("document_id") for record in records)
        if any(
            not isinstance(record, Mapping)
            or record.get("role") != "fit_natural"
            or record.get("role_row_index") != index
            for index, record in enumerate(records)
        ) or len(set(documents)) != 192 or any(not isinstance(x, str) or not x for x in documents):
            raise RuntimeError("parent fit document order changed")
        tokens = rows[:, :256].clone().contiguous()
        del payload, rows, records
        projected = {
            "schema": "terminal_copy_fit_inputs_v1_payload",
            "authority_sha256": authority["authority_sha256"],
            "tokens": tokens,
            "ordered_document_ids": documents,
            "tokens_sha256": tensor_sha256(tokens),
            "ordered_document_ids_sha256": document_digest(documents),
        }
        create_only_torch(INPUTS, projected)
        replay = validate_projection(INPUTS, authority["authority_sha256"])
        manifest = {
            "schema": "terminal_copy_fit_inputs_v1_manifest",
            "authority_sha256": authority["authority_sha256"],
            "inputs_file_sha256": file_sha256(INPUTS),
            "tokens_sha256": replay["tokens_sha256"],
            "ordered_document_ids_sha256": replay["ordered_document_ids_sha256"],
            "exposed_payload_fields": sorted(replay),
            "forbidden_fields_absent": True,
        }
        create_only_json(MANIFEST, manifest)
        require_claim(claim)
        verify_source(source)
        if (
            parent_binding() != parent or validate_audit() != audit
            or stable_json(AUTHORITY) != authority
            or stable_json(MANIFEST) != manifest
            or FAILURE.exists() or RECEIPT.exists()
            or validate_projection(INPUTS, authority["authority_sha256"])["tokens_sha256"]
            != replay["tokens_sha256"]
            or file_sha256(Path(parent["container_path"])) != parent["container_sha256"]
        ):
            raise RuntimeError("fit-input terminal recheck failed")
        receipt = {
            "schema": "terminal_copy_fit_inputs_v1_receipt",
            "status": "complete_receipt_last_input_only_no_model_access",
            "authority_file_sha256": file_sha256(AUTHORITY),
            "authority_sha256": authority["authority_sha256"],
            "inputs_path": str(INPUTS),
            "inputs_file_sha256": file_sha256(INPUTS),
            "manifest_file_sha256": file_sha256(MANIFEST),
            "parent_row_receipt_sha256": PARENT_RECEIPT_SHA256,
            "erratum_sha256": file_sha256(ERRATUM),
            "tokens_shape": [192, 256],
            "tokens_sha256": replay["tokens_sha256"],
            "ordered_document_ids_sha256": replay["ordered_document_ids_sha256"],
            "payload_fields": sorted(replay),
            "label_copy_cell_synthetic_fields_absent": True,
            "model_imported": False,
            "checkpoint_loaded": False,
            "model_forward_calls": 0,
            "scientific_outcomes_read": False,
            "authorized_for_fit_mean_input_only": True,
            "authorized_for_candidate_selection": False,
        }
        create_only_json(RECEIPT, receipt)
        return receipt
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            create_only_json(FAILURE, {
                "schema": "terminal_copy_fit_inputs_v1_failure",
                "status": "terminal_failure_no_success_receipt",
                "authority_sha256": authority["authority_sha256"],
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "receipt_exists": False,
            })
        raise
    finally:
        release_claim(claim)


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
