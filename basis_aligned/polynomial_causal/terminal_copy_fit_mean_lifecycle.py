"""Source-closed lifecycle for the outcome-blind E4 fit-head-mean transaction.

This module is deliberately incapable of selecting an attention candidate or reading
model logits.  It binds the already-published fit-row and physical-adapter receipts,
serializes the six sparse head means, reloads their tensor semantics, and publishes a
receipt last.  The numerical collection itself is owned by
``FitMeanCollectionOwner``; this file owns only authority and artifact integrity.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Mapping, NamedTuple, Sequence

import torch

import bilin18_observed_model_facade as facade
import prepare_terminal_copy_fit_inputs_v1 as fit_input_projection
from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher
from terminal_copy_fit_head_means import (
    FitHeadMeanAccumulator,
    FitHeadMeanBank,
    NAMED_HEADS_BY_LAYER,
    _document_digest,
)
from terminal_copy_fit_mean_owner import (
    FitMeanCollectionOwner,
    FitMeanOwnerClosure,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BQ = ROOT / "basis_aligned" / "bilinear_quotient"

ROW_RECEIPT = BQ / "terminal_copy_induction_v2_rows_receipt.json"
ROW_RECEIPT_SHA256 = "aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8"
FIT_INPUT_AUTHORITY = BQ / "terminal_copy_fit_inputs_v1_authority.json"
FIT_INPUT_RECEIPT = BQ / "terminal_copy_fit_inputs_v1_receipt.json"
FIT_INPUTS = BQ / ".rowcache_terminal_copy_fit_inputs_v1" / "fit_inputs.pt"
FIT_INPUT_MANIFEST = BQ / "terminal_copy_fit_inputs_v1_manifest.json"
FIT_INPUT_ERRATUM = HERE / "TERMINAL_COPY_FIT_INPUT_EXPOSURE_ERRATUM.md"
ADAPTER_RECEIPT = HERE / "terminal_copy_attention_checkpoint_check_v3_receipt.json"
ADAPTER_RECEIPT_SHA256 = "c5ef51670b6e23bb3cddbbef6c5cd451dff55eea8b8f7ddfdf20aca7374bb324"
ADAPTER_RESULT = HERE / "terminal_copy_attention_checkpoint_check_v3_result.json"
ADAPTER_RESULT_SHA256 = "084087dc4264d438d40c64203a60f870629611e125268f1be62e9f0601faec7d"

AUTHORITY = HERE / "terminal_copy_fit_means_v1_authority.json"
BANK = HERE / "terminal_copy_fit_means_v1_bank.pt"
RESULT = HERE / "terminal_copy_fit_means_v1_result.json"
MANIFEST = HERE / "terminal_copy_fit_means_v1_manifest.json"
RECEIPT = HERE / "terminal_copy_fit_means_v1_receipt.json"
FAILURE = HERE / "terminal_copy_fit_means_v1_failure.json"
LOCK = Path("/workspace/runs/.terminal_copy_fit_means_v1.lock")
AUDIT = HERE / "terminal_copy_fit_mean_lifecycle_v1_independent_audit.json"

PROTECTED_PATHS = (
    ROW_RECEIPT,
    FIT_INPUT_AUTHORITY,
    FIT_INPUT_RECEIPT,
    FIT_INPUTS,
    FIT_INPUT_MANIFEST,
    ADAPTER_RECEIPT,
    ADAPTER_RESULT,
    facade.DEFAULT_SNAPSHOT / "config.json",
    facade.DEFAULT_SNAPSHOT / "pytorch_model.bin",
)

SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/TERMINAL_COPY_FIT_HEAD_MEANS_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/TERMINAL_COPY_FIT_INPUT_EXPOSURE_ERRATUM.md",
    "basis_aligned/polynomial_causal/TERMINAL_COPY_INDUCTION_V1_SCREENING_AMENDMENT.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/terminal_copy_attention_adapter.py",
    "basis_aligned/polynomial_causal/terminal_copy_attention_dispatcher.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_head_means.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_owner.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_lifecycle.py",
    "basis_aligned/polynomial_causal/prepare_terminal_copy_fit_inputs_v1.py",
    "basis_aligned/polynomial_causal/terminal_copy_induction_v1.py",
    "basis_aligned/polynomial_causal/terminal_copy_streaming_statistics.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_attention_adapter.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_attention_dispatcher.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_fit_head_means.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_fit_mean_owner.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_fit_mean_lifecycle.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
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


def _stable_json(path: Path) -> dict[str, Any]:
    before = file_sha256(path)
    raw = path.read_bytes()
    if file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"JSON changed while reading: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON is not an object: {path}")
    return value


def output_namespace() -> tuple[Path, ...]:
    return AUTHORITY, BANK, RESULT, MANIFEST, RECEIPT, FAILURE, LOCK


def protected_snapshot(paths: Sequence[Path] = PROTECTED_PATHS) -> dict[str, str | None]:
    """Hash every immutable parent/model input before and after model execution."""

    return {
        str(path): file_sha256(path) if path.is_file() else None
        for path in paths
    }


class RunClaim(NamedTuple):
    descriptor: int
    inode: int
    nonce: str


def acquire_claim(path: Path | None = None) -> RunClaim:
    path = LOCK if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"fit-mean namespace is locked: {path}") from error
    try:
        os.write(descriptor, (nonce + "\n").encode())
        os.fsync(descriptor)
        return RunClaim(descriptor, os.fstat(descriptor).st_ino, nonce)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def require_claim(claim: RunClaim, path: Path | None = None) -> None:
    path = LOCK if path is None else path
    if (
        not isinstance(claim, RunClaim)
        or not path.is_file()
        or path.stat().st_ino != claim.inode
        or path.read_text() != claim.nonce + "\n"
    ):
        raise RuntimeError("fit-mean execution lock ownership changed")


def release_claim(claim: RunClaim, path: Path | None = None) -> None:
    path = LOCK if path is None else path
    try:
        if path.exists() and path.stat().st_ino == claim.inode:
            path.unlink()
    finally:
        os.close(claim.descriptor)


def require_pristine_namespace(paths: Sequence[Path] | None = None) -> None:
    selected = output_namespace() if paths is None else tuple(paths)
    spent = [str(path) for path in selected if path.exists()]
    if spent:
        raise RuntimeError(f"fit-mean output namespace is spent: {spent}")


def require_pristine_execution_namespace() -> None:
    if not AUTHORITY.is_file():
        raise RuntimeError("fit-mean execution authority is absent")
    spent = [str(path) for path in (BANK, RESULT, MANIFEST, RECEIPT, FAILURE, LOCK) if path.exists()]
    if spent:
        raise RuntimeError(f"fit-mean execution namespace is spent: {spent}")


def source_closure() -> dict[str, Any]:
    """Bind every executable source byte to one commit already on origin/main."""

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"fit-mean source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if not (ROOT / relative).is_file() or file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"live fit-mean source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def verify_source_closure(binding: Mapping[str, Any]) -> None:
    body = {"commit": binding.get("commit"), "paths": binding.get("paths")}
    if (
        set(binding) != {"commit", "paths", "sha256"}
        or not isinstance(body["paths"], Mapping)
        or set(body["paths"]) != set(SOURCE_PATHS)
        or logical_sha256(body) != binding.get("sha256")
    ):
        raise RuntimeError("fit-mean source closure is malformed")
    for relative, digest in body["paths"].items():
        completed = subprocess.run(
            ["git", "show", f"{body['commit']}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if (
            completed.returncode != 0
            or hashlib.sha256(completed.stdout).hexdigest() != digest
            or file_sha256(ROOT / relative) != digest
        ):
            raise RuntimeError(f"fit-mean source drift: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(body["commit"]), "origin/main"],
        cwd=ROOT, check=True,
    )


def row_binding() -> dict[str, Any]:
    """Bind the sanitized fit-input projection without opening its tensor."""

    if file_sha256(ROW_RECEIPT) != ROW_RECEIPT_SHA256 or not (
        FIT_INPUT_AUTHORITY.is_file() and FIT_INPUT_RECEIPT.is_file()
        and FIT_INPUT_MANIFEST.is_file() and FIT_INPUTS.is_file()
    ):
        raise RuntimeError("terminal-copy row receipt bytes changed")
    metadata = fit_input_projection.validate_published_metadata()
    receipt = metadata["receipt"]
    projection_authority = metadata["authority"]
    manifest = metadata["manifest"]
    projection_body = {
        key: value for key, value in projection_authority.items()
        if key != "authority_sha256"
    }
    payload_fields = {
        "schema", "authority_sha256", "tokens", "ordered_document_ids",
        "tokens_sha256", "ordered_document_ids_sha256",
    }
    if (
        receipt.get("status") != "complete_receipt_last_input_only_no_model_access"
        or receipt.get("parent_row_receipt_sha256") != ROW_RECEIPT_SHA256
        or receipt.get("erratum_sha256") != file_sha256(FIT_INPUT_ERRATUM)
        or receipt.get("authority_file_sha256") != file_sha256(FIT_INPUT_AUTHORITY)
        or receipt.get("authority_sha256") != projection_authority.get("authority_sha256")
        or logical_sha256(projection_body) != projection_authority.get("authority_sha256")
        or receipt.get("tokens_shape") != [192, 256]
        or set(receipt.get("payload_fields", ())) != payload_fields
        or receipt.get("label_copy_cell_synthetic_fields_absent") is not True
        or receipt.get("E4_fit_model_forward_calls") != 0
        or receipt.get("scientific_outcomes_read") is not False
        or receipt.get("authorized_for_fit_mean_input_only") is not True
        or receipt.get("authorized_for_candidate_selection") is not False
        or projection_authority.get("status")
        != "frozen_before_any_E4_fit_model_forward_after_disclosed_parent_container_access"
        or projection_authority.get("parent_binding", {}).get("receipt_sha256")
        != ROW_RECEIPT_SHA256
        or not isinstance(receipt.get("inputs_path"), str)
        or not Path(receipt["inputs_path"]).is_file()
        or file_sha256(Path(receipt["inputs_path"])) != receipt.get("inputs_file_sha256")
        or receipt.get("manifest_file_sha256") != file_sha256(FIT_INPUT_MANIFEST)
        or manifest.get("inputs_file_sha256") != receipt.get("inputs_file_sha256")
    ):
        raise RuntimeError("sanitized fit-input receipt semantics changed")
    body = {
        "parent_receipt_path": str(ROW_RECEIPT),
        "parent_receipt_sha256": ROW_RECEIPT_SHA256,
        "projection_authority_path": str(FIT_INPUT_AUTHORITY),
        "projection_authority_file_sha256": file_sha256(FIT_INPUT_AUTHORITY),
        "projection_authority_sha256": projection_authority["authority_sha256"],
        "projection_receipt_path": str(FIT_INPUT_RECEIPT),
        "projection_receipt_sha256": file_sha256(FIT_INPUT_RECEIPT),
        "role": "fit_natural",
        "row_count": 192,
        "model_input_width": 256,
        "input_path": receipt["inputs_path"],
        "input_file_sha256": receipt["inputs_file_sha256"],
        "tokens_sha256": receipt["tokens_sha256"],
        "ordered_document_ids_sha256": receipt["ordered_document_ids_sha256"],
        "payload_fields": sorted(payload_fields),
        "labels_copy_cells_synthetic_present": False,
        "authorized_use": "fit_per_position_head_write_means_only",
        "disclosed_parent_container_engineering_access": True,
    }
    return {**body, "sha256": logical_sha256(body)}


def adapter_binding() -> dict[str, Any]:
    """Bind the engineering-only physical adapter check and exact checkpoint."""

    if file_sha256(ADAPTER_RECEIPT) != ADAPTER_RECEIPT_SHA256 or file_sha256(
        ADAPTER_RESULT
    ) != ADAPTER_RESULT_SHA256:
        raise RuntimeError("terminal-copy adapter receipt/result bytes changed")
    receipt = _stable_json(ADAPTER_RECEIPT)
    result = _stable_json(ADAPTER_RESULT)
    rows = result.get("rows")
    if (
        receipt.get("status") != "receipt_last"
        or receipt.get("result_file_sha256") != ADAPTER_RESULT_SHA256
        or receipt.get("all_layers_passed") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or result.get("checkpoint_weights_sha256") != facade.WEIGHTS_SHA256
        or result.get("all_layers_passed") is not True
        or result.get("scientific_claim_authorized") is not False
        or not isinstance(rows, list)
        or [item.get("layer") for item in rows] != list(NAMED_LAYERS)
        or any(
            item.get("passed") is not True
            or item.get("native_full_bit_equal") is not True
            or item.get("bus_bit_equal") is not True
            for item in rows
        )
    ):
        raise RuntimeError("terminal-copy adapter receipt semantics changed")
    body = {
        "receipt_path": str(ADAPTER_RECEIPT),
        "receipt_sha256": ADAPTER_RECEIPT_SHA256,
        "result_path": str(ADAPTER_RESULT),
        "result_sha256": ADAPTER_RESULT_SHA256,
        "checkpoint_weights_sha256": facade.WEIGHTS_SHA256,
        "layers": list(NAMED_LAYERS),
        "native_full_bit_equal": True,
        "value_bus_bit_equal": True,
        "scientific_claim_authorized_by_parent": False,
    }
    return {**body, "sha256": logical_sha256(body)}


def protocol() -> dict[str, Any]:
    return {
        "role": "fit_natural",
        "documents": 192,
        "input_width": 256,
        "model_tokens": 256,
        "layers": list(NAMED_LAYERS),
        "heads_by_layer": {str(k): list(v) for k, v in NAMED_HEADS_BY_LAYER.items()},
        "native_attention_calls_per_document": 18,
        "native_mlp_calls_per_document": 18,
        "adapter_decomposition_layers": list(NAMED_LAYERS),
        "unembedding_calls": 0,
        "loss_or_logit_reads": 0,
        "label_or_copy_cell_reads": 0,
        "source_dtype": "torch.bfloat16",
        "accumulator_dtype": "torch.float64",
        "runtime_dtype": "torch.float32",
        "document_reduction": "one CPU float64 addition per receipt-ordered document",
        "artifact_order": ["authority", "bank", "result", "manifest", "receipt_last"],
        "authorized_for_fit_collection": True,
        "authorized_for_candidate_selection": False,
        "authorized_for_E4_evidence": False,
    }


def verified_draft_authority() -> dict[str, Any]:
    """Construct a checkable draft, while explicitly withholding execution authority."""

    require_pristine_namespace()
    checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
    body = {
        "schema": "terminal_copy_fit_means_v1_authority_draft",
        "status": "nonauthoritative_until_source_commit_and_independent_audit",
        "source_closure": source_closure(),
        "row_binding": row_binding(),
        "adapter_binding": adapter_binding(),
        "checkpoint": asdict(checkpoint),
        "protocol": protocol(),
        "outputs": {name: str(path) for name, path in {
            "authority": AUTHORITY, "bank": BANK, "result": RESULT,
            "manifest": MANIFEST, "receipt": RECEIPT, "failure": FAILURE,
            "lock": LOCK,
        }.items()},
        "authorized_for_fit_execution": False,
        "authorized_for_candidate_selection": False,
        "authorized_for_scored_experiments": False,
    }
    return {**body, "authority_sha256": logical_sha256(body)}


def validate_canonical_audit(path: Path = AUDIT) -> dict[str, Any]:
    if path.resolve() != AUDIT.resolve():
        raise RuntimeError("fit-mean audit is not the canonical audit path")
    audit = _stable_json(path)
    reviewed = audit.get("reviewed_source_sha256s")
    expected_reviewed = {
        "basis_aligned/polynomial_causal/terminal_copy_fit_mean_lifecycle.py":
            file_sha256(Path(__file__).resolve()),
        "basis_aligned/polynomial_causal/test_terminal_copy_fit_mean_lifecycle.py":
            file_sha256(HERE / "test_terminal_copy_fit_mean_lifecycle.py"),
    }
    if (
        set(audit) != {
            "schema", "status", "approved", "outcome_access", "reviewer",
            "reviewed_source_sha256s", "focused_tests", "remaining_launch_blockers",
        }
        or audit.get("schema") != "terminal_copy_fit_mean_lifecycle_independent_audit_v1"
        or audit.get("status") != "approved_outcome_blind_infrastructure"
        or audit.get("approved") is not True
        or audit.get("outcome_access") is not False
        or audit.get("reviewer") != "independent_artifact_audit_agent"
        or reviewed != expected_reviewed
        or not isinstance(audit.get("focused_tests"), Mapping)
        or audit["focused_tests"].get("passed") is not True
        or type(audit["focused_tests"].get("count")) is not int
        or audit["focused_tests"]["count"] <= 0
        or not isinstance(audit.get("remaining_launch_blockers"), list)
    ):
        raise RuntimeError("fit-mean canonical audit semantics or reviewed bytes changed")
    return audit


def freeze_execution_authority(independent_audit_path: Path) -> dict[str, Any]:
    """Publish the sole fit-only authority before any row tensor or model load."""

    require_pristine_namespace()
    audit = validate_canonical_audit(independent_audit_path)
    checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
    body = {
        "schema": "terminal_copy_fit_means_v1_authority",
        "status": "frozen_before_any_fit_model_forward_after_disclosed_parent_container_access",
        "source_closure": source_closure(),
        "row_binding": row_binding(),
        "adapter_binding": adapter_binding(),
        "checkpoint": asdict(checkpoint),
        "protocol": protocol(),
        "outputs": {name: str(path) for name, path in {
            "authority": AUTHORITY, "bank": BANK, "result": RESULT,
            "manifest": MANIFEST, "receipt": RECEIPT, "failure": FAILURE,
            "lock": LOCK,
        }.items()},
        "protected_paths": [str(path) for path in PROTECTED_PATHS],
        "independent_audit": {
            "approved": audit["approved"],
            "outcome_access": audit["outcome_access"],
            "path": str(independent_audit_path.resolve()),
            "sha256": file_sha256(independent_audit_path),
        },
        "authorized_for_fit_execution": True,
        "authorized_for_candidate_selection": False,
        "authorized_for_scored_experiments": False,
    }
    authority = {**body, "authority_sha256": logical_sha256(body)}
    create_only_json(AUTHORITY, authority)
    validate_execution_authority(authority)
    return authority


def validate_execution_authority(authority: Mapping[str, Any]) -> None:
    """Replay all parent/source/model bindings before publication or execution."""

    expected_keys = {
        "schema", "status", "source_closure", "row_binding", "adapter_binding",
        "checkpoint", "protocol", "outputs", "protected_paths",
        "independent_audit", "authorized_for_fit_execution",
        "authorized_for_candidate_selection", "authorized_for_scored_experiments",
        "authority_sha256",
    }
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if (
        set(authority) != expected_keys
        or authority.get("schema") != "terminal_copy_fit_means_v1_authority"
        or authority.get("status")
        != "frozen_before_any_fit_model_forward_after_disclosed_parent_container_access"
        or authority.get("authorized_for_fit_execution") is not True
        or authority.get("authorized_for_candidate_selection") is not False
        or authority.get("authorized_for_scored_experiments") is not False
        or logical_sha256(body) != authority.get("authority_sha256")
        or authority.get("row_binding") != row_binding()
        or authority.get("adapter_binding") != adapter_binding()
        or authority.get("protocol") != protocol()
        or authority.get("checkpoint") != asdict(
            facade.validate_snapshot(verify_weights_sha256=True)
        )
        or authority.get("outputs") != {name: str(path) for name, path in {
            "authority": AUTHORITY, "bank": BANK, "result": RESULT,
            "manifest": MANIFEST, "receipt": RECEIPT, "failure": FAILURE,
            "lock": LOCK,
        }.items()}
        or authority.get("protected_paths") != [str(path) for path in PROTECTED_PATHS]
    ):
        raise RuntimeError("fit-mean execution authority identity changed")
    audit = authority.get("independent_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("approved") is not True
        or audit.get("outcome_access") is not False
        or not isinstance(audit.get("path"), str)
        or not isinstance(audit.get("sha256"), str)
        or Path(audit["path"]).resolve() != AUDIT.resolve()
        or file_sha256(AUDIT) != audit["sha256"]
    ):
        raise RuntimeError("fit-mean independent audit is absent or changed")
    validate_canonical_audit(AUDIT)
    verify_source_closure(authority["source_closure"])
    if not AUTHORITY.is_file() or _stable_json(AUTHORITY) != dict(authority):
        raise RuntimeError("fit-mean authority file differs from supplied authority")


def _bank_payload(bank: FitHeadMeanBank, authority_sha256: str) -> dict[str, Any]:
    if not isinstance(bank, FitHeadMeanBank) or not bank.verify_hashes():
        raise RuntimeError("fit-mean bank is invalid before serialization")
    return {
        "schema": "terminal_copy_fit_means_v1_bank",
        "authority_sha256": authority_sha256,
        "document_count": bank.document_count,
        "ordered_document_ids_sha256": bank.ordered_document_ids_sha256,
        "source_dtype": bank.source_dtype,
        "accumulator_dtype": bank.accumulator_dtype,
        "published_dtype": bank.published_dtype,
        "master_means_sha256": bank.master_means_sha256,
        "runtime_means_sha256": bank.runtime_means_sha256,
        "heads_by_layer": {str(k): list(v) for k, v in NAMED_HEADS_BY_LAYER.items()},
        "master": {str(k): v for k, v in bank.clone_master_means().items()},
        "runtime": {str(k): v for k, v in bank.clone_means().items()},
    }


def load_bank_semantically(
    path: Path, authority_sha256: str, *, require_production: bool = True,
) -> FitHeadMeanBank:
    """Reload tensors and prove their meaning, rather than trusting a file hash alone."""

    before = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    after = file_sha256(path)
    if before != after:
        raise RuntimeError("serialized fit-mean bank changed during semantic reload")
    expected_keys = {
        "schema", "authority_sha256", "document_count",
        "ordered_document_ids_sha256", "source_dtype", "accumulator_dtype",
        "published_dtype", "master_means_sha256", "runtime_means_sha256",
        "heads_by_layer", "master", "runtime",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys or (
        payload.get("schema") != "terminal_copy_fit_means_v1_bank"
        or payload.get("authority_sha256") != authority_sha256
        or payload.get("heads_by_layer") != {
            str(k): list(v) for k, v in NAMED_HEADS_BY_LAYER.items()
        }
    ):
        raise RuntimeError("serialized fit-mean bank schema changed")
    master = payload["master"]
    runtime = payload["runtime"]
    if not isinstance(master, dict) or not isinstance(runtime, dict) or (
        set(master) != {str(k) for k in NAMED_LAYERS}
        or set(runtime) != {str(k) for k in NAMED_LAYERS}
    ):
        raise RuntimeError("serialized fit-mean bank topology changed")
    master_int = {int(k): v for k, v in master.items()}
    runtime_int = {int(k): v for k, v in runtime.items()}
    for layer in NAMED_LAYERS:
        expected_shape = (256, len(NAMED_HEADS_BY_LAYER[layer]), 1152)
        if require_production and (
            tuple(master_int[layer].shape) != expected_shape
            or tuple(runtime_int[layer].shape) != expected_shape
        ):
            raise RuntimeError("serialized production mean shape changed")
        if (
            master_int[layer].device.type != "cpu"
            or runtime_int[layer].device.type != "cpu"
            or master_int[layer].dtype != torch.float64
            or runtime_int[layer].dtype != torch.float32
            or not bool(torch.isfinite(master_int[layer]).all())
            or not bool(torch.isfinite(runtime_int[layer]).all())
            or not torch.equal(master_int[layer].float(), runtime_int[layer])
        ):
            raise RuntimeError("serialized fit-mean tensor semantics changed")
    if require_production and (
        payload["document_count"] != 192
        or payload["source_dtype"] != "torch.bfloat16"
        or payload["accumulator_dtype"] != "torch.float64"
        or payload["published_dtype"] != "torch.float32"
    ):
        raise RuntimeError("serialized production numeric contract changed")
    bank = FitHeadMeanBank(
        per_head_position_means=runtime_int,
        master_per_head_position_means=master_int,
        document_count=payload["document_count"],
        ordered_document_ids_sha256=payload["ordered_document_ids_sha256"],
        runtime_means_sha256=payload["runtime_means_sha256"],
        master_means_sha256=payload["master_means_sha256"],
        accumulator_dtype=payload["accumulator_dtype"],
        published_dtype=payload["published_dtype"],
        source_dtype=payload["source_dtype"],
    )
    if not bank.verify_hashes():
        raise RuntimeError("serialized fit-mean hash replay failed")
    return bank


def validate_closure(closure: FitMeanOwnerClosure) -> dict[str, Any]:
    if not isinstance(closure, FitMeanOwnerClosure) or not closure.closed:
        raise RuntimeError("fit-mean owner did not close")
    batches = closure.batch_calls
    if (
        type(batches) is not int or batches <= 0
        or closure.document_calls != 192
        or closure.native_attention_calls != (batches,) * 18
        or closure.native_mlp_calls != (batches,) * 18
        or closure.adapter_decomposition_calls != tuple(
            batches if layer in NAMED_LAYERS else 0 for layer in range(18)
        )
        or closure.native_unembedding_calls != 0
        or len(closure.final_state_sha256s) != batches
        or any(len(value) != 64 for value in closure.final_state_sha256s)
        or closure.maximum_full_write_abs_error != 0.0
        or not all(math.isfinite(value) and value >= 0.0 for value in (
            closure.maximum_head_recomposition_abs_error,
            closure.maximum_head_recomposition_relative_error,
        ))
        or closure.maximum_head_recomposition_relative_error > 0.01
    ):
        raise RuntimeError("fit-mean physical call census or replay integrity failed")
    return asdict(closure)


class FitRoleInputs(NamedTuple):
    tokens: torch.Tensor
    ordered_document_ids: tuple[str, ...]
    ordered_document_ids_sha256: str
    row_file_sha256: str


_COLLECTION_SEAL = object()


class _CollectedFitTransaction:
    """Opaque return capability created only after the owned collector closes."""

    __slots__ = (
        "bank", "closure", "ordered_document_ids_sha256", "row_file_sha256",
        "checkpoint_weights_sha256_before", "checkpoint_weights_sha256_after",
        "claim_nonce", "authority_sha256",
    )

    def __init__(
        self, seal: object, *, bank: FitHeadMeanBank, closure: FitMeanOwnerClosure,
        ordered_document_ids_sha256: str, row_file_sha256: str,
        checkpoint_weights_sha256_before: str,
        checkpoint_weights_sha256_after: str,
        claim: RunClaim, authority_sha256: str,
    ) -> None:
        if seal is not _COLLECTION_SEAL:
            raise RuntimeError("fit collection capability cannot be constructed externally")
        self.bank = bank
        self.closure = closure
        self.ordered_document_ids_sha256 = ordered_document_ids_sha256
        self.row_file_sha256 = row_file_sha256
        self.checkpoint_weights_sha256_before = checkpoint_weights_sha256_before
        self.checkpoint_weights_sha256_after = checkpoint_weights_sha256_after
        self.claim_nonce = claim.nonce
        self.authority_sha256 = authority_sha256


def _load_fit_role_inputs(
    authority: Mapping[str, Any], claim: RunClaim,
) -> FitRoleInputs:
    """Load only the fit rows/records fields under the owned execution lock."""

    require_claim(claim)
    validate_execution_authority(authority)
    binding = authority["row_binding"]
    path = Path(binding["input_path"])
    before = file_sha256(path)
    if before != binding["input_file_sha256"]:
        raise RuntimeError("fit-input projection differs from authority before load")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    after = file_sha256(path)
    if before != after:
        raise RuntimeError("fit-input projection changed during load")
    if not isinstance(payload, dict) or set(payload) != set(binding["payload_fields"]):
        raise RuntimeError("fit-input projection is not an input-only object")
    tokens = payload.get("tokens")
    document_ids = payload.get("ordered_document_ids")
    if (
        payload.get("schema") != "terminal_copy_fit_inputs_v1_payload"
        or payload.get("authority_sha256") != binding["projection_authority_sha256"]
        or not torch.is_tensor(tokens)
        or tokens.device.type != "cpu"
        or tokens.dtype != torch.long
        or tuple(tokens.shape) != (192, 256)
        or tensor_sha256(tokens) != binding["tokens_sha256"]
        or not isinstance(document_ids, tuple)
        or len(document_ids) != 192
    ):
        raise RuntimeError("fit-input tensor or document topology changed")
    if len(set(document_ids)) != 192 or any(not isinstance(x, str) or not x for x in document_ids):
        raise RuntimeError("fit-role documents are not unique")
    documents = document_ids
    if _document_digest(documents) != binding["ordered_document_ids_sha256"]:
        raise RuntimeError("fit-input ordered document digest changed")
    tokens = tokens.clone()
    del payload
    return FitRoleInputs(
        tokens=tokens,
        ordered_document_ids=documents,
        ordered_document_ids_sha256=_document_digest(documents),
        row_file_sha256=after,
    )


def _create_only_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    _create_only_bytes(path, (json.dumps(
        dict(value), sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode())


def create_only_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        with temporary.open("xb") as destination:
            torch.save(dict(value), destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_fit_mean_bundle(
    *, authority: Mapping[str, Any], claim: RunClaim,
    collected: _CollectedFitTransaction,
    protected_before: Mapping[str, str | None],
    protected_after: Mapping[str, str | None],
) -> dict[str, Any]:
    """Publish bank/result/manifest then the sole success receipt last."""

    require_claim(claim)
    validate_execution_authority(authority)
    authority_sha = str(authority.get("authority_sha256", ""))
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if len(authority_sha) != 64 or logical_sha256(body) != authority_sha:
        raise RuntimeError("fit-mean execution authority digest changed")
    live_protected = protected_snapshot()
    if (
        dict(protected_before) != live_protected
        or dict(protected_after) != live_protected
    ):
        raise RuntimeError("fit-mean collection changed protected artifacts")
    if (
        not isinstance(collected, _CollectedFitTransaction)
        or collected.claim_nonce != claim.nonce
        or collected.authority_sha256 != authority_sha
    ):
        raise RuntimeError("fit-mean collection capability is absent or from another run")
    bank = collected.bank
    closure_payload = validate_closure(collected.closure)
    ordered_document_ids_sha256 = collected.ordered_document_ids_sha256
    row_file_sha256 = collected.row_file_sha256
    weights_before = collected.checkpoint_weights_sha256_before
    weights_after = collected.checkpoint_weights_sha256_after
    if (
        bank.ordered_document_ids_sha256 != ordered_document_ids_sha256
        or row_file_sha256 != authority["row_binding"]["input_file_sha256"]
        or weights_before != facade.WEIGHTS_SHA256
        or weights_after != facade.WEIGHTS_SHA256
    ):
        raise RuntimeError("fit-mean bank is not joined to authorized ordered documents")
    payload = _bank_payload(bank, authority_sha)
    create_only_torch(BANK, payload)
    replay = load_bank_semantically(BANK, authority_sha, require_production=True)
    result = {
        "schema": "terminal_copy_fit_means_v1_result",
        "status": "complete_fit_only_no_outcome_access",
        "authority_sha256": authority_sha,
        "bank_file_sha256": file_sha256(BANK),
        "master_means_sha256": replay.master_means_sha256,
        "runtime_means_sha256": replay.runtime_means_sha256,
        "ordered_document_ids_sha256": ordered_document_ids_sha256,
        "row_file_sha256": row_file_sha256,
        "checkpoint_weights_sha256_before_load": weights_before,
        "checkpoint_weights_sha256_after_load": weights_after,
        "owner_closure": closure_payload,
        "outcome_access": {
            "unembedding_calls": 0, "loss_or_logit_reads": 0,
            "label_or_copy_cell_reads": 0, "candidate_selection": False,
        },
        "authorized_for_E4_evidence": False,
    }
    create_only_json(RESULT, result)
    manifest = {
        "schema": "terminal_copy_fit_means_v1_manifest",
        "authority_sha256": authority_sha,
        "files": {
            str(BANK): file_sha256(BANK),
            str(RESULT): file_sha256(RESULT),
        },
        "protected_before": dict(protected_before),
        "protected_after": dict(protected_after),
        "protected_unchanged": True,
    }
    create_only_json(MANIFEST, manifest)
    # Receipt-last is a separate integrity barrier, not merely the next write.
    require_claim(claim)
    validate_execution_authority(authority)
    final_protected = protected_snapshot()
    final_replay = load_bank_semantically(BANK, authority_sha, require_production=True)
    protected_after_replay = protected_snapshot()
    if (
        final_protected != live_protected
        or protected_after_replay != live_protected
        or FAILURE.exists()
        or RECEIPT.exists()
        or _stable_json(RESULT) != result
        or _stable_json(MANIFEST) != manifest
        or file_sha256(BANK) != result["bank_file_sha256"]
        or file_sha256(RESULT) != manifest["files"][str(RESULT)]
        or final_replay.runtime_means_sha256 != replay.runtime_means_sha256
    ):
        raise RuntimeError("fit-mean terminal publication recheck failed")
    receipt = {
        "schema": "terminal_copy_fit_means_v1_receipt",
        "status": "complete_receipt_last_fit_only",
        "authority_file_sha256": file_sha256(AUTHORITY),
        "authority_sha256": authority_sha,
        "bank_file_sha256": file_sha256(BANK),
        "result_file_sha256": file_sha256(RESULT),
        "manifest_file_sha256": file_sha256(MANIFEST),
        "master_means_sha256": replay.master_means_sha256,
        "runtime_means_sha256": replay.runtime_means_sha256,
        "ordered_document_ids_sha256": ordered_document_ids_sha256,
        "row_file_sha256": row_file_sha256,
        "checkpoint_weights_sha256_before_load": weights_before,
        "checkpoint_weights_sha256_after_load": weights_after,
        "document_count": replay.document_count,
        "selection_or_outcome_access": False,
        "fit_means_prerequisite_complete": True,
        "authorized_for_candidate_selection_parent": False,
        "authorized_for_E4_evidence": False,
    }
    create_only_json(RECEIPT, receipt)
    return receipt


def _publish_failure(
    claim: RunClaim, authority_sha256: str, error: BaseException,
) -> None:
    """Close a failed namespace without ever manufacturing a success receipt."""

    require_claim(claim)
    if RECEIPT.exists():
        raise RuntimeError("cannot publish failure after success receipt")
    create_only_json(FAILURE, {
        "schema": "terminal_copy_fit_means_v1_failure",
        "status": "terminal_failure_no_success_receipt",
        "authority_sha256": authority_sha256,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "bank_exists": BANK.exists(),
        "result_exists": RESULT.exists(),
        "manifest_exists": MANIFEST.exists(),
        "receipt_exists": False,
    })


def execute_fit_mean_collection(
    *, device: str | torch.device = "cuda", batch_size: int = 4,
) -> dict[str, Any]:
    """Run the only evidence-producing fit path under one owned transaction.

    The exact checkpoint is loaded internally after authority validation.  The
    dispatcher, accumulator, owner, mean bank, and closure never cross the public
    boundary; only the non-scientific fit receipt is returned.
    """

    if type(batch_size) is not int or batch_size <= 0 or batch_size > 192:
        raise ValueError("fit-mean batch size is malformed")
    require_pristine_execution_namespace()
    authority = _stable_json(AUTHORITY)
    validate_execution_authority(authority)
    claim = acquire_claim()
    model: torch.nn.Module | None = None
    try:
        require_claim(claim)
        protected_before = protected_snapshot()
        inputs = _load_fit_role_inputs(authority, claim)
        weights_path = facade.DEFAULT_SNAPSHOT / "pytorch_model.bin"
        weights_before = file_sha256(weights_path)
        if weights_before != facade.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint weights changed immediately before load")
        model, loaded = facade.load_bilin18(
            device=device, dtype=torch.bfloat16, verify_weights_sha256=False,
        )
        weights_after = file_sha256(weights_path)
        if (
            weights_after != weights_before
            or weights_after != facade.WEIGHTS_SHA256
            or asdict(loaded) != authority["checkpoint"]
        ):
            raise RuntimeError("loaded checkpoint differs from fit-mean authority")
        means = {
            layer: torch.zeros(
                256, len(NAMED_HEADS_BY_LAYER[layer]), 1152,
                dtype=torch.float32, device=device,
            )
            for layer in NAMED_LAYERS
        }
        dispatcher = PhysicalCandidateDispatcher.from_native(
            attentions={layer: model.transformer.h[layer].attn for layer in NAMED_LAYERS},
            per_head_position_means=means,
        )
        accumulator = FitHeadMeanAccumulator(
            ordered_document_ids=inputs.ordered_document_ids,
            sequence_length=256,
            n_head=9,
            width=1152,
            published_dtype=torch.float32,
            source_dtype=torch.bfloat16,
            require_production=True,
        )
        owner = FitMeanCollectionOwner(dispatcher=dispatcher, accumulator=accumulator)
        for start in range(0, 192, batch_size):
            stop = min(start + batch_size, 192)
            owner.collect_batch(
                model,
                inputs.tokens[start:stop].to(device=device),
                inputs.ordered_document_ids[start:stop],
                require_production=True,
            )
        bank, closure = owner.finalize()
        if bank.ordered_document_ids_sha256 != inputs.ordered_document_ids_sha256:
            raise RuntimeError("fit owner changed authorized document order")
        protected_after = protected_snapshot()
        collected = _CollectedFitTransaction(
            _COLLECTION_SEAL,
            bank=bank,
            closure=closure,
            ordered_document_ids_sha256=inputs.ordered_document_ids_sha256,
            row_file_sha256=inputs.row_file_sha256,
            checkpoint_weights_sha256_before=weights_before,
            checkpoint_weights_sha256_after=weights_after,
            claim=claim,
            authority_sha256=authority["authority_sha256"],
        )
        return _publish_fit_mean_bundle(
            authority=authority,
            claim=claim,
            collected=collected,
            protected_before=protected_before,
            protected_after=protected_after,
        )
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            _publish_failure(
                claim,
                str(authority.get("authority_sha256", "")),
                error,
            )
        raise
    finally:
        del model
        release_claim(claim)
