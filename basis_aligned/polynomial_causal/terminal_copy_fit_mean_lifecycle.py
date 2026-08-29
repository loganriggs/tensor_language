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
from typing import Any, Mapping, Sequence

import torch

import bilin18_observed_model_facade as facade
from terminal_copy_attention_dispatcher import NAMED_LAYERS
from terminal_copy_fit_head_means import (
    FitHeadMeanBank,
    NAMED_HEADS_BY_LAYER,
)
from terminal_copy_fit_mean_owner import FitMeanOwnerClosure


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BQ = ROOT / "basis_aligned" / "bilinear_quotient"

ROW_RECEIPT = BQ / "terminal_copy_induction_v2_rows_receipt.json"
ROW_RECEIPT_SHA256 = "aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8"
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

SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/TERMINAL_COPY_FIT_HEAD_MEANS_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/TERMINAL_COPY_INDUCTION_V1_SCREENING_AMENDMENT.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/terminal_copy_attention_adapter.py",
    "basis_aligned/polynomial_causal/terminal_copy_attention_dispatcher.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_head_means.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_owner.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_lifecycle.py",
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


def require_pristine_namespace(paths: Sequence[Path] | None = None) -> None:
    selected = output_namespace() if paths is None else tuple(paths)
    spent = [str(path) for path in selected if path.exists()]
    if spent:
        raise RuntimeError(f"fit-mean output namespace is spent: {spent}")


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
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"fit-mean source drift: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(body["commit"]), "origin/main"],
        cwd=ROOT, check=True,
    )


def row_binding() -> dict[str, Any]:
    """Bind only fit-role receipt metadata; do not open the row tensor."""

    if file_sha256(ROW_RECEIPT) != ROW_RECEIPT_SHA256:
        raise RuntimeError("terminal-copy row receipt bytes changed")
    receipt = _stable_json(ROW_RECEIPT)
    entry = receipt.get("entries", {}).get("fit_natural", {})
    license_ = receipt.get("role_licenses", {}).get("fit_natural", {})
    if (
        receipt.get("status") != "frozen_before_any_terminal_copy_model_forward"
        or receipt.get("authorized_for_scored_experiments") is not False
        or receipt.get("authorized_for_candidate_or_threshold_selection") is not False
        or receipt.get("summary", {}).get("roles", {}).get("fit_natural") != 192
        or license_ != {
            "authorized_use": "fit_per_position_head_write_means_only",
            "requires_receipt": None,
        }
        or not isinstance(entry.get("path"), str)
        or not isinstance(entry.get("file_sha256"), str)
        or not isinstance(entry.get("rows_tensor_sha256"), str)
    ):
        raise RuntimeError("fit-role row receipt semantics changed")
    body = {
        "receipt_path": str(ROW_RECEIPT),
        "receipt_sha256": ROW_RECEIPT_SHA256,
        "role": "fit_natural",
        "row_count": 192,
        "row_width": 257,
        "model_input_width": 256,
        "row_path": entry["path"],
        "row_file_sha256": entry["file_sha256"],
        "rows_tensor_sha256": entry["rows_tensor_sha256"],
        "authorized_use": license_["authorized_use"],
        "labels_or_copy_cells_authorized": False,
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
        "row_width": 257,
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

    payload = torch.load(path, map_location="cpu", weights_only=True)
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


def publish_fit_mean_bundle(
    *, authority: Mapping[str, Any], bank: FitHeadMeanBank,
    closure: FitMeanOwnerClosure, protected_before: Mapping[str, str | None],
    protected_after: Mapping[str, str | None],
) -> dict[str, Any]:
    """Publish bank/result/manifest then the sole success receipt last."""

    if authority.get("schema") != "terminal_copy_fit_means_v1_authority" or (
        authority.get("authorized_for_fit_execution") is not True
        or authority.get("authorized_for_candidate_selection") is not False
        or authority.get("authorized_for_scored_experiments") is not False
    ):
        raise RuntimeError("fit-mean execution authority is absent or nonauthorizing")
    authority_sha = str(authority.get("authority_sha256", ""))
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if len(authority_sha) != 64 or logical_sha256(body) != authority_sha:
        raise RuntimeError("fit-mean execution authority digest changed")
    verify_source_closure(authority["source_closure"])
    if dict(protected_before) != dict(protected_after):
        raise RuntimeError("fit-mean collection changed protected artifacts")
    closure_payload = validate_closure(closure)
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
        "document_count": replay.document_count,
        "selection_or_outcome_access": False,
        "authorized_for_candidate_selection_parent": True,
        "authorized_for_E4_evidence": False,
    }
    create_only_json(RECEIPT, receipt)
    return receipt


def publish_failure(authority_sha256: str, error: BaseException) -> None:
    """Close a failed namespace without ever manufacturing a success receipt."""

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
