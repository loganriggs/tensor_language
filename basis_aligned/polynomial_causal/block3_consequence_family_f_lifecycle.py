"""Outcome-blind lifecycle bindings for the Block-3 family-F fit transaction.

This module deliberately does not load the n480 tensor or execute the model.  It
constructs and replays everything that must be frozen before the numerical runner is
allowed to do either of those things.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/workspace/tensor_language")
sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
import block3_consequence_fit as core
import collect_block3_native_gate_fit_v1 as collector


HERE = ROOT / "basis_aligned" / "polynomial_causal"
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
AMENDMENT = HERE / "BLOCK3_NATIVE_GATE_SUBSET_V1_FAMILY_F_AMENDMENT.md"
ROWS = BQ / ".rowcache/fineweb_n480_skip80.pt"
ROLE = "n480_skip80"
ROW_COUNT = 480
ROW_WIDTH = 513
ROWS_FILE_SHA256 = "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496"
ROWS_RAW_SHA256 = "343d92ce07f78572e3233120d3361814c63f69fa76e97e58b62d1d6c8f24497f"
DOCUMENT_COUNT = 209

AUTHORITY = HERE / "block3_consequence_family_f_v1_authority.json"
PROGRAMS = HERE / "block3_consequence_family_f_v1_programs.pt"
RESULTS = HERE / "block3_consequence_family_f_v1_fit_results.json"
RECEIPT = HERE / "block3_consequence_family_f_v1_receipt.json"
FAILURE = HERE / "block3_consequence_family_f_v1_failure.json"
LOCK = Path("/workspace/runs/.block3_consequence_family_f_v1.lock")

PRIOR_PATHS = (
    collector.AUTHORITY,
    collector.PAYLOAD,
    collector.RECEIPT,
    HERE / "block3_native_gate_subset_v1_fit_authority.json",
    HERE / "block3_native_gate_subset_v1_programs.pt",
    HERE / "block3_native_gate_subset_v1_fit_results.json",
    HERE / "block3_native_gate_subset_v1_fit_receipt.json",
    HERE / "block3_native_gate_subset_v1_validation_v1_authority.json",
    HERE / "block3_native_gate_subset_v1_validation_v1_results.json",
    HERE / "block3_native_gate_subset_v1_validation_v1_receipt.json",
)
CANONICAL_PRIOR_HASHES = {
    "basis_aligned/polynomial_causal/block3_native_gate_fit_v1_authority.json":
        "cd83bbcd5dbf466a7ab57617a2b28ef2f62943c2ef011e1d71821a4547f8351a",
    "basis_aligned/polynomial_causal/block3_native_gate_fit_v1_payload.pt":
        "8b25774257dd66f34ff6fb0c21fb0613efb12ce2773a0f3dcc8082343ceebdd9",
    "basis_aligned/polynomial_causal/block3_native_gate_fit_v1_receipt.json":
        "3facf67a90beea923a666f29a0770080d2277721e4443df444b6a416b186435e",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_fit_authority.json":
        "d166df7a02a39296c4c95052f392ac29157947db41038788afab83222babae98",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_programs.pt":
        "6f1ac8b2043edd1cb2a73992ee869c16c7516af27c608b56e821d57a06334d36",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_fit_results.json":
        "22517a762c2f7c570c6fc383e3530635d7030a81a7ef1f4a6d735060ce971fbe",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_fit_receipt.json":
        "dbf76301d4c7d5ac03942465da98c331429e1b2e84c771d767b1302b0137ab89",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_validation_v1_authority.json":
        "4fa0ea1c2e64bc7a02d375d4496ff552ae4dd9074c721c23c1bfce6233324db2",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_validation_v1_results.json":
        "1abf36d9ab1313b665f7bf9a7cac4fefb8c4d09aeb2ee33f3f7230641a11e0b4",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_validation_v1_receipt.json":
        "4f0f1153263bb5312041405290940c6e8bdb0a958d3941cc34f0fbc8c2dad3f8",
}

SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_FAMILY_F_AMENDMENT.md",
    "basis_aligned/polynomial_causal/block3_consequence_fit.py",
    "basis_aligned/polynomial_causal/test_block3_consequence_fit.py",
    "basis_aligned/polynomial_causal/block3_consequence_family_f_lifecycle.py",
    "basis_aligned/polynomial_causal/test_block3_consequence_family_f_lifecycle.py",
    "basis_aligned/polynomial_causal/collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/test_collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/native_gate_subset.py",
    "basis_aligned/polynomial_causal/test_native_gate_subset.py",
    "basis_aligned/polynomial_causal/grouped_block_coefficient_screen.py",
    "basis_aligned/polynomial_causal/test_grouped_block_coefficient_screen.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
)


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return collector.file_sha256(path)


def output_namespace() -> tuple[Path, ...]:
    return AUTHORITY, PROGRAMS, RESULTS, RECEIPT, FAILURE, LOCK


def require_pristine_namespace(paths: Sequence[Path] | None = None) -> None:
    spent = [str(path) for path in (output_namespace() if paths is None else paths) if path.exists()]
    if spent:
        raise RuntimeError(f"family-F output namespace is spent: {spent}")


def source_closure() -> dict[str, Any]:
    """Bind live source to one pushed commit before the authority is published."""

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"family-F source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"live family-F source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def verify_source_closure(binding: Mapping[str, Any]) -> None:
    if set(binding) != {"commit", "paths", "sha256"} or set(
        binding["paths"]
    ) != set(SOURCE_PATHS) or logical_sha256({
        "commit": binding["commit"], "paths": binding["paths"],
    }) != binding["sha256"]:
        raise RuntimeError("family-F source closure is malformed")
    for relative, digest in binding["paths"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"family-F source drift: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", binding["commit"], "origin/main"],
        cwd=ROOT, check=True,
    )


def prior_artifact_binding() -> dict[str, Any]:
    """Bind family A's fit and failed-validation branch without loading F outcomes."""

    if not all(path.is_file() for path in PRIOR_PATHS):
        raise RuntimeError("family-F prior artifact lineage is incomplete")
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in PRIOR_PATHS}
    if hashes != CANONICAL_PRIOR_HASHES:
        raise RuntimeError("family-F canonical parent artifact bytes changed")
    collector_authority = json.loads(collector.AUTHORITY.read_text())
    collector_receipt = json.loads(collector.RECEIPT.read_text())
    fit_authority = json.loads(PRIOR_PATHS[3].read_text())
    fit_receipt = json.loads(PRIOR_PATHS[6].read_text())
    validation_authority = json.loads(PRIOR_PATHS[7].read_text())
    validation_result = json.loads(PRIOR_PATHS[8].read_text())
    validation_receipt = json.loads(PRIOR_PATHS[9].read_text())
    expected_action = {
        "kind": "stop_activation_family_and_preregister_finite_suffix_family",
        "budget": None,
    }
    if collector_receipt.get("authority_sha256") != collector_authority.get(
        "authority_sha256"
    ) or collector_receipt.get("authority_file_sha256") != hashes[
        str(PRIOR_PATHS[0].relative_to(ROOT))
    ] or collector_receipt.get("payload_file_sha256") != hashes[
        str(PRIOR_PATHS[1].relative_to(ROOT))
    ] or fit_authority.get("collector_authority_sha256") != collector_authority.get(
        "authority_sha256"
    ) or fit_authority.get("collector_input_file_sha256s") != {
        str(path.relative_to(ROOT)): hashes[str(path.relative_to(ROOT))]
        for path in PRIOR_PATHS[:3]
    } or fit_receipt.get("fit_authority_sha256") != fit_authority.get(
        "fit_authority_sha256"
    ) or fit_receipt.get("fit_authority_file_sha256") != hashes[
        str(PRIOR_PATHS[3].relative_to(ROOT))
    ] or fit_receipt.get("programs_file_sha256") != hashes[
        str(PRIOR_PATHS[4].relative_to(ROOT))
    ] or fit_receipt.get("results_file_sha256") != hashes[
        str(PRIOR_PATHS[5].relative_to(ROOT))
    ] or fit_receipt.get("collector_receipt_file_sha256") != hashes[
        str(PRIOR_PATHS[2].relative_to(ROOT))
    ] or validation_receipt.get("authority_sha256") != validation_authority.get(
        "authority_sha256"
    ) or validation_receipt.get("authority_file_sha256") != hashes[
        str(PRIOR_PATHS[7].relative_to(ROOT))
    ] or validation_receipt.get("results_file_sha256") != hashes[
        str(PRIOR_PATHS[8].relative_to(ROOT))
    ] or validation_receipt.get("fit_authority_sha256") != fit_authority.get(
        "fit_authority_sha256"
    ) or validation_result.get("authority_sha256") != validation_authority.get(
        "authority_sha256"
    ) or validation_result.get("registered_next_action") != expected_action or (
        validation_result.get("validation_eligible_budget") is not None
    ) or validation_result.get("final_rows_loaded") != 0:
        raise RuntimeError("family-F activation branch or artifact joins changed")
    body = {
        "file_sha256s": hashes,
        "collector_authority_sha256": collector_authority["authority_sha256"],
        "fit_authority_sha256": fit_authority["fit_authority_sha256"],
        "validation_authority_sha256": validation_authority["authority_sha256"],
        "registered_branch": expected_action,
    }
    return {**body, "sha256": logical_sha256(body)}


def verify_prior_artifact_binding(binding: Mapping[str, Any]) -> None:
    if prior_artifact_binding() != dict(binding):
        raise RuntimeError("family-F prior artifact binding drift")


def row_binding() -> dict[str, Any]:
    """Freeze n480 identity/provenance and the selector-null map without loading rows."""

    if file_sha256(collector.ROW_RECEIPT) != collector.ROW_RECEIPT_SHA256:
        raise RuntimeError("canonical row receipt bytes changed")
    receipt = json.loads(collector.ROW_RECEIPT.read_text())
    entry = receipt.get("entries", {}).get(ROLE, {})
    records = receipt.get("document_provenance", {}).get("sets", {}).get(ROLE)
    if entry.get("shape") != [ROW_COUNT, ROW_WIDTH] or file_sha256(
        ROWS
    ) != ROWS_FILE_SHA256 or entry.get(
        "tensor_raw_sha256"
    ) != ROWS_RAW_SHA256 or Path(entry.get("cache_path", "")) != ROWS or not isinstance(
        records, list
    ) or len(records) != ROW_COUNT:
        raise RuntimeError("canonical family-F row metadata changed")
    document_ids = [record.get("document_id") for record in records]
    if any(not isinstance(document, str) for document in document_ids):
        raise RuntimeError("family-F document provenance is malformed")
    ordered_documents = list(dict.fromkeys(document_ids))
    document_index = {document: index for index, document in enumerate(ordered_documents)}
    row_to_document = torch.tensor(
        [document_index[document] for document in document_ids], dtype=torch.long,
    )
    if len(ordered_documents) != DOCUMENT_COUNT:
        raise RuntimeError("family-F source-document count changed")
    donor_rows = core.document_deranged_row_map(row_to_document)
    donor_reuse = torch.bincount(donor_rows, minlength=ROW_COUNT)
    reversal = torch.arange(ROW_COUNT, dtype=torch.long).reshape(-1, 8).flip(1).flatten()
    reversal_same_document = row_to_document[reversal] == row_to_document
    if int(reversal_same_document.sum()) != 132:
        raise RuntimeError("family-F weak reversal collision census changed")
    provenance = collector.validate_row_provenance()
    body = {
        "role": ROLE,
        "row_count": ROW_COUNT,
        "row_width": ROW_WIDTH,
        "row_receipt_sha256": collector.ROW_RECEIPT_SHA256,
        "row_file_path": str(ROWS),
        "row_file_sha256": ROWS_FILE_SHA256,
        "row_raw_sha256": ROWS_RAW_SHA256,
        "ordered_document_ids": ordered_documents,
        "ordered_document_ids_sha256": logical_sha256(ordered_documents),
        "row_to_document": row_to_document.tolist(),
        "row_to_document_sha256": logical_sha256(row_to_document.tolist()),
        "document_deranged_donor_rows": donor_rows.tolist(),
        "document_deranged_donor_rows_sha256": logical_sha256(donor_rows.tolist()),
        "logical_batch_reversal_rows": reversal.tolist(),
        "logical_batch_reversal_rows_sha256": logical_sha256(reversal.tolist()),
        "logical_batch_reversal_same_document_count": 132,
        "donor_row_reuse_multiplicities": donor_reuse.tolist(),
        "donor_row_reuse_multiplicities_sha256": logical_sha256(donor_reuse.tolist()),
        "global_role_disjointness_sha256": provenance["disjointness_sha256"],
    }
    return {**body, "sha256": logical_sha256(body)}


def verify_row_binding(binding: Mapping[str, Any]) -> None:
    if row_binding() != dict(binding):
        raise RuntimeError("family-F row binding drift")


def protocol() -> dict[str, Any]:
    """Machine-readable projection of the frozen amendment's numerical contract."""

    return {
        "fit_role": ROLE,
        "evaluation_roles_importable": False,
        "model_tokens": 256,
        "scored_positions_half_open": [64, 256],
        "logical_batch_rows": 8,
        "microbatch_rows": 2,
        "microbatches_per_logical_step": 4,
        "prefilter": 1024,
        "score_budget": 512,
        "published_budgets": [256, 512],
        "score_arms": [
            "teacher", "teacher_row_reversal", "teacher_document_derangement",
        ],
        "promotive_family": "uncalibrated_real_teacher_F",
        "score_epochs": 8,
        "score_logical_steps_per_arm": 480,
        "score_learning_rate": 0.02,
        "score_adam_betas": [0.9, 0.999],
        "score_adam_epsilon": 1e-8,
        "gradient_clip_norm": 1.0,
        "projection_iterations": core.SCORE_PROJECTION_ITERATIONS,
        "projection_dtype": "torch.float64",
        "model_program_dtype": "torch.float32",
        "affine_parameter_adam_dtype": "torch.float64",
        "decoder_relative_ridge": 1e-6,
        "affine_diagnostic_arms": [
            "teacher_F_k512", "family_A_k512", "random_k512",
            "same_support_permuted_cross_k512",
        ],
        "affine_epochs": 4,
        "affine_logical_steps_per_arm": 240,
        "affine_learning_rate": 0.005,
        "affine_fitted_coordinates_per_arm": 1153,
        "affine_fitted_coordinates_total": 4612,
        "total_logical_optimizer_steps": 2400,
        "total_two_row_backwards": 9600,
        "postfit_reporting_batches": 60,
        "postfit_reporting_student_arms_per_batch": 18,
        "total_prefix_calls": 2940,
        "total_teacher_suffix_calls": 2460,
        "total_student_suffix_calls": 10680,
        "total_suffix_returns": 13140,
        "outer_full_model_replays": 1,
        "total_raw_logit_returns": 13141,
        "total_attention_mlp_calls_sites_0_3_each": 2941,
        "total_attention_mlp_calls_sites_4_17_each": 13141,
        "student_native_mlp3_calls": 0,
        "wall_clock_seconds_max": 2700,
        "cuda_allocated_bytes_max": 30 * 1024 ** 3,
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
        "authorized_for_fit_execution": False,
    }


def verified_draft_authority() -> dict[str, Any]:
    """Build a verified, explicitly nonauthorizing draft for the future runner.

    A numerical runner and its tests are intentionally absent from this source closure,
    so this object cannot authorize row deserialization or model execution.
    """

    require_pristine_namespace()
    source = source_closure()
    prior = prior_artifact_binding()
    rows = row_binding()
    checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
    verify_frozen_inputs(source, prior, rows, checkpoint)
    body = {
        "schema": "block3_consequence_family_f_v1_authority_draft",
        "status": "nonauthoritative_lifecycle_scaffold_no_numerical_runner_in_source_closure",
        "authorized_for_fit_execution": False,
        "source_closure": dict(source),
        "prior_artifact_binding": dict(prior),
        "row_binding": dict(rows),
        "checkpoint": asdict(checkpoint),
        "protocol": protocol(),
    }
    return {**body, "authority_sha256": logical_sha256(body)}


def verify_frozen_inputs(
    source: Mapping[str, Any], prior: Mapping[str, Any], rows: Mapping[str, Any],
    checkpoint: facade.CheckpointReceipt,
) -> None:
    verify_source_closure(source)
    verify_prior_artifact_binding(prior)
    verify_row_binding(rows)
    if facade.validate_snapshot(verify_weights_sha256=True) != checkpoint:
        raise RuntimeError("family-F checkpoint binding drift")
