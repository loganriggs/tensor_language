"""Receipt-last lifecycle for the frozen E4 terminal-copy selection screen.

The sole public execution entry point freezes all scientific and artifact choices,
loads only the selection role, reconstructs its labels independently, runs one native
and eight physical counterfactual trajectories per batch, and publishes either a
passer or a scientific-negative receipt last.  Final/OOD roles are protected bytes
but are never deserialized here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
import terminal_copy_fit_mean_lifecycle as fit_life
import terminal_copy_fit_mean_recovery_v3 as fit_v3
import terminal_copy_induction_v1 as contract
import terminal_copy_selection_fit_parent as fit_parent
from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher
from terminal_copy_fit_head_means import FitHeadMeanBank, _document_digest
from terminal_copy_selection_owner import (
    MergedSelectionBatches,
    MergedSyntheticBatches,
    SelectionBatchOwner,
    SelectionBatchResult,
    SyntheticBatchResult,
    SyntheticSelectionBatchOwner,
    merge_selection_batches,
    merge_synthetic_batches,
)
from terminal_copy_streaming_statistics import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CELL_NAMES,
    COLLATERAL_LIMIT,
    FROZEN_CANDIDATES,
    CandidateEffects,
    DocumentCellSums,
    SelectionResult,
    pooled_effects,
    simultaneous_selection_bootstrap,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BQ = ROOT / "basis_aligned" / "bilinear_quotient"

ROW_RECEIPT = BQ / "terminal_copy_induction_v2_rows_receipt.json"
ROW_RECEIPT_SHA256 = "aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8"
SELECTION_PAYLOAD_SHA256 = "cc8e1c3e468b7bc249e0cf8fc00640955ae17251c7f0c7640350f65a86202cac"
FIT_FREQUENCIES_SHA256 = "7ba995e6bcfa2704cc4c2220dfdc8bd5caea53b976359c6b86cb5e14ba7e4c9a"
SELECTION_ROWS_SHA256 = "625258ae1128823194fd27c94c241bd197dfd8daba77cfa2d1a0156ae1daaf8a"
SELECTION_POSITIVE_SHA256 = "31455e7186ed198e7cfc2fed52cd59e9b2a535f5445f825fd91ee295526243e4"
SELECTION_NEGATIVE_SHA256 = "ebce79e3303d5355e513f304d982ba2fc5b8e4d2fe3eba4775187fe5d2989ab7"
FIT_QUERY_FREQUENCY_SHA256 = "5c851e2f76e41f69d85b326bfddfffb12cdf1db4aa5460a981f645bc1f7aa59c"
FIT_TARGET_FREQUENCY_SHA256 = "66a536e0e77e841702fb46651c8ec697140b259e5d1b01bdd91fdc354a8c5cd0"
SELECTION_SYNTHETIC_Y_SHA256 = "4165e59ad701bdcd4221ea18d69ca5458da168fc066f80d7720cf51fd299340b"
SELECTION_SYNTHETIC_Z_SHA256 = "32b95ba0ee304cab0a275717030a57244d952c95e6f2aa0d6df4858088382fee"

PREREGISTRATION = HERE / "TERMINAL_COPY_INDUCTION_V1_PREREGISTRATION.md"
AMENDMENT = HERE / "TERMINAL_COPY_INDUCTION_V1_SCREENING_AMENDMENT.md"
RULING = HERE / "TERMINAL_COPY_SELECTION_V1_EXECUTION_RULING.md"
EXPOSURE_ERRATUM = HERE / "TERMINAL_COPY_SELECTION_INPUT_EXPOSURE_ERRATUM.md"
ADAPTER_RECEIPT = HERE / "terminal_copy_attention_checkpoint_check_v3_receipt.json"
ADAPTER_RESULT = HERE / "terminal_copy_attention_checkpoint_check_v3_result.json"

AUTHORITY = HERE / "terminal_copy_selection_v1_authority.json"
LEDGER = HERE / "terminal_copy_selection_v1_ledger.json"
RESULT = HERE / "terminal_copy_selection_v1_result.json"
MANIFEST = HERE / "terminal_copy_selection_v1_manifest.json"
PASSER_RECEIPT = HERE / "terminal_copy_selection_v1_passer_receipt.json"
NEGATIVE_RECEIPT = HERE / "terminal_copy_selection_v1_negative_receipt.json"
FAILURE = HERE / "terminal_copy_selection_v1_failure.json"
LOCK = Path("/workspace/runs/.terminal_copy_selection_v1.lock")
AUDIT = HERE / "terminal_copy_selection_lifecycle_v1_independent_audit.json"

NATURAL_DOCUMENTS = 192
NATURAL_BATCH_SIZE = 4
NATURAL_BATCHES = 48
SYNTHETIC_PAIRS = 32
SYNTHETIC_ROWS = 64
SYNTHETIC_BATCH_SIZE = 4
SYNTHETIC_BATCHES = 16
CRITICAL_INDEX = 9_499
COORDINATE_COUNT = 24
SYNTHETIC_POSITION_TEMPLATES = ((8, 32, 80), (12, 44, 96), (20, 52, 128), (28, 60, 160))

SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/TERMINAL_COPY_INDUCTION_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/TERMINAL_COPY_INDUCTION_V1_SCREENING_AMENDMENT.md",
    "basis_aligned/polynomial_causal/TERMINAL_COPY_SELECTION_V1_EXECUTION_RULING.md",
    "basis_aligned/polynomial_causal/TERMINAL_COPY_SELECTION_INPUT_EXPOSURE_ERRATUM.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/terminal_copy_attention_adapter.py",
    "basis_aligned/polynomial_causal/terminal_copy_attention_dispatcher.py",
    "basis_aligned/polynomial_causal/terminal_copy_attention_owner.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_head_means.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_lifecycle.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_recovery_v3.py",
    "basis_aligned/polynomial_causal/terminal_copy_induction_v1.py",
    "basis_aligned/polynomial_causal/terminal_copy_selection_fit_parent.py",
    "basis_aligned/polynomial_causal/terminal_copy_selection_owner.py",
    "basis_aligned/polynomial_causal/terminal_copy_selection_lifecycle.py",
    "basis_aligned/polynomial_causal/terminal_copy_streaming_statistics.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_selection_lifecycle.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
)

PROTECTED_PATHS = (
    ROW_RECEIPT,
    Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache_terminal_copy_induction_v2/selection_natural.pt"),
    Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache_terminal_copy_induction_v2/fit_token_frequencies.pt"),
    Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache_terminal_copy_induction_v2/final_natural.pt"),
    Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache_terminal_copy_induction_v2/ood_code.pt"),
    fit_v3.AUTHORITY, fit_v3.BANK, fit_v3.RESULT, fit_v3.MANIFEST, fit_v3.RECEIPT,
    ADAPTER_RECEIPT, ADAPTER_RESULT,
    facade.DEFAULT_SNAPSHOT / "config.json",
    facade.DEFAULT_SNAPSHOT / "pytorch_model.bin",
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
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def stable_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    before = file_sha256(path)
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    after = file_sha256(path)
    value = json.loads(raw)
    if (
        before != raw_sha256 or after != before
        or (expected_sha256 is not None and before != expected_sha256)
        or not isinstance(value, dict)
    ):
        raise RuntimeError(f"selection JSON changed while loading: {path}")
    return value


def output_namespace() -> tuple[Path, ...]:
    return (
        AUTHORITY, LEDGER, RESULT, MANIFEST, PASSER_RECEIPT,
        NEGATIVE_RECEIPT, FAILURE, LOCK,
    )


def expected_outputs() -> dict[str, str]:
    return {
        "authority": str(AUTHORITY), "ledger": str(LEDGER),
        "result": str(RESULT), "manifest": str(MANIFEST),
        "passer_receipt": str(PASSER_RECEIPT),
        "negative_receipt": str(NEGATIVE_RECEIPT),
        "failure": str(FAILURE), "lock": str(LOCK),
    }


def protected_snapshot(paths: Sequence[Path] = PROTECTED_PATHS) -> dict[str, str | None]:
    return {str(path): file_sha256(path) if path.is_file() else None for path in paths}


class RunClaim(NamedTuple):
    descriptor: int
    inode: int
    nonce: str


def acquire_claim() -> RunClaim:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    try:
        descriptor = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("selection namespace is locked") from error
    try:
        os.write(descriptor, (nonce + "\n").encode())
        os.fsync(descriptor)
        return RunClaim(descriptor, os.fstat(descriptor).st_ino, nonce)
    except BaseException:
        os.close(descriptor)
        LOCK.unlink(missing_ok=True)
        raise


def require_claim(claim: RunClaim) -> None:
    if (
        not isinstance(claim, RunClaim) or not LOCK.is_file()
        or LOCK.stat().st_ino != claim.inode or LOCK.read_text() != claim.nonce + "\n"
    ):
        raise RuntimeError("selection execution lock ownership changed")


def release_claim(claim: RunClaim) -> None:
    try:
        if LOCK.exists() and LOCK.stat().st_ino == claim.inode:
            LOCK.unlink()
    finally:
        os.close(claim.descriptor)


def require_pristine_namespace() -> None:
    spent = [str(path) for path in output_namespace() if path.exists()]
    if spent:
        raise RuntimeError(f"selection output namespace is spent: {spent}")


def require_pristine_execution_namespace() -> None:
    if not AUTHORITY.is_file():
        raise RuntimeError("selection execution authority is absent")
    spent = [
        str(path) for path in (
            LEDGER, RESULT, MANIFEST, PASSER_RECEIPT, NEGATIVE_RECEIPT, FAILURE, LOCK,
        ) if path.exists()
    ]
    if spent:
        raise RuntimeError(f"selection execution namespace is spent: {spent}")


def source_closure() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"selection source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if not (ROOT / relative).is_file() or file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"live selection source differs from commit: {relative}")
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
        raise RuntimeError("selection source closure is malformed")
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
            raise RuntimeError(f"selection source drift: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(body["commit"]), "origin/main"],
        cwd=ROOT, check=True,
    )


def row_binding() -> dict[str, Any]:
    receipt = stable_json(ROW_RECEIPT, ROW_RECEIPT_SHA256)
    entry = receipt.get("entries", {}).get("selection_natural", {})
    frequencies = receipt.get("fit_token_frequencies", {})
    license_ = receipt.get("role_licenses", {}).get("selection_natural", {})
    support = receipt.get("support_census", {}).get("selection_natural", {})
    expected_support = {
        "positive_positions": 303, "matched_negative_positions": 303,
        "positive_documents": 137, "matched_negative_documents": 154,
        "passed": True,
    }
    if (
        receipt.get("receipt_kind") != "terminal_copy_induction_v2_rows"
        or receipt.get("status") != "frozen_before_any_terminal_copy_model_forward"
        or receipt.get("authorized_for_candidate_or_threshold_selection") is not False
        or receipt.get("authorized_for_scored_experiments") is not False
        or receipt.get("summary", {}).get("roles", {}).get("selection_natural") != 192
        or receipt.get("summary", {}).get("synthetic_pairs", {}).get("selection_natural") != 32
        or support != expected_support
        or license_ != {
            "authorized_use": "candidate_selection_only",
            "requires_receipt": "terminal_copy_induction_v1_fit_means_receipt",
        }
        or entry.get("file_sha256") != SELECTION_PAYLOAD_SHA256
        or entry.get("rows_tensor_sha256") != SELECTION_ROWS_SHA256
        or entry.get("copy_positive_mask_sha256") != SELECTION_POSITIVE_SHA256
        or entry.get("copy_matched_negative_mask_sha256") != SELECTION_NEGATIVE_SHA256
        or entry.get("query_to_y_tensor_sha256") != SELECTION_SYNTHETIC_Y_SHA256
        or entry.get("query_to_z_tensor_sha256") != SELECTION_SYNTHETIC_Z_SHA256
        or frequencies.get("file_sha256") != FIT_FREQUENCIES_SHA256
        or frequencies.get("query_tensor_sha256") != FIT_QUERY_FREQUENCY_SHA256
        or frequencies.get("target_tensor_sha256") != FIT_TARGET_FREQUENCY_SHA256
        or not isinstance(entry.get("path"), str)
        or not isinstance(frequencies.get("path"), str)
        or file_sha256(Path(entry["path"])) != SELECTION_PAYLOAD_SHA256
        or file_sha256(Path(frequencies["path"])) != FIT_FREQUENCIES_SHA256
    ):
        raise RuntimeError("selection row receipt semantics changed")
    body = {
        "receipt_path": str(ROW_RECEIPT),
        "receipt_sha256": ROW_RECEIPT_SHA256,
        "role": "selection_natural",
        "container_path": entry["path"],
        "container_sha256": SELECTION_PAYLOAD_SHA256,
        "rows_tensor_sha256": SELECTION_ROWS_SHA256,
        "positive_mask_sha256": SELECTION_POSITIVE_SHA256,
        "matched_negative_mask_sha256": SELECTION_NEGATIVE_SHA256,
        "query_to_y_tensor_sha256": SELECTION_SYNTHETIC_Y_SHA256,
        "query_to_z_tensor_sha256": SELECTION_SYNTHETIC_Z_SHA256,
        "fit_frequencies_path": frequencies["path"],
        "fit_frequencies_sha256": FIT_FREQUENCIES_SHA256,
        "fit_query_frequency_sha256": FIT_QUERY_FREQUENCY_SHA256,
        "fit_target_frequency_sha256": FIT_TARGET_FREQUENCY_SHA256,
        "support_census": expected_support,
        "authorized_use": "candidate_selection_only",
        "requires_completed_fit_prerequisite": True,
        "schema_only_pre_authority_container_exposure": True,
        "no_selection_values_or_model_outcomes_observed_in_exposure": True,
        "pristine_container_secrecy_claimed": False,
        "final_ood_deserialization_authorized": False,
    }
    return {**body, "sha256": logical_sha256(body)}


def protocol() -> dict[str, Any]:
    return {
        "role": "selection_natural",
        "natural_documents": NATURAL_DOCUMENTS,
        "natural_batch_size": NATURAL_BATCH_SIZE,
        "natural_batches": NATURAL_BATCHES,
        "synthetic_pairs": SYNTHETIC_PAIRS,
        "synthetic_rows": SYNTHETIC_ROWS,
        "synthetic_batch_size": SYNTHETIC_BATCH_SIZE,
        "synthetic_batches": SYNTHETIC_BATCHES,
        "native_plus_candidate_forwards_per_batch": 9,
        "total_outer_forwards": 576,
        "candidates": list(FROZEN_CANDIDATES),
        "candidate_plans": {
            candidate: [[layer, list(heads)] for layer, heads in PhysicalCandidateDispatcher.plan(candidate)]
            for candidate in FROZEN_CANDIDATES
        },
        "all_mlps_native": True,
        "late_mlp_screen_omitted": True,
        "intervention_arithmetic": "(native_full - selected_heads) + fit_position_mean",
        "intervention_positions": [0, 256],
        "score_positions": [64, 256],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_coordinates": COORDINATE_COUNT,
        "critical_index_zero_based": CRITICAL_INDEX,
        "selection_coordinates": ["tau_positive", "specificity", "collateral_margin"],
        "collateral_limit": COLLATERAL_LIMIT,
        "pass_rule": "all three simultaneous lower bounds strictly positive",
        "winner_rule": "largest specificity lower bound then lexicographic candidate",
        "synthetic_role": "descriptive_only_no_selection_credit",
        "final_ood_opening": "passer_receipt_only",
        "artifact_order": ["authority", "ledger", "result", "manifest", "passer_or_negative_receipt_last"],
    }


def adapter_binding() -> dict[str, Any]:
    binding = fit_life.adapter_binding()
    if (
        binding.get("receipt_sha256") != fit_life.ADAPTER_RECEIPT_SHA256
        or binding.get("result_sha256") != fit_life.ADAPTER_RESULT_SHA256
        or binding.get("checkpoint_weights_sha256") != facade.WEIGHTS_SHA256
        or binding.get("native_full_bit_equal") is not True
        or binding.get("value_bus_bit_equal") is not True
    ):
        raise RuntimeError("selection adapter binding changed")
    return binding


def verified_draft_authority() -> dict[str, Any]:
    require_pristine_namespace()
    body = {
        "schema": "terminal_copy_selection_v1_authority_draft",
        "status": "nonauthoritative_until_source_commit_and_independent_audit",
        "source_closure": source_closure(),
        "row_binding": row_binding(),
        "fit_parent_binding": fit_parent.replay_fit_parent(),
        "adapter_binding": adapter_binding(),
        "checkpoint": asdict(facade.validate_snapshot(verify_weights_sha256=True)),
        "protocol": protocol(),
        "outputs": expected_outputs(),
        "authorized_for_selection_execution": False,
        "authorized_for_final_ood": False,
    }
    return {**body, "authority_sha256": logical_sha256(body)}


def validate_canonical_audit(path: Path = AUDIT) -> dict[str, Any]:
    if path.resolve() != AUDIT.resolve():
        raise RuntimeError("selection audit is not the canonical audit path")
    audit = stable_json(path)
    expected_reviewed = {
        "basis_aligned/polynomial_causal/terminal_copy_selection_lifecycle.py":
            file_sha256(Path(__file__).resolve()),
        "basis_aligned/polynomial_causal/test_terminal_copy_selection_lifecycle.py":
            file_sha256(HERE / "test_terminal_copy_selection_lifecycle.py"),
        "basis_aligned/polynomial_causal/terminal_copy_selection_owner.py":
            file_sha256(HERE / "terminal_copy_selection_owner.py"),
        "basis_aligned/polynomial_causal/test_terminal_copy_selection_owner.py":
            file_sha256(HERE / "test_terminal_copy_selection_owner.py"),
        "basis_aligned/polynomial_causal/terminal_copy_selection_fit_parent.py":
            file_sha256(HERE / "terminal_copy_selection_fit_parent.py"),
        "basis_aligned/polynomial_causal/TERMINAL_COPY_SELECTION_V1_EXECUTION_RULING.md":
            file_sha256(RULING),
        "basis_aligned/polynomial_causal/TERMINAL_COPY_SELECTION_INPUT_EXPOSURE_ERRATUM.md":
            file_sha256(EXPOSURE_ERRATUM),
    }
    if (
        set(audit) != {
            "schema", "status", "approved", "outcome_access", "reviewer",
            "reviewed_source_sha256s", "focused_tests", "remaining_launch_blockers",
        }
        or audit.get("schema") != "terminal_copy_selection_lifecycle_independent_audit_v1"
        or audit.get("status") != "approved_outcome_blind_selection_infrastructure"
        or audit.get("approved") is not True
        or audit.get("outcome_access") is not False
        or audit.get("reviewer") != "independent_artifact_audit_agent"
        or audit.get("reviewed_source_sha256s") != expected_reviewed
        or not isinstance(audit.get("focused_tests"), Mapping)
        or audit["focused_tests"].get("passed") is not True
        or type(audit["focused_tests"].get("count")) is not int
        or audit["focused_tests"]["count"] <= 0
        or audit.get("remaining_launch_blockers") != []
    ):
        raise RuntimeError("selection canonical audit semantics or reviewed bytes changed")
    return audit


def freeze_execution_authority(independent_audit_path: Path = AUDIT) -> dict[str, Any]:
    """Publish the sole selection authority before any further label/model access."""

    require_pristine_namespace()
    audit = validate_canonical_audit(independent_audit_path)
    body = {
        "schema": "terminal_copy_selection_v1_authority",
        "status": "frozen_before_selection_value_or_model_outcome_access_after_disclosed_schema_exposure",
        "source_closure": source_closure(),
        "row_binding": row_binding(),
        "fit_parent_binding": fit_parent.replay_fit_parent(),
        "adapter_binding": adapter_binding(),
        "checkpoint": asdict(facade.validate_snapshot(verify_weights_sha256=True)),
        "protocol": protocol(),
        "outputs": expected_outputs(),
        "protected_paths": [str(path) for path in PROTECTED_PATHS],
        "independent_audit": {
            "approved": audit["approved"],
            "outcome_access": audit["outcome_access"],
            "path": str(AUDIT.resolve()),
            "sha256": file_sha256(AUDIT),
        },
        "authorized_for_selection_execution": True,
        "authorized_for_final_ood": False,
        "fit_receipt_self_authorizes_selection": False,
        "selection_authority_independently_licenses_exact_fit_bank": True,
        "amendment_governs_conflicts": True,
        "pristine_selection_container_secrecy_claimed": False,
    }
    authority = {**body, "authority_sha256": logical_sha256(body)}
    create_only_json(AUTHORITY, authority)
    validate_execution_authority(authority)
    return authority


def validate_execution_authority(authority: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema", "status", "source_closure", "row_binding", "fit_parent_binding",
        "adapter_binding", "checkpoint", "protocol", "outputs", "protected_paths",
        "independent_audit", "authorized_for_selection_execution",
        "authorized_for_final_ood", "fit_receipt_self_authorizes_selection",
        "selection_authority_independently_licenses_exact_fit_bank",
        "amendment_governs_conflicts", "pristine_selection_container_secrecy_claimed",
        "authority_sha256",
    }
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if (
        set(authority) != expected_keys
        or authority.get("schema") != "terminal_copy_selection_v1_authority"
        or authority.get("status")
            != "frozen_before_selection_value_or_model_outcome_access_after_disclosed_schema_exposure"
        or authority.get("authorized_for_selection_execution") is not True
        or authority.get("authorized_for_final_ood") is not False
        or authority.get("fit_receipt_self_authorizes_selection") is not False
        or authority.get("selection_authority_independently_licenses_exact_fit_bank") is not True
        or authority.get("amendment_governs_conflicts") is not True
        or authority.get("pristine_selection_container_secrecy_claimed") is not False
        or logical_sha256(body) != authority.get("authority_sha256")
        or authority.get("row_binding") != row_binding()
        or authority.get("fit_parent_binding") != fit_parent.replay_fit_parent()
        or authority.get("adapter_binding") != adapter_binding()
        or authority.get("checkpoint")
            != asdict(facade.validate_snapshot(verify_weights_sha256=True))
        or authority.get("protocol") != protocol()
        or authority.get("outputs") != expected_outputs()
        or authority.get("protected_paths") != [str(path) for path in PROTECTED_PATHS]
    ):
        raise RuntimeError("selection execution authority identity changed")
    audit = authority.get("independent_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("approved") is not True
        or audit.get("outcome_access") is not False
        or audit.get("path") != str(AUDIT.resolve())
        or audit.get("sha256") != file_sha256(AUDIT)
    ):
        raise RuntimeError("selection independent audit is absent or changed")
    validate_canonical_audit(AUDIT)
    verify_source_closure(authority["source_closure"])
    if not AUTHORITY.is_file() or stable_json(AUTHORITY) != dict(authority):
        raise RuntimeError("selection authority file differs from supplied authority")


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
