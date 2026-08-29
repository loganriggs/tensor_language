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


def _semantic_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left) and torch.is_tensor(right)
            and left.dtype == right.dtype and left.shape == right.shape
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_semantic_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_semantic_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _copy_cells_payload(cells: contract.CopyCells) -> dict[str, Any]:
    return {
        "all_positive": cells.all_positive,
        "positive": cells.positive,
        "matched_negative": cells.matched_negative,
        "off_target": cells.off_target,
        "pair_indices": cells.pair_indices,
        "unmatched_positive_count": cells.unmatched_positive_count,
        "negative_candidate_count": cells.negative_candidate_count,
        "eligible_stratum_count": cells.eligible_stratum_count,
        "excluded_low_document_stratum_count": cells.excluded_low_document_stratum_count,
    }


def _support_census(cells: contract.CopyCells) -> dict[str, Any]:
    positive = torch.nonzero(cells.positive, as_tuple=False)
    negative = torch.nonzero(cells.matched_negative, as_tuple=False)
    return {
        "positive_positions": len(positive),
        "matched_negative_positions": len(negative),
        "positive_documents": len(set(int(value) for value in positive[:, 0])),
        "matched_negative_documents": len(set(int(value) for value in negative[:, 0])),
        "passed": (
            len(positive) >= 48 and len(negative) >= 48
            and len(set(int(value) for value in positive[:, 0])) >= 24
            and len(set(int(value) for value in negative[:, 0])) >= 24
        ),
    }


@dataclass(frozen=True)
class SelectionInputs:
    rows: torch.Tensor
    masks: Mapping[str, torch.Tensor]
    ordered_document_ids: tuple[str, ...]
    ordered_document_ids_sha256: str
    selection_file_sha256: str
    frequencies_file_sha256: str
    synthetic_rows: torch.Tensor
    synthetic_item_ids: tuple[str, ...]
    synthetic_query_positions: tuple[int, ...]
    synthetic_successor_y: tuple[int, ...]
    synthetic_successor_z: tuple[int, ...]
    expected_support_sha256s: Mapping[str, Mapping[str, str]]


def _load_selection_inputs(
    authority: Mapping[str, Any], claim: RunClaim,
) -> SelectionInputs:
    """Load only selection-role bytes and independently reconstruct every mask."""

    require_claim(claim)
    validate_execution_authority(authority)
    binding = authority["row_binding"]
    selection_path = Path(binding["container_path"])
    frequencies_path = Path(binding["fit_frequencies_path"])
    before_selection = file_sha256(selection_path)
    before_frequencies = file_sha256(frequencies_path)
    if (
        before_selection != SELECTION_PAYLOAD_SHA256
        or before_frequencies != FIT_FREQUENCIES_SHA256
    ):
        raise RuntimeError("selection role or fit-frequency bytes changed before load")
    payload = torch.load(selection_path, map_location="cpu", weights_only=True)
    frequency_payload = torch.load(frequencies_path, map_location="cpu", weights_only=True)
    after_selection = file_sha256(selection_path)
    after_frequencies = file_sha256(frequencies_path)
    if before_selection != after_selection or before_frequencies != after_frequencies:
        raise RuntimeError("selection role or fit-frequency bytes changed during load")
    expected_fields = {
        "rows", "records", "synthetic", "synthetic_token_banks",
        "synthetic_position_templates", "copy_cells",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RuntimeError("selection role payload schema changed")
    if not isinstance(frequency_payload, dict) or set(frequency_payload) != {"query", "target"}:
        raise RuntimeError("fit-frequency payload schema changed")
    rows = payload["rows"]
    records = payload["records"]
    query_frequency = frequency_payload["query"]
    target_frequency = frequency_payload["target"]
    if (
        not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long
        or tuple(rows.shape) != (NATURAL_DOCUMENTS, 257)
        or tensor_sha256(rows) != SELECTION_ROWS_SHA256
        or not isinstance(records, list) or len(records) != NATURAL_DOCUMENTS
        or not torch.is_tensor(query_frequency) or query_frequency.device.type != "cpu"
        or query_frequency.dtype != torch.long or tuple(query_frequency.shape) != (50_257,)
        or tensor_sha256(query_frequency) != FIT_QUERY_FREQUENCY_SHA256
        or not torch.is_tensor(target_frequency) or target_frequency.device.type != "cpu"
        or target_frequency.dtype != torch.long or tuple(target_frequency.shape) != (50_257,)
        or tensor_sha256(target_frequency) != FIT_TARGET_FREQUENCY_SHA256
    ):
        raise RuntimeError("selection rows or fit frequencies changed semantically")
    documents: list[str] = []
    for index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or record.get("role") != "selection_natural"
            or record.get("role_row_index") != index
            or not isinstance(record.get("document_id"), str)
            or not record["document_id"]
        ):
            raise RuntimeError("selection record topology changed")
        documents.append(record["document_id"])
    document_ids = tuple(documents)
    if len(set(document_ids)) != NATURAL_DOCUMENTS:
        raise RuntimeError("selection documents are not unique")
    frequencies = contract.FitTokenFrequencies(
        query=query_frequency, target=target_frequency,
    )
    cells = contract.build_copy_cells(rows, frequencies, document_ids)
    if (
        tensor_sha256(cells.positive) != SELECTION_POSITIVE_SHA256
        or tensor_sha256(cells.matched_negative) != SELECTION_NEGATIVE_SHA256
        or _support_census(cells) != binding["support_census"]
        or not _semantic_equal(payload["copy_cells"], _copy_cells_payload(cells))
    ):
        raise RuntimeError("independent selection mask reconstruction changed")
    masks = {
        "positive": cells.positive.clone(),
        "matched_negative": cells.matched_negative.clone(),
        "off_target": cells.off_target.clone(),
    }
    expected_support: dict[str, dict[str, str]] = {}
    from terminal_copy_streaming_statistics import _support_digest
    for index, document in enumerate(document_ids):
        expected_support[document] = {
            cell: _support_digest(rows[index], masks[cell][index]) for cell in CELL_NAMES
        }

    synthetic = payload["synthetic"]
    banks = payload["synthetic_token_banks"]
    templates = payload["synthetic_position_templates"]
    if (
        not isinstance(synthetic, Mapping) or set(synthetic) != {"query_to_y", "query_to_z"}
        or not torch.is_tensor(synthetic["query_to_y"])
        or not torch.is_tensor(synthetic["query_to_z"])
        or synthetic["query_to_y"].device.type != "cpu"
        or synthetic["query_to_z"].device.type != "cpu"
        or synthetic["query_to_y"].dtype != torch.long
        or synthetic["query_to_z"].dtype != torch.long
        or tuple(synthetic["query_to_y"].shape) != (SYNTHETIC_PAIRS, 257)
        or tuple(synthetic["query_to_z"].shape) != (SYNTHETIC_PAIRS, 257)
        or tensor_sha256(synthetic["query_to_y"]) != SELECTION_SYNTHETIC_Y_SHA256
        or tensor_sha256(synthetic["query_to_z"]) != SELECTION_SYNTHETIC_Z_SHA256
        or tuple(tuple(value) for value in templates) != SYNTHETIC_POSITION_TEMPLATES
        or not isinstance(banks, list) or len(banks) != SYNTHETIC_PAIRS
    ):
        raise RuntimeError("selection synthetic payload changed")
    query_positions: list[int] = []
    successor_y: list[int] = []
    successor_z: list[int] = []
    alternating_rows: list[torch.Tensor] = []
    item_ids: list[str] = []
    for index, bank in enumerate(banks):
        if (
            not isinstance(bank, list) or len(bank) != 4
            or any(type(token) is not int or not 0 <= token < 50_256 for token in bank)
            or len(set(bank)) != 4
        ):
            raise RuntimeError("selection synthetic token bank changed")
        first, reciprocal, query = SYNTHETIC_POSITION_TEMPLATES[index % 4]
        replay = contract.build_synthetic_association_crossover(
            tuple(int(value) for value in rows[index]),
            first_query_position=first,
            reciprocal_position=reciprocal,
            query_position=query,
            query_token=bank[0], reciprocal_query=bank[1],
            successor_y=bank[2], successor_z=bank[3],
        )
        if (
            not torch.equal(replay.query_to_y, synthetic["query_to_y"][index])
            or not torch.equal(replay.query_to_z, synthetic["query_to_z"][index])
        ):
            raise RuntimeError("selection synthetic crossover failed independent replay")
        alternating_rows.extend((replay.query_to_y, replay.query_to_z))
        query_positions.append(query)
        successor_y.append(bank[2])
        successor_z.append(bank[3])
        item_ids.append(f"selection_synthetic_{index:03d}")

    # Clone every retained tensor and destroy the label-bearing parent container.
    output = SelectionInputs(
        rows=rows.clone(), masks={key: value.clone() for key, value in masks.items()},
        ordered_document_ids=document_ids,
        ordered_document_ids_sha256=_document_digest(document_ids),
        selection_file_sha256=after_selection,
        frequencies_file_sha256=after_frequencies,
        synthetic_rows=torch.stack(alternating_rows).contiguous(),
        synthetic_item_ids=tuple(item_ids),
        synthetic_query_positions=tuple(query_positions),
        synthetic_successor_y=tuple(successor_y),
        synthetic_successor_z=tuple(successor_z),
        expected_support_sha256s=expected_support,
    )
    del payload, frequency_payload, rows, records, cells, synthetic, query_frequency, target_frequency
    if (
        file_sha256(selection_path) != before_selection
        or file_sha256(frequencies_path) != before_frequencies
    ):
        raise RuntimeError("selection inputs changed after semantic reconstruction")
    return output


def _load_fit_bank(authority: Mapping[str, Any]) -> FitHeadMeanBank:
    """Replay the exact five-file fit parent and return a cloned semantic bank."""

    binding = fit_parent.replay_fit_parent()
    if binding != authority["fit_parent_binding"]:
        raise RuntimeError("selection fit-parent binding changed before bank load")
    names = ("AUTHORITY", "BANK", "RESULT", "MANIFEST", "RECEIPT", "FAILURE", "LOCK")
    original_outputs = {name: getattr(fit_life, name) for name in names}
    original_sources = fit_life.SOURCE_PATHS
    original_protected = fit_life.PROTECTED_PATHS
    original_snapshot = fit_life.protected_snapshot
    try:
        fit_v3.configure()
        bank = fit_life.load_bank_semantically(
            fit_v3.BANK, binding["fit_authority_sha256"], require_production=True,
        )
        if (
            bank.document_count != binding["document_count"]
            or bank.master_means_sha256 != binding["master_means_sha256"]
            or bank.runtime_means_sha256 != binding["runtime_means_sha256"]
            or bank.ordered_document_ids_sha256 != binding["ordered_document_ids_sha256"]
        ):
            raise RuntimeError("selection fit bank differs from licensed parent")
        return bank
    finally:
        for name, value in original_outputs.items():
            setattr(fit_life, name, value)
        fit_life.SOURCE_PATHS = original_sources
        fit_life.PROTECTED_PATHS = original_protected
        fit_life.protected_snapshot = original_snapshot


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = tuple(model.named_parameters()) + tuple(model.named_buffers())
    for name, value in state:
        tensor = value.detach().to("cpu").contiguous()
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_exact_support(
    merged: MergedSelectionBatches, inputs: SelectionInputs,
) -> None:
    if merged.ordered_document_ids != inputs.ordered_document_ids:
        raise RuntimeError("selection merged document order changed")
    index_by_document = {
        document: index for index, document in enumerate(inputs.ordered_document_ids)
    }
    for candidate in FROZEN_CANDIDATES:
        for document in inputs.ordered_document_ids:
            row_index = index_by_document[document]
            for cell in CELL_NAMES:
                value = merged.ledgers[candidate][document][cell]
                if (
                    value.support_sha256 != inputs.expected_support_sha256s[document][cell]
                    or value.n != int(inputs.masks[cell][row_index].sum())
                ):
                    raise RuntimeError("selection sufficient statistics use unauthorized support")


def _closure_payload(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _ledger_payload(
    authority_sha256: str,
    natural: MergedSelectionBatches,
    synthetic: MergedSyntheticBatches,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for candidate in FROZEN_CANDIDATES:
        candidates[candidate] = {
            document: {
                cell: asdict(natural.ledgers[candidate][document][cell])
                for cell in CELL_NAMES
            }
            for document in natural.ordered_document_ids
        }
    synthetic_effects = {
        candidate: [asdict(effect) for effect in synthetic.effects[candidate]]
        for candidate in FROZEN_CANDIDATES
    }
    return {
        "schema": "terminal_copy_selection_v1_ledger",
        "authority_sha256": authority_sha256,
        "ordered_document_ids": list(natural.ordered_document_ids),
        "ordered_document_ids_sha256": _document_digest(natural.ordered_document_ids),
        "candidates": candidates,
        "natural_batch_closures": [
            _closure_payload(closure) for closure in natural.batch_closures
        ],
        "synthetic_item_ids": list(synthetic.ordered_item_ids),
        "synthetic_effects": synthetic_effects,
        "synthetic_batch_closures": [
            _closure_payload(closure) for closure in synthetic.batch_closures
        ],
        "raw_logits_published": False,
    }


def _deserialize_ledger(
    payload: Mapping[str, Any], authority_sha256: str,
) -> tuple[dict[str, dict[str, dict[str, DocumentCellSums]]], tuple[str, ...]]:
    documents = payload.get("ordered_document_ids")
    candidates = payload.get("candidates")
    if (
        payload.get("schema") != "terminal_copy_selection_v1_ledger"
        or payload.get("authority_sha256") != authority_sha256
        or payload.get("raw_logits_published") is not False
        or not isinstance(documents, list) or len(documents) != NATURAL_DOCUMENTS
        or len(set(documents)) != NATURAL_DOCUMENTS
        or any(not isinstance(document, str) or not document for document in documents)
        or payload.get("ordered_document_ids_sha256") != _document_digest(tuple(documents))
        or not isinstance(candidates, Mapping) or set(candidates) != set(FROZEN_CANDIDATES)
        or not isinstance(payload.get("natural_batch_closures"), list)
        or len(payload["natural_batch_closures"]) != NATURAL_BATCHES
        or not isinstance(payload.get("synthetic_item_ids"), list)
        or len(payload["synthetic_item_ids"]) != SYNTHETIC_PAIRS
        or not isinstance(payload.get("synthetic_effects"), Mapping)
        or set(payload["synthetic_effects"]) != set(FROZEN_CANDIDATES)
        or not isinstance(payload.get("synthetic_batch_closures"), list)
        or len(payload["synthetic_batch_closures"]) != SYNTHETIC_BATCHES
    ):
        raise RuntimeError("selection serialized ledger topology changed")
    output: dict[str, dict[str, dict[str, DocumentCellSums]]] = {}
    document_tuple = tuple(documents)
    expected_fields = set(DocumentCellSums.__dataclass_fields__)
    for candidate in FROZEN_CANDIDATES:
        candidate_payload = candidates[candidate]
        if not isinstance(candidate_payload, Mapping) or set(candidate_payload) != set(document_tuple):
            raise RuntimeError("selection serialized candidate documents changed")
        output[candidate] = {}
        for document in document_tuple:
            cells = candidate_payload[document]
            if not isinstance(cells, Mapping) or set(cells) != set(CELL_NAMES):
                raise RuntimeError("selection serialized cell topology changed")
            output[candidate][document] = {}
            for cell in CELL_NAMES:
                fields = cells[cell]
                if not isinstance(fields, Mapping) or set(fields) != expected_fields:
                    raise RuntimeError("selection serialized sufficient statistic changed")
                try:
                    value = DocumentCellSums(**fields)
                except (TypeError, ValueError) as error:
                    raise RuntimeError("selection serialized sufficient statistic is malformed") from error
                output[candidate][document][cell] = value
    _validate_serialized_closures_and_synthetic(payload, document_tuple)
    # Full statistical validation, including identical support across candidates.
    replay = simultaneous_selection_bootstrap(
        output,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        expected_candidates=FROZEN_CANDIDATES,
    )
    if len(replay.coordinate_names) != COORDINATE_COUNT:
        raise RuntimeError("selection bootstrap coordinate count changed")
    return output, document_tuple


def _validate_serialized_closures_and_synthetic(
    payload: Mapping[str, Any], documents: tuple[str, ...],
) -> None:
    def validate_candidate_closure(value: Any, candidate: str, document_calls: int) -> None:
        plan = PhysicalCandidateDispatcher.plan(candidate)
        plan_json = [[layer, list(heads)] for layer, heads in plan]
        selected = dict(plan)
        if (
            not isinstance(value, Mapping)
            or value.get("candidate") != candidate
            or value.get("attempted_batch_calls") != 1
            or value.get("batch_calls") != 1
            or value.get("document_calls") != document_calls
            or value.get("native_attention_calls") != [
                0 if layer in selected else 1 for layer in range(18)
            ]
            or value.get("adapter_attention_calls") != [
                1 if layer in selected else 0 for layer in range(18)
            ]
            or value.get("native_mlp_calls") != [1] * 18
            or value.get("selected_layer_heads") != plan_json
            or value.get("closed") is not True
            or not isinstance(value.get("maximum_head_recomposition_abs_error"), (int, float))
            or not isinstance(value.get("maximum_head_recomposition_relative_error"), (int, float))
            or not math.isfinite(value["maximum_head_recomposition_abs_error"])
            or not math.isfinite(value["maximum_head_recomposition_relative_error"])
            or value["maximum_head_recomposition_abs_error"] < 0
            or not 0 <= value["maximum_head_recomposition_relative_error"] <= 0.003
        ):
            raise RuntimeError("selection serialized candidate closure changed")

    natural = payload["natural_batch_closures"]
    observed_documents: list[str] = []
    for closure in natural:
        ids = closure.get("document_ids") if isinstance(closure, Mapping) else None
        candidate_closures = closure.get("candidate_closures") if isinstance(closure, Mapping) else None
        if (
            not isinstance(ids, list) or len(ids) != NATURAL_BATCH_SIZE
            or not isinstance(candidate_closures, list) or len(candidate_closures) != len(FROZEN_CANDIDATES)
            or closure.get("native_attention_calls") != [1] * 18
            or closure.get("native_mlp_calls") != [1] * 18
            or closure.get("native_unembedding_calls") != 1
            or closure.get("candidate_unembedding_calls") != [1] * len(FROZEN_CANDIDATES)
            or closure.get("raw_logits_returned") is not False
            or closure.get("closed") is not True
        ):
            raise RuntimeError("selection serialized natural closure changed")
        for candidate, candidate_closure in zip(
            FROZEN_CANDIDATES, candidate_closures, strict=True,
        ):
            validate_candidate_closure(candidate_closure, candidate, NATURAL_BATCH_SIZE)
        observed_documents.extend(ids)
    if tuple(observed_documents) != documents:
        raise RuntimeError("selection serialized natural closure order changed")

    item_ids = payload["synthetic_item_ids"]
    if len(set(item_ids)) != SYNTHETIC_PAIRS or any(
        item != f"selection_synthetic_{index:03d}" for index, item in enumerate(item_ids)
    ):
        raise RuntimeError("selection serialized synthetic item bank changed")
    synthetic_closures = payload["synthetic_batch_closures"]
    observed_items: list[str] = []
    for closure in synthetic_closures:
        ids = closure.get("item_ids") if isinstance(closure, Mapping) else None
        candidate_closures = closure.get("candidate_closures") if isinstance(closure, Mapping) else None
        if (
            not isinstance(ids, list) or len(ids) != 2
            or not isinstance(candidate_closures, list) or len(candidate_closures) != len(FROZEN_CANDIDATES)
            or closure.get("native_attention_calls") != [1] * 18
            or closure.get("native_mlp_calls") != [1] * 18
            or closure.get("native_unembedding_calls") != 1
            or closure.get("candidate_unembedding_calls") != [1] * len(FROZEN_CANDIDATES)
            or closure.get("raw_logits_returned") is not False
            or closure.get("closed") is not True
        ):
            raise RuntimeError("selection serialized synthetic closure changed")
        for candidate, candidate_closure in zip(
            FROZEN_CANDIDATES, candidate_closures, strict=True,
        ):
            validate_candidate_closure(candidate_closure, candidate, SYNTHETIC_BATCH_SIZE)
        observed_items.extend(ids)
    if observed_items != item_ids:
        raise RuntimeError("selection serialized synthetic closure order changed")
    for candidate in FROZEN_CANDIDATES:
        values = payload["synthetic_effects"][candidate]
        if not isinstance(values, list) or len(values) != SYNTHETIC_PAIRS:
            raise RuntimeError("selection serialized synthetic effects changed")
        for item, value in zip(item_ids, values, strict=True):
            if (
                not isinstance(value, Mapping)
                or set(value) != set((
                    "item_id", "native_did", "candidate_did", "candidate_minus_native_did",
                ))
                or value.get("item_id") != item
                or any(
                    not isinstance(value.get(key), (int, float))
                    or not math.isfinite(value[key])
                    for key in ("native_did", "candidate_did", "candidate_minus_native_did")
                )
                or value["candidate_minus_native_did"]
                    != value["candidate_did"] - value["native_did"]
            ):
                raise RuntimeError("selection serialized synthetic effect changed")
    reference = payload["synthetic_effects"][FROZEN_CANDIDATES[0]]
    for candidate in FROZEN_CANDIDATES:
        if any(
            left["item_id"] != right["item_id"] or left["native_did"] != right["native_did"]
            for left, right in zip(reference, payload["synthetic_effects"][candidate], strict=True)
        ):
            raise RuntimeError("selection serialized synthetic native baseline changed")


def _aggregate_candidate_report(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]],
) -> dict[str, Any]:
    effects = pooled_effects(ledger)
    cells: dict[str, Any] = {}
    for cell in CELL_NAMES:
        values = [ledger[document][cell] for document in ledger]
        n = sum(value.n for value in values)
        if n <= 0:
            raise RuntimeError("selection aggregate cell has zero support")
        native_nll = sum(value.native_nll_sum for value in values)
        candidate_nll = sum(value.ablated_nll_sum for value in values)
        cells[cell] = {
            "count": n,
            "native_ce": native_nll / n,
            "candidate_ce": candidate_nll / n,
            "tau": (candidate_nll - native_nll) / n,
            "native_top1_accuracy": sum(value.native_correct_count for value in values) / n,
            "candidate_top1_accuracy": sum(value.ablated_correct_count for value in values) / n,
            "native_to_candidate_kl": sum(value.native_to_ablated_kl_sum for value in values) / n,
        }
    return {"effects": asdict(effects), "cells": cells}


def _selection_result_payload(
    authority_sha256: str,
    ledger_sha256: str,
    ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
    selection: SelectionResult,
    synthetic: MergedSyntheticBatches,
    *,
    ordered_document_ids_sha256: str,
    selection_file_sha256: str,
    frequencies_file_sha256: str,
    checkpoint_weights_sha256_before: str,
    checkpoint_weights_sha256_after: str,
    model_state_sha256_before: str,
    model_state_sha256_after: str,
) -> dict[str, Any]:
    if (
        selection.candidates != tuple(sorted(FROZEN_CANDIDATES))
        or len(selection.coordinate_names) != COORDINATE_COUNT
        or selection.point_estimates.dtype != torch.float64
        or selection.simultaneous_lower_bounds.dtype != torch.float64
    ):
        raise RuntimeError("selection bootstrap result topology changed")
    return {
        "schema": "terminal_copy_selection_v1_result",
        "status": "complete_selection_screen",
        "authority_sha256": authority_sha256,
        "ledger_file_sha256": ledger_sha256,
        "ordered_document_ids_sha256": ordered_document_ids_sha256,
        "selection_file_sha256": selection_file_sha256,
        "fit_frequencies_file_sha256": frequencies_file_sha256,
        "checkpoint_weights_sha256_before_load": checkpoint_weights_sha256_before,
        "checkpoint_weights_sha256_after_load": checkpoint_weights_sha256_after,
        "model_state_sha256_before": model_state_sha256_before,
        "model_state_sha256_after": model_state_sha256_after,
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "critical_index_zero_based": CRITICAL_INDEX,
            "coordinate_names": list(selection.coordinate_names),
            "point_estimates": selection.point_estimates.tolist(),
            "simultaneous_lower_bounds": selection.simultaneous_lower_bounds.tolist(),
            "critical_value": selection.critical_value,
            "selected_candidate": selection.selected_candidate,
        },
        "candidate_reports": {
            candidate: _aggregate_candidate_report(ledgers[candidate])
            for candidate in FROZEN_CANDIDATES
        },
        "synthetic_descriptive": {
            candidate: {
                "item_count": len(synthetic.effects[candidate]),
                "native_did_mean": sum(
                    value.native_did for value in synthetic.effects[candidate]
                ) / len(synthetic.effects[candidate]),
                "candidate_did_mean": sum(
                    value.candidate_did for value in synthetic.effects[candidate]
                ) / len(synthetic.effects[candidate]),
                "candidate_minus_native_did_mean": sum(
                    value.candidate_minus_native_did
                    for value in synthetic.effects[candidate]
                ) / len(synthetic.effects[candidate]),
                "eligible_for_selection": False,
            }
            for candidate in FROZEN_CANDIDATES
        },
        "integrity": {
            "natural_documents": NATURAL_DOCUMENTS,
            "natural_batches": NATURAL_BATCHES,
            "synthetic_pairs": SYNTHETIC_PAIRS,
            "synthetic_batches": SYNTHETIC_BATCHES,
            "raw_logits_published": False,
            "all_mlps_native": True,
            "shared_native_baseline_enforced": True,
            "independent_mask_reconstruction": True,
            "fit_receipt_self_authorized_selection": False,
            "selection_authority_licensed_fit_bank": True,
            "final_ood_deserialized": False,
        },
        "claim_boundary": (
            "Total effect of the registered position-mean intervention on registered copy cells; "
            "not representation, uniqueness, interaction resolution, extraction, or a standalone program."
        ),
    }


_COLLECTION_SEAL = object()


class _CollectedSelectionTransaction:
    __slots__ = (
        "natural", "synthetic", "selection", "inputs", "weights_before",
        "weights_after", "model_state_before", "model_state_after",
        "authority_sha256", "claim_nonce",
    )

    def __init__(
        self, seal: object, *, natural: MergedSelectionBatches,
        synthetic: MergedSyntheticBatches, selection: SelectionResult,
        inputs: SelectionInputs, weights_before: str, weights_after: str,
        model_state_before: str, model_state_after: str,
        authority_sha256: str, claim: RunClaim,
    ) -> None:
        if seal is not _COLLECTION_SEAL:
            raise RuntimeError("selection collection capability cannot be constructed externally")
        self.natural = natural
        self.synthetic = synthetic
        self.selection = selection
        self.inputs = inputs
        self.weights_before = weights_before
        self.weights_after = weights_after
        self.model_state_before = model_state_before
        self.model_state_after = model_state_after
        self.authority_sha256 = authority_sha256
        self.claim_nonce = claim.nonce


def _synthetic_report_from_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in FROZEN_CANDIDATES:
        values = payload["synthetic_effects"][candidate]
        output[candidate] = {
            "item_count": len(values),
            "native_did_mean": sum(value["native_did"] for value in values) / len(values),
            "candidate_did_mean": sum(value["candidate_did"] for value in values) / len(values),
            "candidate_minus_native_did_mean": sum(
                value["candidate_minus_native_did"] for value in values
            ) / len(values),
            "eligible_for_selection": False,
        }
    return output


def _validate_result_semantically(
    result: Mapping[str, Any], ledger_payload: Mapping[str, Any],
    ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
    selection: SelectionResult, authority: Mapping[str, Any],
) -> None:
    expected_bootstrap = {
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "critical_index_zero_based": CRITICAL_INDEX,
        "coordinate_names": list(selection.coordinate_names),
        "point_estimates": selection.point_estimates.tolist(),
        "simultaneous_lower_bounds": selection.simultaneous_lower_bounds.tolist(),
        "critical_value": selection.critical_value,
        "selected_candidate": selection.selected_candidate,
    }
    expected_integrity = {
        "natural_documents": NATURAL_DOCUMENTS,
        "natural_batches": NATURAL_BATCHES,
        "synthetic_pairs": SYNTHETIC_PAIRS,
        "synthetic_batches": SYNTHETIC_BATCHES,
        "raw_logits_published": False,
        "all_mlps_native": True,
        "shared_native_baseline_enforced": True,
        "independent_mask_reconstruction": True,
        "fit_receipt_self_authorized_selection": False,
        "selection_authority_licensed_fit_bank": True,
        "final_ood_deserialized": False,
    }
    if (
        result.get("schema") != "terminal_copy_selection_v1_result"
        or result.get("status") != "complete_selection_screen"
        or result.get("authority_sha256") != authority["authority_sha256"]
        or result.get("ledger_file_sha256") != file_sha256(LEDGER)
        or result.get("ordered_document_ids_sha256")
            != ledger_payload["ordered_document_ids_sha256"]
        or result.get("selection_file_sha256") != SELECTION_PAYLOAD_SHA256
        or result.get("fit_frequencies_file_sha256") != FIT_FREQUENCIES_SHA256
        or result.get("checkpoint_weights_sha256_before_load") != facade.WEIGHTS_SHA256
        or result.get("checkpoint_weights_sha256_after_load") != facade.WEIGHTS_SHA256
        or result.get("model_state_sha256_before") != result.get("model_state_sha256_after")
        or not isinstance(result.get("model_state_sha256_before"), str)
        or len(result["model_state_sha256_before"]) != 64
        or result.get("bootstrap") != expected_bootstrap
        or result.get("candidate_reports") != {
            candidate: _aggregate_candidate_report(ledgers[candidate])
            for candidate in FROZEN_CANDIDATES
        }
        or result.get("synthetic_descriptive") != _synthetic_report_from_ledger(ledger_payload)
        or result.get("integrity") != expected_integrity
        or not isinstance(result.get("claim_boundary"), str)
    ):
        raise RuntimeError("selection result failed independent semantic replay")


def _publish_selection_bundle(
    *, authority: Mapping[str, Any], claim: RunClaim,
    collected: _CollectedSelectionTransaction,
    protected_before: Mapping[str, str | None],
    protected_after: Mapping[str, str | None],
) -> dict[str, Any]:
    """Publish ledger/result/manifest and exactly one decision receipt last."""

    require_claim(claim)
    validate_execution_authority(authority)
    live_protected = protected_snapshot()
    authority_sha = authority["authority_sha256"]
    if (
        dict(protected_before) != live_protected
        or dict(protected_after) != live_protected
        or not isinstance(collected, _CollectedSelectionTransaction)
        or collected.claim_nonce != claim.nonce
        or collected.authority_sha256 != authority_sha
        or collected.weights_before != facade.WEIGHTS_SHA256
        or collected.weights_after != facade.WEIGHTS_SHA256
        or collected.model_state_before != collected.model_state_after
        or collected.inputs.selection_file_sha256 != SELECTION_PAYLOAD_SHA256
        or collected.inputs.frequencies_file_sha256 != FIT_FREQUENCIES_SHA256
        or len(collected.natural.batch_closures) != NATURAL_BATCHES
        or len(collected.synthetic.batch_closures) != SYNTHETIC_BATCHES
    ):
        raise RuntimeError("selection collection capability or protected inputs changed")
    _validate_exact_support(collected.natural, collected.inputs)
    ledger_payload = _ledger_payload(authority_sha, collected.natural, collected.synthetic)
    create_only_json(LEDGER, ledger_payload)
    reloaded_ledger = stable_json(LEDGER)
    if reloaded_ledger != ledger_payload:
        raise RuntimeError("selection ledger failed exact reload")
    replay_ledgers, replay_documents = _deserialize_ledger(reloaded_ledger, authority_sha)
    if replay_documents != collected.inputs.ordered_document_ids:
        raise RuntimeError("selection ledger document replay changed")
    replay_selection = simultaneous_selection_bootstrap(
        replay_ledgers,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        expected_candidates=FROZEN_CANDIDATES,
    )
    if (
        replay_selection.candidates != collected.selection.candidates
        or replay_selection.coordinate_names != collected.selection.coordinate_names
        or not torch.equal(
            replay_selection.point_estimates, collected.selection.point_estimates,
        )
        or not torch.equal(
            replay_selection.simultaneous_lower_bounds,
            collected.selection.simultaneous_lower_bounds,
        )
        or replay_selection.critical_value != collected.selection.critical_value
        or replay_selection.selected_candidate != collected.selection.selected_candidate
    ):
        raise RuntimeError("selection bootstrap changed after ledger serialization")
    result_payload = _selection_result_payload(
        authority_sha,
        file_sha256(LEDGER),
        replay_ledgers,
        replay_selection,
        collected.synthetic,
        ordered_document_ids_sha256=collected.inputs.ordered_document_ids_sha256,
        selection_file_sha256=collected.inputs.selection_file_sha256,
        frequencies_file_sha256=collected.inputs.frequencies_file_sha256,
        checkpoint_weights_sha256_before=collected.weights_before,
        checkpoint_weights_sha256_after=collected.weights_after,
        model_state_sha256_before=collected.model_state_before,
        model_state_sha256_after=collected.model_state_after,
    )
    create_only_json(RESULT, result_payload)
    reloaded_result = stable_json(RESULT)
    _validate_result_semantically(
        reloaded_result, reloaded_ledger, replay_ledgers, replay_selection, authority,
    )
    manifest = {
        "schema": "terminal_copy_selection_v1_manifest",
        "authority_sha256": authority_sha,
        "files": {str(LEDGER): file_sha256(LEDGER), str(RESULT): file_sha256(RESULT)},
        "protected_before": dict(protected_before),
        "protected_after": dict(protected_after),
        "protected_unchanged": True,
        "selected_candidate": replay_selection.selected_candidate,
        "terminal_state": "passer" if replay_selection.selected_candidate is not None else "scientific_negative",
    }
    create_only_json(MANIFEST, manifest)

    # Full terminal replay before constructing the receipt.
    require_claim(claim)
    validate_execution_authority(authority)
    verify_source_closure(authority["source_closure"])
    terminal_protected_before = protected_snapshot()
    terminal_ledger = stable_json(LEDGER)
    terminal_ledgers, terminal_documents = _deserialize_ledger(terminal_ledger, authority_sha)
    terminal_selection = simultaneous_selection_bootstrap(
        terminal_ledgers,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        expected_candidates=FROZEN_CANDIDATES,
    )
    terminal_result = stable_json(RESULT)
    _validate_result_semantically(
        terminal_result, terminal_ledger, terminal_ledgers, terminal_selection, authority,
    )
    terminal_manifest = stable_json(MANIFEST)
    terminal_protected_after = protected_snapshot()
    if (
        terminal_documents != collected.inputs.ordered_document_ids
        or terminal_selection.selected_candidate != replay_selection.selected_candidate
        or terminal_manifest != manifest
        or terminal_protected_before != live_protected
        or terminal_protected_after != live_protected
        or FAILURE.exists() or PASSER_RECEIPT.exists() or NEGATIVE_RECEIPT.exists()
        or file_sha256(LEDGER) != manifest["files"][str(LEDGER)]
        or file_sha256(RESULT) != manifest["files"][str(RESULT)]
    ):
        raise RuntimeError("selection terminal publication replay failed")
    is_passer = terminal_selection.selected_candidate is not None
    receipt_path = PASSER_RECEIPT if is_passer else NEGATIVE_RECEIPT
    other_receipt = NEGATIVE_RECEIPT if is_passer else PASSER_RECEIPT
    receipt = {
        "schema": (
            "terminal_copy_selection_v1_passer_receipt"
            if is_passer else "terminal_copy_selection_v1_negative_receipt"
        ),
        "status": "complete_receipt_last_passer" if is_passer else "complete_receipt_last_scientific_negative",
        "authority_file_sha256": file_sha256(AUTHORITY),
        "authority_sha256": authority_sha,
        "ledger_file_sha256": file_sha256(LEDGER),
        "result_file_sha256": file_sha256(RESULT),
        "manifest_file_sha256": file_sha256(MANIFEST),
        "selected_candidate": terminal_selection.selected_candidate,
        "bootstrap_coordinate_names": list(terminal_selection.coordinate_names),
        "bootstrap_point_estimates": terminal_selection.point_estimates.tolist(),
        "bootstrap_simultaneous_lower_bounds": terminal_selection.simultaneous_lower_bounds.tolist(),
        "bootstrap_critical_value": terminal_selection.critical_value,
        "ordered_document_ids_sha256": collected.inputs.ordered_document_ids_sha256,
        "natural_documents": NATURAL_DOCUMENTS,
        "natural_batches": NATURAL_BATCHES,
        "synthetic_pairs": SYNTHETIC_PAIRS,
        "synthetic_batches": SYNTHETIC_BATCHES,
        "protected_unchanged": True,
        "final_ood_opening_authorized": is_passer,
        "negative_forbids_final_ood_opening": not is_passer,
        "E4_selection_screen_complete": True,
        "strict_whole_model_ledger_credit": False,
    }

    # Adjacent receipt gate: no mutable helper work occurs after this check.
    require_claim(claim)
    validate_execution_authority(authority)
    adjacent_protected_before = protected_snapshot()
    adjacent_ledger = stable_json(LEDGER)
    adjacent_ledgers, adjacent_documents = _deserialize_ledger(adjacent_ledger, authority_sha)
    adjacent_selection = simultaneous_selection_bootstrap(
        adjacent_ledgers,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        expected_candidates=FROZEN_CANDIDATES,
    )
    adjacent_result = stable_json(RESULT)
    _validate_result_semantically(
        adjacent_result, adjacent_ledger, adjacent_ledgers, adjacent_selection, authority,
    )
    adjacent_manifest = stable_json(MANIFEST)
    adjacent_protected_after = protected_snapshot()
    require_claim(claim)
    if (
        adjacent_documents != terminal_documents
        or adjacent_selection.selected_candidate != receipt["selected_candidate"]
        or adjacent_manifest != manifest
        or adjacent_protected_before != live_protected
        or adjacent_protected_after != live_protected
        or FAILURE.exists() or receipt_path.exists() or other_receipt.exists()
        or file_sha256(AUTHORITY) != receipt["authority_file_sha256"]
        or file_sha256(LEDGER) != receipt["ledger_file_sha256"]
        or file_sha256(RESULT) != receipt["result_file_sha256"]
        or file_sha256(MANIFEST) != receipt["manifest_file_sha256"]
    ):
        raise RuntimeError("selection adjacent receipt gate failed")
    create_only_json(receipt_path, receipt)
    return receipt


def _publish_failure(claim: RunClaim, authority_sha256: str, error: BaseException) -> None:
    require_claim(claim)
    if PASSER_RECEIPT.exists() or NEGATIVE_RECEIPT.exists():
        raise RuntimeError("cannot publish selection failure after a decision receipt")
    create_only_json(FAILURE, {
        "schema": "terminal_copy_selection_v1_failure",
        "status": "terminal_integrity_or_execution_failure_no_decision_receipt",
        "authority_sha256": authority_sha256,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "ledger_exists": LEDGER.exists(),
        "result_exists": RESULT.exists(),
        "manifest_exists": MANIFEST.exists(),
        "passer_receipt_exists": False,
        "negative_receipt_exists": False,
        "same_authority_retry_authorized": False,
    })


def execute_selection() -> dict[str, Any]:
    """Run the sole authorized natural+descriptive-synthetic selection transaction."""

    require_pristine_execution_namespace()
    authority = stable_json(AUTHORITY)
    validate_execution_authority(authority)
    claim = acquire_claim()
    model: torch.nn.Module | None = None
    try:
        require_claim(claim)
        protected_before = protected_snapshot()
        inputs = _load_selection_inputs(authority, claim)
        fit_bank = _load_fit_bank(authority)
        weights_path = facade.DEFAULT_SNAPSHOT / "pytorch_model.bin"
        weights_before = file_sha256(weights_path)
        if weights_before != facade.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint weights changed immediately before selection load")
        model, loaded = facade.load_bilin18(
            device="cuda", dtype=torch.bfloat16, verify_weights_sha256=False,
        )
        weights_after_load = file_sha256(weights_path)
        if (
            weights_after_load != weights_before
            or weights_after_load != facade.WEIGHTS_SHA256
            or asdict(loaded) != authority["checkpoint"]
        ):
            raise RuntimeError("loaded checkpoint differs from selection authority")
        facade.validate_production_model(model)
        model_state_before = model_state_sha256(model)
        means = {
            layer: value.to(device="cuda", dtype=torch.float32)
            for layer, value in fit_bank.per_head_position_means.items()
        }
        dispatcher = PhysicalCandidateDispatcher.from_native(
            attentions={layer: model.transformer.h[layer].attn for layer in NAMED_LAYERS},
            per_head_position_means=means,
        )
        dispatcher.assert_matches_native({
            layer: model.transformer.h[layer].attn for layer in NAMED_LAYERS
        })

        natural_batches: list[SelectionBatchResult] = []
        for start in range(0, NATURAL_DOCUMENTS, NATURAL_BATCH_SIZE):
            stop = start + NATURAL_BATCH_SIZE
            owner = SelectionBatchOwner(dispatcher)
            natural_batches.append(owner.run(
                model,
                inputs.rows[start:stop, :256].to(device="cuda"),
                inputs.rows[start:stop],
                {cell: inputs.masks[cell][start:stop] for cell in CELL_NAMES},
                inputs.ordered_document_ids[start:stop],
                require_production=True,
            ))
        natural = merge_selection_batches(
            tuple(natural_batches), inputs.ordered_document_ids,
        )
        if len(natural.batch_closures) != NATURAL_BATCHES:
            raise RuntimeError("selection natural batch census changed")
        _validate_exact_support(natural, inputs)

        synthetic_batches: list[SyntheticBatchResult] = []
        for row_start in range(0, SYNTHETIC_ROWS, SYNTHETIC_BATCH_SIZE):
            row_stop = row_start + SYNTHETIC_BATCH_SIZE
            item_start, item_stop = row_start // 2, row_stop // 2
            owner = SyntheticSelectionBatchOwner(dispatcher)
            synthetic_batches.append(owner.run(
                model,
                inputs.synthetic_rows[row_start:row_stop, :256].to(device="cuda"),
                inputs.synthetic_rows[row_start:row_stop],
                inputs.synthetic_item_ids[item_start:item_stop],
                inputs.synthetic_query_positions[item_start:item_stop],
                inputs.synthetic_successor_y[item_start:item_stop],
                inputs.synthetic_successor_z[item_start:item_stop],
                require_production=True,
            ))
        synthetic = merge_synthetic_batches(
            tuple(synthetic_batches), inputs.synthetic_item_ids,
        )
        if len(synthetic.batch_closures) != SYNTHETIC_BATCHES:
            raise RuntimeError("selection synthetic batch census changed")

        # Literal arguments are part of the authority; there is no runtime override.
        selection = simultaneous_selection_bootstrap(
            natural.ledgers,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            expected_candidates=FROZEN_CANDIDATES,
        )
        if (
            BOOTSTRAP_REPETITIONS != 10_000
            or BOOTSTRAP_SEED != "terminal-copy-v1-document-bootstrap:0"
            or len(selection.coordinate_names) != COORDINATE_COUNT
        ):
            raise RuntimeError("selection bootstrap constants changed at execution")
        model_state_after = model_state_sha256(model)
        if model_state_after != model_state_before:
            raise RuntimeError("selection execution mutated model parameters or buffers")
        weights_after = file_sha256(weights_path)
        if weights_after != weights_before:
            raise RuntimeError("checkpoint file changed during selection execution")
        protected_after = protected_snapshot()
        collected = _CollectedSelectionTransaction(
            _COLLECTION_SEAL,
            natural=natural, synthetic=synthetic, selection=selection, inputs=inputs,
            weights_before=weights_before, weights_after=weights_after,
            model_state_before=model_state_before, model_state_after=model_state_after,
            authority_sha256=authority["authority_sha256"], claim=claim,
        )
        return _publish_selection_bundle(
            authority=authority, claim=claim, collected=collected,
            protected_before=protected_before, protected_after=protected_after,
        )
    except BaseException as error:
        if (
            not PASSER_RECEIPT.exists() and not NEGATIVE_RECEIPT.exists()
            and not FAILURE.exists()
        ):
            _publish_failure(claim, str(authority.get("authority_sha256", "")), error)
        raise
    finally:
        del model
        release_claim(claim)


if __name__ == "__main__":
    print(json.dumps(execute_selection(), indent=2, allow_nan=False))
