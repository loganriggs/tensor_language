#!/usr/bin/env python3
"""Run the preregistered authoritative mixed PCA/exact early-MLP lattice."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
TENSOR_ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "joint_early_mlp_pca_composition_v3_preregistration.json"
PURE_CONTRACT = HERE / "joint_early_mlp_pca_composition_v3.py"
TEST_CONTRACT = HERE / "test_joint_early_mlp_pca_composition_v3.py"
RESULT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_results.json"
MANIFEST = BQ / "joint_early_mlp_pca_composition_authoritative_v3_manifest.json"
AUTHORITY_RECEIPT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_authority.json"
BASIS_ARTIFACT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
BASIS_RECEIPT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json"
LOCK = Path("/workspace/runs/.bilin18_joint_early_mlp_pca_composition_authoritative_v3.lock")
PREREG_SHA256 = "a621227696f09980029e594d3faeb3687c22895b09e8a29fc9ab8ea40f4a441d"
PCA_SEED = 161803
PCA_RANK = 64
SUPPORT_RANK = 256
BOOTSTRAP_SEED = 31415926
BOOTSTRAP_DRAWS = 2000
EXACT_RESULT = BQ / "joint_early_mlp_oracle_factorial_authoritative_v4_results.json"
EXACT_MANIFEST = BQ / "joint_early_mlp_oracle_factorial_authoritative_v4_manifest.json"
EXACT_AUTHORITY = BQ / "joint_early_mlp_oracle_factorial_authoritative_v4_authority.json"

sys.path.insert(0, str(HERE))
import code_ood_oracle as code_oracle  # noqa: E402
import frozen_ship_oracle_v2 as frozen  # noqa: E402
import joint_early_mlp_oracle_factorial as joint  # noqa: E402
import joint_early_mlp_oracle_factorial_authoritative as exact_runner  # noqa: E402
from joint_early_mlp_pca_composition_v3 import (  # noqa: E402
    ARM_STATES,
    arm_name,
    paired_document_cluster_lattice,
    score_registered_predictions,
)
import prepare_fineweb_oracle_rows as row_prep  # noqa: E402
import source_global_preflight  # noqa: E402


PINNED_INPUTS = {
    **exact_runner.PINNED_INPUTS,
    PREREG: PREREG_SHA256,
    EXACT_RESULT: "0bf3988e82c8d381063d7badfc1a143d2dc0d9921bac76570d7ba4c370e5aa98",
    EXACT_MANIFEST: "de6524fd4c91b19a4cae7f9942c59792ee30b181568a9afcd3498ce73bdf8406",
    EXACT_AUTHORITY: "62a62262bf89b1a2a59b9a1dab092f57910f02e4550f17bd75689a9b3e583e33",
}
SOURCE_CLOSURE = tuple(dict.fromkeys((
    Path(__file__), PURE_CONTRACT, TEST_CONTRACT, *exact_runner.SOURCE_CLOSURE,
)))
PROTECTED_EXISTING = tuple(dict.fromkeys((
    *PINNED_INPUTS,
    *exact_runner.PROTECTED_EXISTING,
    HERE / "joint_early_mlp_pca_composition_v1_preregistration.json",
    BQ / "oracle_local_pca_strength_control_v1_results.json",
    BQ / "oracle_local_pca_strength_control_v1_manifest.json",
    BQ / "oracle_local_pca_strength_control_v1_scale_receipt.json",
)))
OUTPUTS = (RESULT, MANIFEST, AUTHORITY_RECEIPT, BASIS_ARTIFACT, BASIS_RECEIPT)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def logical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_torch_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def protected_snapshot() -> dict[str, str | None]:
    return {
        str(path): file_sha256(path) if path.is_file() else None
        for path in PROTECTED_EXISTING
    }


def verify_pinned_inputs() -> None:
    for path, expected in PINNED_INPUTS.items():
        if not path.is_file():
            raise RuntimeError(f"pinned input is absent: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"pinned input changed: {path}; expected={expected} observed={observed}"
            )
    validate_exact_authority_binding(json.loads(EXACT_AUTHORITY.read_text()))


def validate_exact_authority_binding(exact_authority: Mapping[str, Any]) -> None:
    """Require the sole v4 receipt to bind the exact reference payloads."""

    required = {
        "authorized_for_scored_experiments": True,
        "result_sha256": PINNED_INPUTS[EXACT_RESULT],
        "manifest_sha256": PINNED_INPUTS[EXACT_MANIFEST],
        "ship_realization_sha256": (
            "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
        ),
    }
    for key, expected in required.items():
        if exact_authority.get(key) != expected:
            raise RuntimeError(f"v4 authority binding changed at {key}")


def validate_pending_integrity(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    basis_payload: Mapping[str, Any],
    basis_receipt: Mapping[str, Any],
    *,
    pending_result_sha256: str,
    current_basis_artifact_sha256: str,
    current_basis_receipt_sha256: str,
    authority_exists: bool,
) -> None:
    """Validate the immutable pre-evaluation basis and pending scientific payload."""

    if authority_exists:
        raise RuntimeError("refusing to overwrite an existing authority receipt")
    if result.get("status") != "scored_pending_integrity":
        raise RuntimeError("result is not ready for integrity finalization")
    if manifest.get("status") != "scored_pending_integrity":
        raise RuntimeError("manifest is not ready for integrity finalization")
    if manifest.get("pending_result_sha256") != pending_result_sha256:
        raise RuntimeError("pending scientific payload hash changed before finalization")
    for container_name, container in (("result", result), ("manifest", manifest)):
        if container.get("basis_artifact_sha256") != current_basis_artifact_sha256:
            raise RuntimeError(f"{container_name} no longer binds the scored basis artifact")
        if container.get("basis_receipt_sha256") != current_basis_receipt_sha256:
            raise RuntimeError(f"{container_name} no longer binds the scored basis receipt")
    if basis_receipt.get("artifact_sha256") != current_basis_artifact_sha256:
        raise RuntimeError("basis receipt no longer binds the basis artifact")
    identities = {
        result.get("ship_realization_sha256"),
        manifest.get("ship_realization_sha256"),
        basis_payload.get("ship_realization_sha256"),
        basis_receipt.get("ship_realization_sha256"),
    }
    if identities != {
        "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
    }:
        raise RuntimeError("pending artifacts disagree on frozen ship identity")
    if basis_receipt.get("preregistration_sha256") != PREREG_SHA256:
        raise RuntimeError("basis receipt no longer binds the v3 preregistration")


def verify_committed_source_closure() -> dict[str, str]:
    hashes = {}
    for path in SOURCE_CLOSURE:
        relative = path.resolve().relative_to(TENSOR_ROOT.resolve())
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=TENSOR_ROOT, capture_output=True, text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"behavior-bearing source is not committed: {relative}")
        clean = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=TENSOR_ROOT
        )
        if clean.returncode != 0:
            raise RuntimeError(f"behavior-bearing source differs from HEAD: {relative}")
        hashes[str(relative)] = file_sha256(path)
    return hashes


def validate_basis_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    if not BASIS_ARTIFACT.is_file() or not BASIS_RECEIPT.is_file():
        raise RuntimeError("authoritative PCA basis artifact/receipt pair is incomplete")
    receipt = json.loads(BASIS_RECEIPT.read_text())
    if receipt.get("schema_version") != 2 or receipt.get("status") != "frozen_before_evaluation":
        raise RuntimeError("authoritative PCA basis receipt metadata changed")
    if receipt.get("artifact_path") != str(BASIS_ARTIFACT.resolve()):
        raise RuntimeError("authoritative PCA basis receipt path changed")
    if receipt.get("artifact_sha256") != file_sha256(BASIS_ARTIFACT):
        raise RuntimeError("authoritative PCA basis artifact hash changed")
    payload = torch.load(BASIS_ARTIFACT, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 2 or payload.get("pca_seed") != PCA_SEED:
        raise RuntimeError("authoritative PCA basis payload metadata changed")
    if payload.get("ship_realization_sha256") != (
        "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
    ):
        raise RuntimeError("authoritative PCA basis realization changed")
    if set(payload.get("sites", {})) != {0, 1}:
        raise RuntimeError("authoritative PCA basis sites changed")
    for site in (0, 1):
        basis = payload["sites"][site]["basis"]
        if not torch.is_tensor(basis) or tuple(basis.shape) != (1152, PCA_RANK):
            raise RuntimeError(f"invalid PCA basis shape at MLP{site}")
        if basis.dtype != torch.float32 or not bool(torch.isfinite(basis).all()):
            raise RuntimeError(f"invalid PCA basis values at MLP{site}")
        gram_error = float((basis.T @ basis - torch.eye(PCA_RANK)).abs().max())
        if not math.isfinite(gram_error) or gram_error > 2e-4:
            raise RuntimeError(f"PCA basis lost orthonormality at MLP{site}: {gram_error}")
        if code_oracle.tensor_sha256(basis) != payload["sites"][site]["basis_sha256"]:
            raise RuntimeError(f"PCA basis tensor hash changed at MLP{site}")
    return payload, receipt


@torch.no_grad()
def fit_and_freeze_bases(
    sa: Any,
    rows: torch.Tensor,
    twall: dict,
    all_attention: frozenset[int],
    realization_hash: str,
    row_split_receipt: Mapping[str, Any],
    source_commit: str,
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if BASIS_ARTIFACT.exists() or BASIS_RECEIPT.exists():
        raise RuntimeError("refusing to overwrite PCA basis artifacts")
    joint.clear_oracle_corrections(sa.ORACLE_CORR)
    sa.CONTENT_CORR["on"] = False
    exact_runner.require_inert_correction_state(sa)
    sa.ORACLE_CORR.update({"on": False, "capture": {0: [], 1: []}})
    try:
        for start in range(0, len(rows), 8):
            idx = rows[start:start + 8, :-1].to(sa.DEV).contiguous()
            sa.fwd_arm(idx, all_attention, twall, frozenset(range(18)))
        captured = {
            site: torch.cat(parts).reshape(-1, 1152).contiguous()
            for site, parts in sa.ORACLE_CORR["capture"].items()
        }
    finally:
        sa.ORACLE_CORR["capture"] = None
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
    exact_runner.require_inert_correction_state(sa)

    sites: dict[int, dict[str, Any]] = {}
    torch.manual_seed(PCA_SEED)
    torch.cuda.manual_seed_all(PCA_SEED)
    for site in (0, 1):
        residual_cpu = captured[site]
        if tuple(residual_cpu.shape) != (96 * 64, 1152):
            raise RuntimeError(f"unexpected captured residual shape at MLP{site}")
        residual = residual_cpu.to(sa.DEV)
        _, singular_values, vectors = torch.pca_lowrank(
            residual, q=SUPPORT_RANK, center=False, niter=4
        )
        basis = vectors[:, :PCA_RANK].float().contiguous()
        gram_error = float(
            (basis.T @ basis - torch.eye(PCA_RANK, device=sa.DEV)).abs().max()
        )
        if not math.isfinite(gram_error) or gram_error > 2e-4:
            raise RuntimeError(f"fitted PCA basis is not orthonormal at MLP{site}")
        coefficients = residual @ basis
        projected_energy = float(coefficients.double().square().sum())
        total_energy = float(residual.double().square().sum())
        basis_cpu = basis.detach().cpu().contiguous()
        sites[site] = {
            "basis": basis_cpu,
            "basis_sha256": code_oracle.tensor_sha256(basis_cpu),
            "captured_residual_shape": list(residual_cpu.shape),
            "captured_residual_sha256": code_oracle.tensor_sha256(residual_cpu),
            "captured_residual_rms": math.sqrt(total_energy / residual.numel()),
            "projected_correction_rms": math.sqrt(projected_energy / residual.numel()),
            "captured_energy_fraction": projected_energy / total_energy,
            "support_singular_values": singular_values.detach().cpu().double(),
            "gram_max_abs_error": gram_error,
        }
        del residual, vectors, coefficients, basis
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 2,
        "status": "frozen_before_evaluation",
        "authority": "canonical_fineweb_basis_split",
        "authorized_for_training": False,
        "pca_seed": PCA_SEED,
        "projection_rank": PCA_RANK,
        "support_rank": SUPPORT_RANK,
        "capture_positions": "64::3 over 256 model-input positions",
        "ship_realization_sha256": realization_hash,
        "basis_row_receipt": dict(row_split_receipt),
        "source_commit": source_commit,
        "source_hashes": dict(source_hashes),
        "sites": sites,
    }
    write_torch_atomic(payload, BASIS_ARTIFACT)
    receipt = {
        "schema_version": 2,
        "status": "frozen_before_evaluation",
        "authority": "canonical_fineweb_basis_split",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "preregistration_sha256": PREREG_SHA256,
        "ship_realization_sha256": realization_hash,
        "basis_row_receipt": dict(row_split_receipt),
        "artifact_path": str(BASIS_ARTIFACT.resolve()),
        "artifact_sha256": file_sha256(BASIS_ARTIFACT),
        "artifact_bytes": BASIS_ARTIFACT.stat().st_size,
        "site_basis_sha256": {
            str(site): sites[site]["basis_sha256"] for site in (0, 1)
        },
        "source_commit": source_commit,
        "source_hashes": dict(source_hashes),
        "freeze_rule": "Written and validated before discovery or heldout arm scoring.",
    }
    write_json_atomic(receipt, BASIS_RECEIPT)
    return validate_basis_pair()


def correction_map(
    arm: tuple[str, str, str], bases: Mapping[int, Mapping[str, Any]], device: Any
) -> dict[int, dict[str, Any]]:
    corrections: dict[int, dict[str, Any]] = {}
    for site, state in enumerate(arm):
        if state == "N":
            continue
        if state == "P":
            if site not in (0, 1):
                raise RuntimeError("projected state is unregistered at MLP2")
            basis = bases[site]["basis"].to(device)
        elif state == "E":
            basis = None
        else:
            raise RuntimeError(f"unknown arm state: {state}")
        corrections[site] = {"basis": basis, "scale": 1.0}
    return corrections


def score_arm(
    sa: Any,
    rows: torch.Tensor,
    twall: dict,
    all_attention: frozenset[int],
    rare_vocab: torch.Tensor,
    arm: tuple[str, str, str],
    bases: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    joint.clear_oracle_corrections(sa.ORACLE_CORR)
    exact_runner.require_inert_correction_state(sa)
    joint.configure_oracle_corrections(
        sa.ORACLE_CORR, correction_map(arm, bases, sa.DEV)
    )
    try:
        return sa._score_content_rows(
            rows, twall, all_attention, frozenset(range(18)),
            rare_vocab=rare_vocab, retain_row_ce=True,
        )
    finally:
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
        exact_runner.require_inert_correction_state(sa)


def exact_v4_name(arm: tuple[str, str, str]) -> str | None:
    if any(state == "P" for state in arm):
        return None
    groups = [f"mlp{site}" for site, state in enumerate(arm) if state == "E"]
    return "+".join(groups) if groups else "baseline"


def max_abs_row_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("row-CE reproduction has mismatched lengths")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def run_claimed(protected_before: Mapping[str, str | None]) -> None:
    verify_pinned_inputs()
    source_hashes = verify_committed_source_closure()
    source_global_preflight.require_defined_globals([
        Path(__file__), PURE_CONTRACT,
        HERE / "joint_early_mlp_oracle_factorial_authoritative.py",
        HERE / "frozen_ship_oracle_v2.py",
        BQ / "ship_error_attrib.py",
    ])
    row_receipt, frozen_rows = row_prep.validate_receipt()
    if row_receipt.get("authority") != "pinned_local_ordered_manifest":
        raise RuntimeError("FineWeb receipt lacks authoritative ordered-manifest status")
    if row_receipt.get("authorized_for_scored_experiments") is not True:
        raise RuntimeError("FineWeb receipt does not authorize scored experiments")
    document_ids, split_receipts = exact_runner.validate_document_provenance(
        row_receipt, frozen_rows
    )
    code_rows, _ = code_oracle.load_frozen_corpus()
    frozen_payload, _ = frozen.validate_frozen_ship_pair(row_receipt)
    if frozen_payload["ship_realization_sha256"] != (
        "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
    ):
        raise RuntimeError("frozen ship realization identity changed")
    exact_prior = json.loads(EXACT_RESULT.read_text())
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=TENSOR_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "running_authoritative_mixed_composition",
        "authority": "canonical_fineweb",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "training_license_sites": [],
        "scope_guardrail": (
            "Same-realization modular oracle-subspace composition only; projected arms "
            "still call the original missing MLP for residual coefficients. No executable "
            "program, OOD, semantic, compression, edit, simplicity, or whole-model claim."
        ),
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": PREREG_SHA256,
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "pinned_input_hashes": {str(path): digest for path, digest in PINNED_INPUTS.items()},
        "row_splits": split_receipts,
        "protected_paths_before": dict(protected_before),
    }
    write_json_atomic(manifest, MANIFEST)

    def registered_fineweb_rows(n: int = 120, skip: int = 0) -> torch.Tensor:
        spec = (n, skip)
        if spec not in frozen_rows:
            raise RuntimeError(f"unregistered FineWeb row request: {spec}")
        return frozen_rows[spec].clone()

    torch.manual_seed(exact_runner.SHIP_SEED)
    torch.cuda.manual_seed_all(exact_runner.SHIP_SEED)
    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    torch.manual_seed(exact_runner.SHIP_SEED)
    torch.cuda.manual_seed_all(exact_runner.SHIP_SEED)
    sa.cl.fineweb_rows = registered_fineweb_rows
    start_time = time.time()

    def callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        realization_hash, frozen_manifest = frozen.restore_ship_realization(
            sa, twall, all_attention, row_receipt, code_rows
        )
        component_before = exact_runner.component_tree_sha256(sa, twall, all_attention)
        if component_before != realization_hash:
            raise RuntimeError("fresh component tree differs from frozen realization")
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
        sa.CONTENT_CORR["on"] = False
        exact_runner.require_inert_correction_state(sa)
        patch_canary = exact_runner.exact_patch_canary(sa)
        basis_rows = frozen_rows[(96, 1200)][:, :257].contiguous()
        basis_payload, basis_receipt = fit_and_freeze_bases(
            sa, basis_rows, twall, all_attention, realization_hash,
            split_receipts["basis"], source_commit, source_hashes,
        )
        if exact_runner.component_tree_sha256(sa, twall, all_attention) != component_before:
            raise RuntimeError("component tree changed during PCA basis fitting")

        discovery_rows = frozen_rows[(192, 7000)][:, :257].contiguous()
        heldout_rows = frozen_rows[(192, 11000)][:, :257].contiguous()
        rare_vocab = sa._token_masks(discovery_rows)
        evaluations: dict[str, Any] = {}
        analyses: dict[str, Any] = {}
        reproduction: dict[str, Any] = {}
        row_ce_by_split: dict[str, dict[tuple[str, str, str], list[float]]] = {}
        partial: dict[str, Any] = {
            "schema_version": 2,
            "status": "running_authoritative_mixed_composition",
            "authority": "canonical_fineweb",
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "training_license_sites": [],
            "preregistration_sha256": PREREG_SHA256,
            "ship_realization_sha256": realization_hash,
            "component_tree_before_sha256": component_before,
            "exact_patch_canary": patch_canary,
            "basis_artifact_sha256": basis_receipt["artifact_sha256"],
            "basis_receipt_sha256": file_sha256(BASIS_RECEIPT),
            "row_splits": split_receipts,
            "source_hashes": source_hashes,
            "evaluations": evaluations,
            "split_analyses": analyses,
            "exact_v4_row_reproduction": reproduction,
        }
        write_json_atomic(partial, RESULT)

        for split_index, (split_name, rows) in enumerate((
            ("discovery", discovery_rows), ("heldout", heldout_rows)
        )):
            evaluations[split_name] = {}
            reproduction[split_name] = {}
            row_ce_by_arm: dict[tuple[str, str, str], list[float]] = {}
            for arm in ARM_STATES:
                scored = score_arm(
                    sa, rows, twall, all_attention, rare_vocab, arm,
                    basis_payload["sites"],
                )
                name = arm_name(arm)
                evaluations[split_name][name] = scored
                row_ce_by_arm[arm] = scored["row_global_ce"]
                prior_name = exact_v4_name(arm)
                if prior_name is not None:
                    prior_rows = exact_prior["evaluations"][split_name][prior_name][
                        "row_global_ce"
                    ]
                    error = max_abs_row_error(scored["row_global_ce"], prior_rows)
                    if error != 0.0:
                        raise RuntimeError(
                            f"authoritative v4 row reproduction failed split={split_name} "
                            f"arm={name} error={error}"
                        )
                    reproduction[split_name][name] = error
                write_json_atomic(partial, RESULT)
                print(f"authoritative PCA composition {split_name} arm={name} done", flush=True)
            analyses[split_name] = paired_document_cluster_lattice(
                row_ce_by_arm, document_ids[split_name],
                draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + split_index,
            )
            row_ce_by_split[split_name] = row_ce_by_arm
            write_json_atomic(partial, RESULT)

        decisions = score_registered_predictions(analyses)
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
        exact_runner.require_inert_correction_state(sa)
        replay = sa._score_content_rows(
            heldout_rows, twall, all_attention, frozenset(range(18)),
            rare_vocab=rare_vocab, retain_row_ce=True,
        )
        baseline_rows = row_ce_by_split["heldout"][("N", "N", "N")]
        replay_difference = (
            torch.tensor(replay["row_global_ce"], dtype=torch.float64)
            - torch.tensor(baseline_rows, dtype=torch.float64)
        )
        baseline_replay = {
            "max_abs_row_ce_difference": float(replay_difference.abs().max()),
            "mean_abs_row_ce_difference": float(replay_difference.abs().mean()),
            "required": "bit-identical paired row CE",
        }
        if (
            baseline_replay["max_abs_row_ce_difference"] != 0.0
            or baseline_replay["mean_abs_row_ce_difference"] != 0.0
        ):
            raise RuntimeError(f"heldout baseline replay changed: {baseline_replay}")
        component_after = exact_runner.component_tree_sha256(sa, twall, all_attention)
        if component_after != component_before:
            raise RuntimeError("component tree changed during mixed composition lattice")
        result = {
            **partial,
            "status": "scored_pending_integrity",
            "interpretation_guardrail": manifest["scope_guardrail"],
            "config": {
                "arm_count": len(ARM_STATES),
                "arms": [arm_name(arm) for arm in ARM_STATES],
                "state_order": "MLP0,MLP1,MLP2; N=deployed P=rank64 E=exact",
                "projection_rank": PCA_RANK,
                "support_rank": SUPPORT_RANK,
                "projection_scale": 1.0,
                "basis_fit_rows": 96,
                "evaluation_rows": {"discovery": 192, "heldout": 192},
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seeds": {
                    "discovery": BOOTSTRAP_SEED,
                    "heldout": BOOTSTRAP_SEED + 1,
                },
                "bootstrap_unit": "FineWeb source document cluster",
                "same_currency_residual_denominator": None,
                "oracle_coefficient_warning": (
                    "P arms still evaluate exact original-minus-deployed residuals; "
                    "this is not a coefficient predictor or executable replacement."
                ),
            },
            "ship_realization_sha256": realization_hash,
            "component_tree_after_sha256": component_after,
            "component_tree_unchanged": True,
            "heldout_baseline_replay": baseline_replay,
            "basis_fit_summary": {
                str(site): {
                    key: value for key, value in basis_payload["sites"][site].items()
                    if key not in ("basis", "support_singular_values")
                }
                for site in (0, 1)
            },
            "frozen_ship_artifact_sha256": frozen_manifest["artifact_sha256"],
            "registered_decisions": decisions["decisions"],
            "registered_predictions": decisions["registered_predictions"],
            "runtime_s": round(time.time() - start_time, 1),
            "source_commit": source_commit,
            "preregistration_sha256": PREREG_SHA256,
        }
        write_json_atomic(result, RESULT)
        pending_result_hash = file_sha256(RESULT)
        manifest.update({
            "status": "scored_pending_integrity",
            "result_path": str(RESULT.resolve()),
            "basis_artifact_path": str(BASIS_ARTIFACT.resolve()),
            "basis_artifact_sha256": basis_receipt["artifact_sha256"],
            "basis_receipt_path": str(BASIS_RECEIPT.resolve()),
            "basis_receipt_sha256": file_sha256(BASIS_RECEIPT),
            "pending_result_sha256": pending_result_hash,
            "ship_realization_sha256": realization_hash,
            "component_tree_after_sha256": component_after,
            "runtime_s": result["runtime_s"],
        })
        write_json_atomic(manifest, MANIFEST)

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)


def mark_failed(error: BaseException, protected_after: Mapping[str, str | None]) -> None:
    if AUTHORITY_RECEIPT.exists():
        raise RuntimeError("refusing to invalidate an existing authority receipt") from error
    try:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    except Exception:
        manifest = {}
    manifest.update({
        "schema_version": 2,
        "status": "failed_authoritative_mixed_composition",
        "authority": "canonical_fineweb",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "training_license_sites": [],
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "protected_paths_after": dict(protected_after),
        "recovery": "Preserve every artifact and use a new versioned namespace for retry.",
    })
    write_json_atomic(manifest, MANIFEST)
    if RESULT.exists():
        try:
            result = json.loads(RESULT.read_text())
        except Exception:
            result = {}
        result.update({
            "status": "failed_authoritative_mixed_composition",
            "invalidated_by_failure": True,
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "training_license_sites": [],
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        write_json_atomic(result, RESULT)


def finalize_success(protected_after: Mapping[str, str | None]) -> None:
    if AUTHORITY_RECEIPT.exists():
        raise RuntimeError("refusing to overwrite an existing authority receipt")
    result = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    basis_payload, basis_receipt = validate_basis_pair()
    pending_result_hash = file_sha256(RESULT)
    basis_artifact_hash = file_sha256(BASIS_ARTIFACT)
    basis_receipt_hash = file_sha256(BASIS_RECEIPT)
    validate_pending_integrity(
        result, manifest, basis_payload, basis_receipt,
        pending_result_sha256=pending_result_hash,
        current_basis_artifact_sha256=basis_artifact_hash,
        current_basis_receipt_sha256=basis_receipt_hash,
        authority_exists=AUTHORITY_RECEIPT.exists(),
    )
    lifecycle = exact_runner.frozen_lifecycle_receipt(
        json.loads(row_prep.RECEIPT.read_text())
    )
    if not lifecycle.get("validated"):
        raise RuntimeError(f"frozen state does not validate at finalization: {lifecycle}")
    if result.get("component_tree_unchanged") is not True:
        raise RuntimeError("component tree integrity flag is false")
    if result.get("heldout_baseline_replay", {}).get("max_abs_row_ce_difference") != 0.0:
        raise RuntimeError("heldout baseline replay is not exactly identical")
    result.update({
        "status": "completed_payload_awaiting_authority_receipt",
        "authorized_for_scored_experiments": False,
        "authorization_rule": (
            "Evidence is authoritative only when the separate last-written atomic "
            "authority receipt binds this exact payload and its frozen bases."
        ),
        "protected_paths_after": dict(protected_after),
        "protected_paths_unchanged": True,
        "frozen_state_lifecycle": lifecycle,
        "integrity_finalized": True,
    })
    write_json_atomic(result, RESULT)
    manifest.update({
        "status": "completed_payload_awaiting_authority_receipt",
        "authorized_for_scored_experiments": False,
        "protected_paths_after": dict(protected_after),
        "protected_paths_unchanged": True,
        "frozen_state_lifecycle": lifecycle,
        "basis_payload_schema_version": basis_payload["schema_version"],
        "basis_artifact_sha256": basis_artifact_hash,
        "basis_receipt_sha256": basis_receipt_hash,
        "result_sha256": file_sha256(RESULT),
        "integrity_finalized": True,
    })
    write_json_atomic(manifest, MANIFEST)
    result_hash = file_sha256(RESULT)
    manifest_hash = file_sha256(MANIFEST)
    authority = {
        "schema_version": 2,
        "status": "completed_authoritative_mixed_composition",
        "authority": "canonical_fineweb",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
        "training_license_sites": [],
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": PREREG_SHA256,
        "source_commit": manifest["source_commit"],
        "source_hashes": manifest["source_hashes"],
        "ship_realization_sha256": result["ship_realization_sha256"],
        "result_path": str(RESULT.resolve()),
        "result_sha256": result_hash,
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": manifest_hash,
        "basis_artifact_path": str(BASIS_ARTIFACT.resolve()),
        "basis_artifact_sha256": basis_artifact_hash,
        "basis_receipt_path": str(BASIS_RECEIPT.resolve()),
        "basis_receipt_sha256": basis_receipt_hash,
        "protected_paths_unchanged": True,
        "frozen_state_validated": True,
        "authorization_rule": (
            "This last-written atomic receipt is the sole scored-evidence authority; "
            "the bound result, manifest, and basis receipt remain non-self-authorizing."
        ),
    }
    write_json_atomic(authority, AUTHORITY_RECEIPT)


def main() -> None:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite mixed-composition artifacts: {existing}")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"authoritative mixed-composition launch already claimed: {LOCK}") from error
    protected_before = protected_snapshot()
    run_error: BaseException | None = None
    try:
        run_claimed(protected_before)
    except BaseException as error:
        run_error = error
    protected_after = protected_snapshot()
    try:
        if protected_after != protected_before:
            contamination = RuntimeError("mixed composition changed protected prior artifacts")
            mark_failed(contamination, protected_after)
            raise contamination from run_error
        if run_error is not None:
            mark_failed(run_error, protected_after)
            raise run_error
        try:
            finalize_success(protected_after)
        except BaseException as finalization_error:
            mark_failed(finalization_error, protected_after)
            raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
