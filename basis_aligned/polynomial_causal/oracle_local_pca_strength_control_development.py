#!/usr/bin/env python3
"""Run the preregistered dual-strength local-PCA oracle control.

This authority-none delta run reconstructs and verifies the exact curated-v2
ship, reuses the serialized local-PCA and twenty Haar subspaces without refitting,
calibrates null strength only on the frozen spare rows, and evaluates only on the
frozen discovery/heldout rows.  It cannot write canonical oracle artifacts or
create training/OOD licenses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
TENSOR_ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "oracle_local_pca_strength_preregistration.json"
RESULT = BQ / "oracle_local_pca_strength_control_v1_results.json"
MANIFEST = BQ / "oracle_local_pca_strength_control_v1_manifest.json"
SCALE_RECEIPT = BQ / "oracle_local_pca_strength_control_v1_scale_receipt.json"
LOCK = Path("/workspace/runs/.bilin18_oracle_local_pca_strength_control_v1.lock")
SAVED_SHIP = Path("/workspace/runs/bilin18_curated_dev_v2_ship.pt")
SAVED_BASES = Path("/workspace/runs/bilin18_curated_dev_v2_oracle_bases.pt")
PREVIOUS_RESULT = BQ / "ship_content_oracle_curated_dev_v2_results.json"
PREVIOUS_MANIFEST = BQ / "ship_content_oracle_curated_dev_v2_manifest.json"
SAVED_SHIP_SHA256 = "85b848cc5d355bd99a29d43d7168f95113a11fb4c84a42fa9efe3393225dd530"
SAVED_BASES_SHA256 = "aa086ed4a14ea0474882a5443eb0e9add21734649e705398eebf3fce7a09b801"
PREVIOUS_RESULT_SHA256 = "85a7e2327c94bba21cc00bd20b8d563bf684fa0c341d03343a398bbda216b2e3"
PREVIOUS_MANIFEST_SHA256 = "d58b8a9c4d69a071b64d20ed4f5fbcc6a7f349587f8edc784d17fdbd9ffd9781"
CORPUS_SHA256 = "faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd"
SHIP_SEED = 27182818
SITES = (0, 1)
NULL_NAMES = tuple(f"null_{index:02d}" for index in range(20))
EXPECTED_COMPONENT_HASHES = {
    "ship": "c70106d20b8843ec64bb59e641e02b05335ed1437c8c119e5fa0b387c995a36a",
    "corr": "783fc65de3ae61a59e29238228eaff2660f3dede00b93612dbdc8ca33e5e8055",
    "attention": "b08302b629b53e3378f64afcf886faf623e87c8cb6e7b427e8e56b487e6e21f6",
    "all_attention": "a24255e6b168e35ba9da67e300c300d37fac3eb0ab98c851619d1cf4f2462ea7",
}
PROTECTED_EXISTING = (
    SAVED_SHIP,
    SAVED_BASES,
    PREVIOUS_RESULT,
    PREREG,
    BQ / "ship_content_oracle_curated_dev_v2_manifest.json",
    BQ / "joint_early_mlp_oracle_factorial_curated_dev_v2_results.json",
    BQ / "joint_early_mlp_oracle_factorial_curated_dev_v2_manifest.json",
    BQ / "joint_early_mlp_oracle_factorial_curated_dev_v1_manifest.json",
)

sys.path.insert(0, str(HERE))
import joint_early_mlp_oracle_factorial as joint  # noqa: E402
import joint_early_mlp_oracle_factorial_development as factorial_runner  # noqa: E402
import local_ship_oracle_development as local  # noqa: E402
from oracle_local_pca_strength_control import (  # noqa: E402
    analyze_strength_control,
    match_monotone_scale,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def protected_snapshot() -> dict[str, str | None]:
    paths = tuple(local.CANONICAL_PATHS) + PROTECTED_EXISTING
    return {str(path): file_sha256(path) if path.exists() else None for path in paths}


def validate_serialized_bases(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1 or payload.get("authority") != "none":
        raise RuntimeError("saved oracle realization metadata changed")
    sites = payload.get("sites")
    if not isinstance(sites, dict) or not set(SITES).issubset(sites):
        raise RuntimeError("saved oracle realization lacks registered sites")
    tensor_hashes: set[str] = set()
    for site in SITES:
        arms = sites[site]
        expected = {"full", "content", "local_pca", *NULL_NAMES}
        if set(arms) != expected:
            raise RuntimeError(f"saved site {site} arm identities changed")
        for name in ("local_pca", *NULL_NAMES):
            basis = arms[name].get("basis")
            if not torch.is_tensor(basis) or tuple(basis.shape) != (1152, 64):
                raise RuntimeError(f"invalid saved basis site={site} arm={name}")
            if basis.dtype != torch.float32 or not bool(torch.isfinite(basis).all()):
                raise RuntimeError(f"nonfinite or wrong-dtype basis site={site} arm={name}")
            basis_hash = local.tensor_sha256(basis)
            if basis_hash in tensor_hashes:
                raise RuntimeError(f"duplicate saved basis identity site={site} arm={name}")
            tensor_hashes.add(basis_hash)
            gram_error = float((basis.float().T @ basis.float()
                                - torch.eye(64)).abs().max())
            if not math.isfinite(gram_error) or gram_error > 2e-4:
                raise RuntimeError(
                    f"saved basis lost orthonormality site={site} arm={name}: {gram_error}"
                )
        if arms["full"].get("basis") is not None:
            raise RuntimeError(f"saved full oracle basis changed at site {site}")
        content_rms = float(arms["content"]["fit_correction_rms"])
        for name in NULL_NAMES:
            inherited = float(arms[name]["scale"])
            raw = float(arms[name]["raw_fit_correction_rms"])
            if not math.isclose(inherited * raw, content_rms, rel_tol=1e-10, abs_tol=1e-8):
                raise RuntimeError(
                    f"inherited content-RMS invariant changed site={site} arm={name}"
                )


@torch.no_grad()
def median_suffix_kl(sa: Any, rows: torch.Tensor, twall: dict,
                     all_attention: frozenset[int], all_mlps: frozenset[int],
                     *, site: int, basis: torch.Tensor, scale: float) -> float:
    """Median row KL(ship || candidate), averaged over positions 64--255."""
    row_values: list[torch.Tensor] = []
    try:
        for start in range(0, len(rows), 4):
            idx = rows[start:start + 4, :-1].to(sa.DEV).contiguous()
            joint.clear_oracle_corrections(sa.ORACLE_CORR)
            baseline = sa.fwd_arm(idx, all_attention, twall, all_mlps).float()
            joint.configure_oracle_corrections(
                sa.ORACLE_CORR,
                {site: {"basis": basis, "scale": scale}},
            )
            candidate = sa.fwd_arm(idx, all_attention, twall, all_mlps).float()
            log_candidate = F.log_softmax(candidate, dim=-1)
            log_baseline = F.log_softmax(baseline, dim=-1)
            kl = (log_baseline.exp() * (log_baseline - log_candidate)).sum(-1)
            row_values.append(kl[:, 64:].mean(1).detach().cpu())
            del baseline, candidate, log_candidate, log_baseline, kl
    finally:
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
    values = torch.cat(row_values).double()
    if len(values) != len(rows) or not bool(torch.isfinite(values).all()):
        raise RuntimeError("invalid spare-row suffix KL measurements")
    return float(values.median())


def raw_rms_scales(saved_bases: dict[str, Any], site: int) -> dict[str, dict[str, float]]:
    target = float(saved_bases["sites"][site]["local_pca"]["fit_correction_rms"])
    output = {}
    for name in NULL_NAMES:
        raw = float(saved_bases["sites"][site][name]["raw_fit_correction_rms"])
        scale = target / raw
        if (not math.isfinite(raw) or raw <= 0.0 or not math.isfinite(scale)
                or not 0.1 <= scale <= 10.0):
            raise RuntimeError(f"invalid raw-RMS scale site={site} arm={name}")
        output[name] = {
            "scale": scale,
            "target_fit_correction_rms": target,
            "raw_fit_correction_rms": raw,
            "matched_fit_correction_rms": raw * scale,
        }
    return output


def calibration_scales(sa: Any, saved_bases: dict[str, Any], rows: torch.Tensor,
                       twall: dict, all_attention: frozenset[int],
                       all_mlps: frozenset[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for site in SITES:
        local_basis = saved_bases["sites"][site]["local_pca"]["basis"].to(sa.DEV)
        target = median_suffix_kl(
            sa, rows, twall, all_attention, all_mlps,
            site=site, basis=local_basis, scale=1.0,
        )
        site_rows: dict[str, Any] = {"local_pca_target_median_suffix_kl": target,
                                    "nulls": {}}
        for name in NULL_NAMES:
            basis = saved_bases["sites"][site][name]["basis"].to(sa.DEV)
            matched = match_monotone_scale(
                target,
                lambda scale, basis=basis: median_suffix_kl(
                    sa, rows, twall, all_attention, all_mlps,
                    site=site, basis=basis, scale=scale,
                ),
            )
            site_rows["nulls"][name] = matched
            print(
                f"PCA strength calibration site={site} arm={name} "
                f"scale={matched['scale']:.6g} relerr={matched['relative_error']:.3g}",
                flush=True,
            )
        output[str(site)] = site_rows
    return output


def score_arm(sa: Any, rows: torch.Tensor, twall: dict,
              all_attention: frozenset[int], all_mlps: frozenset[int],
              rare_vocab: torch.Tensor, *, site: int | None = None,
              basis: torch.Tensor | None = None, scale: float = 1.0) -> dict[str, Any]:
    if site is None:
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
    else:
        joint.configure_oracle_corrections(
            sa.ORACLE_CORR, {site: {"basis": basis, "scale": scale}},
        )
    try:
        return sa._score_content_rows(
            rows, twall, all_attention, all_mlps,
            rare_vocab=rare_vocab, retain_row_ce=True,
        )
    finally:
        joint.clear_oracle_corrections(sa.ORACLE_CORR)


def run_claimed(canonical_before: dict[str, str | None]) -> None:
    pinned = {
        local.CORPUS: CORPUS_SHA256,
        SAVED_SHIP: SAVED_SHIP_SHA256,
        SAVED_BASES: SAVED_BASES_SHA256,
        PREVIOUS_RESULT: PREVIOUS_RESULT_SHA256,
        PREVIOUS_MANIFEST: PREVIOUS_MANIFEST_SHA256,
        local.GLUE: local.GLUE_SHA256,
        local.MODEL_SNAPSHOT / "config.json": local.MODEL_CONFIG_SHA256,
        local.MODEL_SNAPSHOT / "pytorch_model.bin": local.MODEL_WEIGHTS_SHA256,
    }
    for path, expected in pinned.items():
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"pinned input hash changed for {path}: {observed}")
    prereg_hash = file_sha256(PREREG)
    splits = local.allocate_whole_document_splits(
        torch.load(local.CORPUS, map_location="cpu", weights_only=True)
    )
    saved_ship = torch.load(SAVED_SHIP, map_location="cpu", weights_only=True)
    saved_bases = torch.load(SAVED_BASES, map_location="cpu", weights_only=True)
    validate_serialized_bases(saved_bases)
    previous = json.loads(PREVIOUS_RESULT.read_text())
    previous_manifest = json.loads(PREVIOUS_MANIFEST.read_text())
    current_receipts = factorial_runner.row_split_receipts(splits)
    for role, receipt in current_receipts.items():
        prior = previous_manifest["row_splits"][role]
        if (receipt["tensor_raw_sha256"] != prior["tensor_raw_sha256"]
                or receipt["indices"] != prior["indices"]
                or receipt["document_ids"] != prior["document_ids"]):
            raise RuntimeError(f"curated-v2 split identity changed for {role}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running_development_only",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "training_license_sites": [],
        "code_ood_licensed": False,
        "scope_guardrail": "Out-of-fit-row oracle-subspace evidence only; no semantic, simplicity, fresh-corpus, code-OOD, predictor, or generalization claim.",
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": prereg_hash,
        "canonical_paths_before": canonical_before,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=TENSOR_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_hashes": {
            "runner": file_sha256(Path(__file__)),
            "pure_contract": file_sha256(HERE / "oracle_local_pca_strength_control.py"),
            "ship_error_attrib": file_sha256(BQ / "ship_error_attrib.py"),
            "joint_runtime": file_sha256(HERE / "joint_early_mlp_oracle_factorial.py"),
        },
        "pinned_input_hashes": {str(path): value for path, value in pinned.items()},
        "row_splits": current_receipts,
        "implementation_choices": {
            "suffix_positions": [64, 255],
            "kl_orientation": "KL(deployed ship || candidate intervention)",
            "row_reduction": "mean over suffix tokens, then median over the 40 spare rows",
            "scale_search": "monotonicity guard on [0.1,0.25,0.5,1,2,4,10], then 14-step geometric bisection inside the adjacent bracket",
            "maximum_relative_strength_mismatch": 0.01,
        },
    }
    write_json_atomic(manifest, MANIFEST)

    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415
    import source_global_preflight  # noqa: PLC0415

    source_global_preflight.require_defined_globals([
        Path(__file__), HERE / "oracle_local_pca_strength_control.py",
        BQ / "ship_error_attrib.py",
    ])

    def frozen_local_rows(n: int = 120, skip: int = 0) -> torch.Tensor:
        role = local.REQUEST_ROLES.get((n, skip))
        if role is None:
            raise RuntimeError(f"unregistered curated row request {(n, skip)}")
        return splits[role]["rows"].clone()

    torch.manual_seed(SHIP_SEED)
    torch.cuda.manual_seed_all(SHIP_SEED)
    sa.cl.fineweb_rows = frozen_local_rows
    start_time = time.time()

    def callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        observed = factorial_runner.current_component_hashes(sa, twall, all_attention)
        expected = factorial_runner.verify_realization(saved_ship, observed)
        all_mlps = frozenset(range(18))
        calibrations = calibration_scales(
            sa, saved_bases, splits["spare"]["rows"], twall, all_attention, all_mlps
        )
        rms = {str(site): raw_rms_scales(saved_bases, site) for site in SITES}
        scale_receipt = {
            "schema_version": 1,
            "status": "frozen_before_evaluation",
            "authority": "none",
            "preregistration_sha256": prereg_hash,
            "saved_ship_sha256": SAVED_SHIP_SHA256,
            "saved_bases_sha256": SAVED_BASES_SHA256,
            "realization_component_hashes": expected,
            "calibration_split": factorial_runner.row_split_receipts(
                {"spare": splits["spare"]}
            )["spare"],
            "discovery_row_sha256": splits["discovery"]["tensor_raw_sha256"],
            "heldout_row_sha256": splits["heldout"]["tensor_raw_sha256"],
            "downstream_kl": calibrations,
            "raw_rms": rms,
            "absolute_scale_rule": "Every selected value directly multiplies the projected live residual; inherited content-matched null scales are not composed with it.",
            "basis_rms_sampling_rule": "Inherited raw RMS was measured on frozen basis rows at positions 64::3.",
        }
        write_json_atomic(scale_receipt, SCALE_RECEIPT)
        scale_receipt_hash = file_sha256(SCALE_RECEIPT)
        frozen_scales = json.loads(SCALE_RECEIPT.read_text())
        if file_sha256(SCALE_RECEIPT) != scale_receipt_hash:
            raise RuntimeError("scale receipt changed while being frozen")
        calibrations = frozen_scales["downstream_kl"]
        rms = frozen_scales["raw_rms"]
        rare_vocab = saved_bases["rare_vocab"].to(sa.DEV)
        evaluations: dict[str, Any] = {}
        gains: dict[str, Any] = {str(site): {} for site in SITES}
        partial: dict[str, Any] = {
            "schema_version": 1,
            "status": "running_exploratory_only",
            "authority": "none",
            "authorized_for_scored_experiments": False,
            "training_license_sites": [],
            "code_ood_licensed": False,
            "downstream_kl_calibration": calibrations,
            "raw_rms_calibration": rms,
            "scale_receipt_path": str(SCALE_RECEIPT.resolve()),
            "scale_receipt_sha256": scale_receipt_hash,
            "evaluations": evaluations,
            "paired_gains": gains,
            "preregistration_sha256": prereg_hash,
        }
        write_json_atomic(partial, RESULT)
        for split in ("discovery", "heldout"):
            rows = splits[split]["rows"]
            baseline = score_arm(
                sa, rows, twall, all_attention, all_mlps, rare_vocab
            )
            evaluations[split] = {"ship_baseline": baseline, "sites": {}}
            for site in SITES:
                key = str(site)
                site_evaluations: dict[str, Any] = {}
                evaluations[split]["sites"][key] = site_evaluations
                arms = saved_bases["sites"][site]
                for name, basis, scale in (
                    ("full", None, 1.0),
                    ("local_pca", arms["local_pca"]["basis"].to(sa.DEV), 1.0),
                ):
                    site_evaluations[name] = score_arm(
                        sa, rows, twall, all_attention, all_mlps, rare_vocab,
                        site=site, basis=basis, scale=scale,
                    )
                    print(f"PCA control {split} site={site} arm={name} done", flush=True)
                previous_baseline = previous["evaluations"][split]["ship_baseline"]["row_global_ce"]
                previous_site = previous["evaluations"][split]["sites"][key]
                reproduction = {
                    "ship_baseline_max_abs_row_ce_error": max(
                        abs(float(left) - float(right))
                        for left, right in zip(
                            baseline["row_global_ce"], previous_baseline, strict=True
                        )
                    ),
                    "full_max_abs_row_ce_error": max(
                        abs(float(left) - float(right))
                        for left, right in zip(
                            site_evaluations["full"]["row_global_ce"],
                            previous_site["full"]["row_global_ce"], strict=True,
                        )
                    ),
                    "local_pca_max_abs_row_ce_error": max(
                        abs(float(left) - float(right))
                        for left, right in zip(
                            site_evaluations["local_pca"]["row_global_ce"],
                            previous_site["local_pca"]["row_global_ce"], strict=True,
                        )
                    ),
                }
                if any(value > 1e-6 for value in reproduction.values()):
                    raise RuntimeError(
                        f"prior curated-v2 row-CE reproduction failed {split} site={site}: "
                        f"{reproduction}"
                    )
                site_evaluations["prior_v2_row_ce_reproduction"] = reproduction
                for control in ("downstream_kl", "raw_rms"):
                    site_evaluations[control] = {}
                    for name in NULL_NAMES:
                        scale = (
                            calibrations[key]["nulls"][name]["scale"]
                            if control == "downstream_kl" else rms[key][name]["scale"]
                        )
                        site_evaluations[control][name] = score_arm(
                            sa, rows, twall, all_attention, all_mlps, rare_vocab,
                            site=site, basis=arms[name]["basis"].to(sa.DEV), scale=scale,
                        )
                        print(
                            f"PCA control {split} site={site} control={control} "
                            f"arm={name} done", flush=True,
                        )
                candidate_gain = sa._paired_bootstrap_gain(
                    baseline["row_global_ce"], site_evaluations["local_pca"]["row_global_ce"],
                    161803 + 10000 * site + (0 if split == "discovery" else 5000)
                    + sum(map(ord, "local_pca")),
                )
                full_gain = sa._paired_bootstrap_gain(
                    baseline["row_global_ce"], site_evaluations["full"]["row_global_ce"],
                    161803 + 10000 * site + (0 if split == "discovery" else 5000)
                    + sum(map(ord, "full")),
                )
                gains[key].setdefault("local_pca", {})[split] = candidate_gain
                gains[key].setdefault("full", {})[split] = full_gain
                for control in ("downstream_kl", "raw_rms"):
                    control_gains = gains[key].setdefault(control, {})
                    for name in NULL_NAMES:
                        control_gains.setdefault(name, {})[split] = sa._paired_bootstrap_gain(
                            baseline["row_global_ce"],
                            site_evaluations[control][name]["row_global_ce"],
                            161803 + 10000 * site
                            + (0 if split == "discovery" else 5000)
                            + sum(map(ord, f"{control}_{name}")),
                        )
                write_json_atomic(partial, RESULT)

        decisions: dict[str, Any] = {}
        for site in SITES:
            key = str(site)
            decisions[key] = {}
            for control in ("downstream_kl", "raw_rms"):
                decisions[key][control] = analyze_strength_control(
                    gains[key]["local_pca"], gains[key][control],
                    full_heldout_gain=gains[key]["full"]["heldout"]["mean"],
                    bootstrap_ci95=gains[key]["local_pca"]["heldout"]["ci95"],
                )
            decisions[key]["passes_both_strength_controls"] = all(
                decisions[key][control]["decision"]["passes"]
                for control in ("downstream_kl", "raw_rms")
            )
        predictions = {
            "pred_a_mlp0_local_interface_passes": decisions["0"]["passes_both_strength_controls"],
            "pred_b_mlp1_local_interface_passes": decisions["1"]["passes_both_strength_controls"],
            "pred_c_both_strength_controls_agree_at_both_sites": all(
                decisions[str(site)]["downstream_kl"]["decision"]["passes"]
                == decisions[str(site)]["raw_rms"]["decision"]["passes"]
                for site in SITES
            ),
        }
        result = {
            **partial,
            "status": "completed_exploratory_only",
            "interpretation_guardrail": manifest["scope_guardrail"],
            "config": {
                "sites": list(SITES),
                "projection_rank": 64,
                "support_rank": 256,
                "nulls_per_control_per_site": 20,
                "spare_calibration_rows": len(splits["spare"]["rows"]),
                "evaluation_rows": {name: len(splits[name]["rows"])
                                    for name in ("discovery", "heldout")},
                "same_null_identities_across_controls": True,
            },
            "realization_component_hashes": expected,
            "site_decisions": decisions,
            "registered_predictions": predictions,
            "previous_result_reference": {
                "path": str(PREVIOUS_RESULT.resolve()),
                "sha256": PREVIOUS_RESULT_SHA256,
                "previous_local_pca_heldout_gains": {
                    str(site): previous["paired_gains"][str(site)]["heldout"]
                    ["local_pca"]["global"]["mean"] for site in SITES
                },
                "rule": "Reference only; all decision inputs were rescored in this run.",
            },
            "runtime_s": round(time.time() - start_time, 1),
        }
        if file_sha256(SCALE_RECEIPT) != scale_receipt_hash:
            raise RuntimeError("scale receipt changed during evaluation")
        if file_sha256(SAVED_BASES) != SAVED_BASES_SHA256:
            raise RuntimeError("saved oracle bases mutated during evaluation")
        write_json_atomic(result, RESULT)
        manifest.update({
            "status": "completed_exploratory_only",
            "realization_component_hashes": expected,
            "result_path": str(RESULT.resolve()),
            "result_sha256": file_sha256(RESULT),
            "runtime_s": result["runtime_s"],
        })
        write_json_atomic(manifest, MANIFEST)
        print(json.dumps({
            "site_decisions": decisions,
            "registered_predictions": predictions,
            "training_license_sites": [],
            "code_ood_licensed": False,
        }, indent=2), flush=True)

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)


def mark_failed(error: BaseException, canonical_after: dict[str, str | None]) -> None:
    try:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    except Exception:
        manifest = {}
    manifest.update({
        "schema_version": 1,
        "status": "failed_exploratory_run",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "training_license_sites": [],
        "code_ood_licensed": False,
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "canonical_paths_after": canonical_after,
        "recovery": "Preserve these artifacts and use a new versioned namespace for any retry.",
    })
    write_json_atomic(manifest, MANIFEST)
    if RESULT.exists():
        try:
            result = json.loads(RESULT.read_text())
        except Exception:
            result = {}
        result.update({
            "status": "failed_exploratory_run",
            "invalidated_by_failure": True,
            "authority": "none",
            "authorized_for_scored_experiments": False,
            "training_license_sites": [],
            "code_ood_licensed": False,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        write_json_atomic(result, RESULT)


def main() -> None:
    if RESULT.exists() or MANIFEST.exists() or SCALE_RECEIPT.exists():
        raise RuntimeError("refusing to overwrite existing PCA strength-control artifacts")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"PCA strength-control launch already claimed: {LOCK}") from error
    canonical_before = protected_snapshot()
    run_error: BaseException | None = None
    try:
        run_claimed(canonical_before)
    except BaseException as error:
        run_error = error
    canonical_after = protected_snapshot()
    try:
        if canonical_after != canonical_before:
            contamination = RuntimeError("PCA strength-control run changed protected artifacts")
            mark_failed(contamination, canonical_after)
            raise contamination from run_error
        if run_error is not None:
            mark_failed(run_error, canonical_after)
            raise run_error
        manifest = json.loads(MANIFEST.read_text())
        manifest["canonical_paths_after"] = canonical_after
        manifest["canonical_paths_unchanged"] = True
        write_json_atomic(manifest, MANIFEST)
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
