#!/usr/bin/env python3
"""Execute the preregistered exact-versus-projected early-MLP composition cubes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
TENSOR_ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "joint_early_mlp_pca_composition_v1_preregistration.json"
RESULT = BQ / "joint_early_mlp_pca_composition_v1_results.json"
MANIFEST = BQ / "joint_early_mlp_pca_composition_v1_manifest.json"
LOCK = Path("/workspace/runs/.bilin18_joint_early_mlp_pca_composition_v1.lock")
SAVED_SHIP = Path("/workspace/runs/bilin18_curated_dev_v2_ship.pt")
SAVED_BASES = Path("/workspace/runs/bilin18_curated_dev_v2_oracle_bases.pt")
EXACT_RESULT = BQ / "joint_early_mlp_oracle_factorial_curated_dev_v2_results.json"
PCA_RESULT = BQ / "oracle_local_pca_strength_control_v1_results.json"
SAVED_SHIP_SHA256 = "85b848cc5d355bd99a29d43d7168f95113a11fb4c84a42fa9efe3393225dd530"
SAVED_BASES_SHA256 = "aa086ed4a14ea0474882a5443eb0e9add21734649e705398eebf3fce7a09b801"
EXACT_RESULT_SHA256 = "2092cf602ebfb5bbf4459b7a42196e689cc6e5d744de928f9dfe1be9fdd26e55"
PCA_RESULT_SHA256 = "46c7c2556838b79ccbdac892179b4663e8a74b754b16b622b53d6b9b1204826a"
CORPUS_SHA256 = "faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd"
SHIP_SEED = 27182818
BOOTSTRAP_SEED = 424242
EXPECTED_COMPONENT_HASHES = {
    "ship": "c70106d20b8843ec64bb59e641e02b05335ed1437c8c119e5fa0b387c995a36a",
    "corr": "783fc65de3ae61a59e29238228eaff2660f3dede00b93612dbdc8ca33e5e8055",
    "attention": "b08302b629b53e3378f64afcf886faf623e87c8cb6e7b427e8e56b487e6e21f6",
    "all_attention": "a24255e6b168e35ba9da67e300c300d37fac3eb0ab98c851619d1cf4f2462ea7",
}
PROTECTED_EXISTING = (
    SAVED_SHIP,
    SAVED_BASES,
    EXACT_RESULT,
    PCA_RESULT,
    PREREG,
    BQ / "joint_early_mlp_oracle_factorial_curated_dev_v2_manifest.json",
    BQ / "oracle_local_pca_strength_control_v1_manifest.json",
    BQ / "oracle_local_pca_strength_control_v1_scale_receipt.json",
)

sys.path.insert(0, str(HERE))
import joint_early_mlp_oracle_factorial as joint  # noqa: E402
import joint_early_mlp_oracle_factorial_development as exact_runner  # noqa: E402
from joint_early_mlp_pca_composition import analyze_composition  # noqa: E402
import local_ship_oracle_development as local  # noqa: E402
import oracle_local_pca_strength_control_development as pca_runner  # noqa: E402
from factorial_causal_attribution import powerset  # noqa: E402


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
    return {
        str(path): exact_runner.file_sha256(path) if path.exists() else None
        for path in paths
    }


def arm_name(arm: tuple[str, ...]) -> str:
    return "+".join(arm) if arm else "baseline"


def max_abs_row_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("row-CE reproduction has mismatched lengths")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def score(sa: Any, rows: torch.Tensor, twall: dict,
          all_attention: frozenset[int], all_mlps: frozenset[int],
          rare_vocab: torch.Tensor, corrections: dict[int, dict[str, Any]]) -> dict[str, Any]:
    joint.configure_oracle_corrections(sa.ORACLE_CORR, corrections)
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
        EXACT_RESULT: EXACT_RESULT_SHA256,
        PCA_RESULT: PCA_RESULT_SHA256,
        local.GLUE: local.GLUE_SHA256,
        local.MODEL_SNAPSHOT / "config.json": local.MODEL_CONFIG_SHA256,
        local.MODEL_SNAPSHOT / "pytorch_model.bin": local.MODEL_WEIGHTS_SHA256,
    }
    for path, expected in pinned.items():
        observed = exact_runner.file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"pinned input hash changed for {path}: {observed}")
    prereg_hash = exact_runner.file_sha256(PREREG)
    splits = local.allocate_whole_document_splits(
        torch.load(local.CORPUS, map_location="cpu", weights_only=True)
    )
    saved_ship = torch.load(SAVED_SHIP, map_location="cpu", weights_only=True)
    saved_bases = torch.load(SAVED_BASES, map_location="cpu", weights_only=True)
    pca_runner.validate_serialized_bases(saved_bases)
    exact_prior = json.loads(EXACT_RESULT.read_text())
    pca_prior = json.loads(PCA_RESULT.read_text())
    if not all(pca_prior["registered_predictions"].values()):
        raise RuntimeError("pinned PCA control did not pass its registered predictions")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running_development_only",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "training_license_sites": [],
        "code_ood_licensed": False,
        "scope_guardrail": "Coupled oracle-interface composition only; no fresh-corpus, semantic, simplicity, coefficient-predictor, MLP2-subspace, training, OOD, or generalization claim.",
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": prereg_hash,
        "canonical_paths_before": canonical_before,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=TENSOR_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_hashes": {
            "runner": exact_runner.file_sha256(Path(__file__)),
            "composition_contract": exact_runner.file_sha256(
                HERE / "joint_early_mlp_pca_composition.py"
            ),
            "joint_contract": exact_runner.file_sha256(
                HERE / "joint_early_mlp_oracle_factorial.py"
            ),
            "ship_error_attrib": exact_runner.file_sha256(BQ / "ship_error_attrib.py"),
        },
        "pinned_input_hashes": {str(path): value for path, value in pinned.items()},
        "row_splits": exact_runner.row_split_receipts(splits),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    write_json_atomic(manifest, MANIFEST)

    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415
    import source_global_preflight  # noqa: PLC0415

    source_global_preflight.require_defined_globals([
        Path(__file__), HERE / "joint_early_mlp_pca_composition.py",
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
        observed = exact_runner.current_component_hashes(sa, twall, all_attention)
        expected = exact_runner.verify_realization(saved_ship, observed)
        all_mlps = frozenset(range(18))
        rare_vocab = saved_bases["rare_vocab"].to(sa.DEV)
        evaluations: dict[str, Any] = {}
        exact_analyses: dict[str, Any] = {}
        projected_analyses: dict[str, Any] = {}
        reproduction: dict[str, Any] = {}
        row_cubes: dict[str, dict[str, dict[tuple[str, ...], list[float]]]] = {}
        partial: dict[str, Any] = {
            "schema_version": 1,
            "status": "running_exploratory_only",
            "authority": "none",
            "authorized_for_scored_experiments": False,
            "training_license_sites": [],
            "code_ood_licensed": False,
            "evaluations": evaluations,
            "exact_cube_analyses": exact_analyses,
            "projected_cube_analyses": projected_analyses,
            "exact_v2_row_reproduction": reproduction,
            "preregistration_sha256": prereg_hash,
        }
        write_json_atomic(partial, RESULT)
        for split in ("discovery", "heldout"):
            rows = splits[split]["rows"]
            evaluations[split] = {"exact": {}, "projected": {}}
            exact_rows: dict[tuple[str, ...], list[float]] = {}
            projected_rows: dict[tuple[str, ...], list[float]] = {}
            reproduction[split] = {"exact_cube": {}, "pca_singletons": {}}
            for arm in powerset(joint.EARLY_MLP_GROUPS):
                correction_map = {
                    int(group[-1]): {"basis": None, "scale": 1.0}
                    for group in arm
                }
                scored = score(
                    sa, rows, twall, all_attention, all_mlps, rare_vocab,
                    correction_map,
                )
                name = arm_name(arm)
                evaluations[split]["exact"][name] = scored
                exact_rows[arm] = scored["row_global_ce"]
                prior_rows = exact_prior["evaluations"][split][name]["row_global_ce"]
                error = max_abs_row_error(scored["row_global_ce"], prior_rows)
                if error > 1e-6:
                    raise RuntimeError(
                        f"exact-v2 row reproduction failed split={split} arm={name}: {error}"
                    )
                reproduction[split]["exact_cube"][name] = error
                write_json_atomic(partial, RESULT)
                print(f"PCA composition exact {split} arm={name} done", flush=True)
            exact_analyses[split] = joint.analyze_full_live_subset_rows(exact_rows)

            for arm in powerset(joint.EARLY_MLP_GROUPS):
                name = arm_name(arm)
                if arm in ((), ("mlp2",)):
                    scored = evaluations[split]["exact"][name]
                    evaluations[split]["projected"][name] = {
                        **scored,
                        "shared_exact_arm": True,
                    }
                    projected_rows[arm] = exact_rows[arm]
                    continue
                correction_map: dict[int, dict[str, Any]] = {}
                for group in arm:
                    site = int(group[-1])
                    basis = (
                        saved_bases["sites"][site]["local_pca"]["basis"].to(sa.DEV)
                        if site in (0, 1) else None
                    )
                    correction_map[site] = {"basis": basis, "scale": 1.0}
                scored = score(
                    sa, rows, twall, all_attention, all_mlps, rare_vocab,
                    correction_map,
                )
                evaluations[split]["projected"][name] = scored
                projected_rows[arm] = scored["row_global_ce"]
                if arm in (("mlp0",), ("mlp1",)):
                    site = int(arm[0][-1])
                    prior_rows = pca_prior["evaluations"][split]["sites"][str(site)][
                        "local_pca"
                    ]["row_global_ce"]
                    error = max_abs_row_error(scored["row_global_ce"], prior_rows)
                    if error > 1e-6:
                        raise RuntimeError(
                            f"PCA-v1 singleton row reproduction failed split={split} "
                            f"arm={name}: {error}"
                        )
                    reproduction[split]["pca_singletons"][name] = error
                write_json_atomic(partial, RESULT)
                print(f"PCA composition projected {split} arm={name} done", flush=True)
            projected_analyses[split] = joint.analyze_full_live_subset_rows(projected_rows)
            row_cubes[split] = {"exact": exact_rows, "projected": projected_rows}
            write_json_atomic(partial, RESULT)

        heldout_baseline = row_cubes["heldout"]["projected"][()]
        upstream_arm = ("mlp0", "mlp1")
        joint_arm = joint.EARLY_MLP_GROUPS
        upstream_bootstrap = sa._paired_bootstrap_gain(
            heldout_baseline,
            row_cubes["heldout"]["projected"][upstream_arm],
            BOOTSTRAP_SEED + 1,
        )
        joint_bootstrap = sa._paired_bootstrap_gain(
            heldout_baseline,
            row_cubes["heldout"]["projected"][joint_arm],
            BOOTSTRAP_SEED + 2,
        )
        composition = analyze_composition(
            exact_analyses, projected_analyses,
            upstream_heldout_ci95=upstream_bootstrap["ci95"],
            joint_heldout_ci95=joint_bootstrap["ci95"],
        )
        result = {
            **partial,
            "status": "completed_exploratory_only",
            "interpretation_guardrail": manifest["scope_guardrail"],
            "config": {
                "groups": list(joint.EARLY_MLP_GROUPS),
                "exact_cube": "unrestricted exact live restoration at MLP0/1/2",
                "projected_cube": "rank-64 frozen local PCA at MLP0/1 plus unrestricted exact MLP2",
                "projection_scale": 1.0,
                "same_run_exact_denominators": True,
                "row_counts": {name: len(splits[name]["rows"])
                               for name in ("discovery", "heldout")},
            },
            "realization_component_hashes": expected,
            "composition_analysis": composition,
            "registered_predictions": composition["registered_predictions"],
            "runtime_s": round(time.time() - start_time, 1),
        }
        if exact_runner.file_sha256(SAVED_BASES) != SAVED_BASES_SHA256:
            raise RuntimeError("saved oracle bases mutated during composition scoring")
        write_json_atomic(result, RESULT)
        manifest.update({
            "status": "completed_exploratory_only",
            "realization_component_hashes": expected,
            "result_path": str(RESULT.resolve()),
            "result_sha256": exact_runner.file_sha256(RESULT),
            "runtime_s": result["runtime_s"],
        })
        write_json_atomic(manifest, MANIFEST)
        print(json.dumps({
            "composition_analysis": composition,
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
        "recovery": "Preserve these artifacts and use a new versioned namespace for retry.",
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
    if RESULT.exists() or MANIFEST.exists():
        raise RuntimeError("refusing to overwrite existing PCA composition artifacts")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"PCA composition launch already claimed: {LOCK}") from error
    canonical_before = protected_snapshot()
    run_error: BaseException | None = None
    try:
        run_claimed(canonical_before)
    except BaseException as error:
        run_error = error
    canonical_after = protected_snapshot()
    try:
        if canonical_after != canonical_before:
            contamination = RuntimeError("PCA composition run changed protected artifacts")
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
