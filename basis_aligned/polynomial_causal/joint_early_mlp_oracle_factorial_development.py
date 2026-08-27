#!/usr/bin/env python3
"""Run the preregistered joint MLP0--2 live-oracle factorial, development-only.

This rebuilds the exact curated-v2 ship deterministically, verifies each frozen
ship component against the saved v2 realization, and then scores all eight
subsets of exact live MLP0/1/2 correction on the frozen discovery and heldout
rows.  It is authority-none evidence and cannot write a canonical oracle path.
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


HERE = Path(__file__).resolve().parent
TENSOR_ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "joint_early_mlp_oracle_factorial_v1_preregistration.json"
RESULT = BQ / "joint_early_mlp_oracle_factorial_curated_dev_v1_results.json"
MANIFEST = BQ / "joint_early_mlp_oracle_factorial_curated_dev_v1_manifest.json"
LOCK = Path("/workspace/runs/.bilin18_joint_early_mlp_oracle_factorial_curated_dev_v1.lock")
SAVED_SHIP = Path("/workspace/runs/bilin18_curated_dev_v2_ship.pt")
SAVED_SHIP_SHA256 = "85b848cc5d355bd99a29d43d7168f95113a11fb4c84a42fa9efe3393225dd530"
CORPUS_SHA256 = "faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd"
SHIP_SEED = 27182818
EXPECTED_COMPONENT_HASHES = {
    "ship": "c70106d20b8843ec64bb59e641e02b05335ed1437c8c119e5fa0b387c995a36a",
    "corr": "783fc65de3ae61a59e29238228eaff2660f3dede00b93612dbdc8ca33e5e8055",
    "attention": "b08302b629b53e3378f64afcf886faf623e87c8cb6e7b427e8e56b487e6e21f6",
    "all_attention": "a24255e6b168e35ba9da67e300c300d37fac3eb0ab98c851619d1cf4f2462ea7",
}
PROTECTED_EXISTING = (
    BQ / "ship_content_oracle_curated_dev_preregistration.json",
    BQ / "ship_content_oracle_curated_dev_manifest.json",
    BQ / "ship_content_oracle_curated_dev_v2_preregistration.json",
    BQ / "ship_content_oracle_curated_dev_v2_manifest.json",
    BQ / "ship_content_oracle_curated_dev_v2_results.json",
    Path("/workspace/runs/bilin18_curated_dev_ship.pt"),
    SAVED_SHIP,
    Path("/workspace/runs/bilin18_curated_dev_v2_oracle_bases.pt"),
    PREREG,
)

sys.path.insert(0, str(HERE))
import joint_early_mlp_oracle_factorial as joint  # noqa: E402
import local_ship_oracle_development as local  # noqa: E402
from factorial_causal_attribution import powerset  # noqa: E402


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
        if temporary.exists():
            temporary.unlink()


def protected_snapshot() -> dict[str, str | None]:
    paths = tuple(local.CANONICAL_PATHS) + PROTECTED_EXISTING
    return {str(path): file_sha256(path) if path.exists() else None for path in paths}


def tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda child: str(child)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(("list" if isinstance(item, list) else "tuple").encode() + b"\0")
            for child in item:
                update(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(repr(item).encode() + b"\0")
        else:
            raise TypeError(f"unsupported ship-state value for hashing: {type(item)}")

    update(value)
    return digest.hexdigest()


def realization_component_hashes(payload: dict[str, Any]) -> dict[str, str]:
    required = ("ship", "corr", "attention", "all_attention")
    missing = set(required).difference(payload)
    if missing:
        raise RuntimeError(f"saved ship missing components: {sorted(missing)}")
    return {
        key: tensor_tree_sha256(payload[key])
        for key in required
    }


def current_component_hashes(sa: Any, twall: dict, all_attention: frozenset[int]) -> dict[str, str]:
    return realization_component_hashes({
        "ship": sa.SHIP,
        "corr": sa.CORR,
        "attention": twall,
        "all_attention": sorted(all_attention),
    })


def verify_realization(
    saved: dict[str, Any], current_hashes: dict[str, str],
    pinned_hashes: dict[str, str] = EXPECTED_COMPONENT_HASHES,
) -> dict[str, str]:
    expected = realization_component_hashes(saved)
    if expected != pinned_hashes:
        raise RuntimeError(
            f"saved curated-v2 component pins changed: expected={pinned_hashes} "
            f"observed={expected}"
        )
    if current_hashes != expected:
        mismatches = {
            key: {"expected": expected[key], "observed": current_hashes.get(key)}
            for key in expected if current_hashes.get(key) != expected[key]
        }
        raise RuntimeError(f"fresh ship does not match curated-v2 realization: {mismatches}")
    return expected


def score_decisions(split_analyses: dict[str, dict[str, Any]]) -> dict[str, bool]:
    heldout = split_analyses["heldout"]
    required = [
        heldout["joint_gain"],
        heldout["mlp2_conditional_marginal_after_mlp0_mlp1"],
        heldout["interaction_l1_fraction_of_joint_gain"],
        *(heldout["gain_by_arm"][group] for group in joint.EARLY_MLP_GROUPS),
    ]
    if not all(math.isfinite(float(value)) for value in required):
        raise ValueError("registered decision inputs must be finite")
    singleton_best = max(
        heldout["gain_by_arm"][group] for group in joint.EARLY_MLP_GROUPS
    )
    return {
        "pred_a_joint_gain_exceeds_best_singleton": heldout["joint_gain"] > singleton_best,
        "pred_b_mlp2_conditional_marginal_after_mlp0_mlp1_is_positive": (
            heldout["mlp2_conditional_marginal_after_mlp0_mlp1"] > 0.0
        ),
        "pred_c_interaction_l1_fraction_is_at_least_0p20": (
            heldout["interaction_l1_fraction_of_joint_gain"] >= 0.20
        ),
    }


def run_claimed(canonical_before: dict[str, str | None]) -> None:
    if file_sha256(local.CORPUS) != CORPUS_SHA256:
        raise RuntimeError("curated corpus hash changed")
    if file_sha256(SAVED_SHIP) != SAVED_SHIP_SHA256:
        raise RuntimeError("saved curated-v2 ship file hash changed")
    pinned_inputs = {
        local.GLUE: local.GLUE_SHA256,
        local.MODEL_SNAPSHOT / "config.json": local.MODEL_CONFIG_SHA256,
        local.MODEL_SNAPSHOT / "pytorch_model.bin": local.MODEL_WEIGHTS_SHA256,
    }
    for path, expected_hash in pinned_inputs.items():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"pinned input hash changed: {path}")
    prereg_hash = file_sha256(PREREG)
    payload = torch.load(local.CORPUS, map_location="cpu", weights_only=True)
    splits = local.allocate_whole_document_splits(payload)
    saved = torch.load(SAVED_SHIP, map_location="cpu", weights_only=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running_development_only",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "training_license_sites": [],
        "scope_guardrail": "Within-realization joint causal signs only; no fresh-corpus, FineWeb, code-OOD, training, generalization, or simplicity claim.",
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": prereg_hash,
        "saved_ship_path": str(SAVED_SHIP),
        "saved_ship_file_sha256": SAVED_SHIP_SHA256,
        "corpus_path": str(local.CORPUS.resolve()),
        "corpus_sha256": CORPUS_SHA256,
        "canonical_paths_before": canonical_before,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=TENSOR_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_hashes": {
            "runner": file_sha256(Path(__file__)),
            "joint_contract": file_sha256(HERE / "joint_early_mlp_oracle_factorial.py"),
            "ship_error_attrib": file_sha256(BQ / "ship_error_attrib.py"),
            "local_split_allocator": file_sha256(HERE / "local_ship_oracle_development.py"),
        },
        "pinned_input_hashes": {str(path): value for path, value in pinned_inputs.items()},
        "row_splits": {
            role: local.split_receipt(row)
            for role, row in splits.items()
        },
    }
    write_json_atomic(manifest, MANIFEST)

    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    request_roles = local.REQUEST_ROLES

    def frozen_local_rows(n: int = 120, skip: int = 0) -> torch.Tensor:
        role = request_roles.get((n, skip))
        if role is None:
            raise RuntimeError(f"unregistered curated row request {(n, skip)}")
        return splits[role]["rows"].clone()

    torch.manual_seed(SHIP_SEED)
    torch.cuda.manual_seed_all(SHIP_SEED)
    sa.cl.fineweb_rows = frozen_local_rows
    start_time = time.time()

    def factorial_callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        observed_hashes = current_component_hashes(sa, twall, all_attention)
        expected_hashes = verify_realization(saved, observed_hashes)
        all_mlps = frozenset(range(18))
        rare_vocab = sa._token_masks(splits["discovery"]["rows"])
        evaluations: dict[str, Any] = {}
        analyses: dict[str, Any] = {}
        partial = {
            "schema_version": 1,
            "status": "running_exploratory_only",
            "authority": "none",
            "authorized_for_scored_experiments": False,
            "training_license_sites": [],
            "evaluations": evaluations,
            "split_analyses": analyses,
            "preregistration_sha256": prereg_hash,
        }
        write_json_atomic(partial, RESULT)
        try:
            for split_name in ("discovery", "heldout"):
                rows = splits[split_name]["rows"]
                row_ce_by_arm: dict[tuple[str, ...], list[float]] = {}
                evaluations[split_name] = {}
                for arm in powerset(joint.EARLY_MLP_GROUPS):
                    correction_map = {
                        int(group[-1]): {"basis": None, "scale": 1.0}
                        for group in arm
                    }
                    joint.configure_oracle_corrections(sa.ORACLE_CORR, correction_map)
                    scored = sa._score_content_rows(
                        rows, twall, all_attention, all_mlps,
                        rare_vocab=rare_vocab, retain_row_ce=True,
                    )
                    name = "+".join(arm) if arm else "baseline"
                    evaluations[split_name][name] = scored
                    row_ce_by_arm[arm] = scored["row_global_ce"]
                    write_json_atomic(partial, RESULT)
                    print(f"joint oracle {split_name} arm={name} done", flush=True)
                analyses[split_name] = joint.analyze_full_live_subset_rows(row_ce_by_arm)
                write_json_atomic(partial, RESULT)
        finally:
            joint.clear_oracle_corrections(sa.ORACLE_CORR)

        decisions = score_decisions(analyses)
        result = {
            "schema_version": 1,
            "status": "completed_exploratory_only",
            "authority": "none",
            "authorized_for_scored_experiments": False,
            "training_license_sites": [],
            "interpretation_guardrail": manifest["scope_guardrail"],
            "config": {
                "groups": list(joint.EARLY_MLP_GROUPS),
                "arms": ["+".join(arm) if arm else "baseline"
                         for arm in powerset(joint.EARLY_MLP_GROUPS)],
                "row_counts": {name: len(splits[name]["rows"])
                               for name in ("discovery", "heldout")},
                "intervention": "exact live original-minus-deployed-plank at every selected site on that arm's current state",
                "same_currency_residual_denominator": None,
                "denominator_rule": "No cross-run MLP0-2 denominator imported; recovery fraction remains null.",
            },
            "realization_component_hashes": expected_hashes,
            "evaluations": evaluations,
            "split_analyses": analyses,
            "registered_predictions": decisions,
            "runtime_s": round(time.time() - start_time, 1),
            "preregistration_sha256": prereg_hash,
        }
        write_json_atomic(result, RESULT)
        manifest.update({
            "status": "completed_exploratory_only",
            "realization_component_hashes": expected_hashes,
            "result_path": str(RESULT.resolve()),
            "result_sha256": file_sha256(RESULT),
            "runtime_s": result["runtime_s"],
        })
        write_json_atomic(manifest, MANIFEST)
        print(json.dumps({
            "split_analyses": analyses,
            "registered_predictions": decisions,
            "training_license_sites": [],
        }, indent=2), flush=True)

    sa.run_oracle_content_screen = factorial_callback
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
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "canonical_paths_after": canonical_after,
        "recovery": "Preserve these artifacts; diagnose and use a new versioned namespace for any retry.",
    })
    write_json_atomic(manifest, MANIFEST)
    if RESULT.exists():
        try:
            result = json.loads(RESULT.read_text())
        except Exception:
            result = {}
        result.update({
            "status": "failed_exploratory_run",
            "authority": "none",
            "authorized_for_scored_experiments": False,
            "training_license_sites": [],
            "invalidated_by_failure": True,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        write_json_atomic(result, RESULT)


def main() -> None:
    if RESULT.exists() or MANIFEST.exists():
        raise RuntimeError("refusing to overwrite existing joint factorial artifacts")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"joint factorial launch already claimed: {LOCK}") from error
    canonical_before = protected_snapshot()
    run_error: BaseException | None = None
    try:
        run_claimed(canonical_before)
    except BaseException as error:
        run_error = error
    canonical_after = protected_snapshot()
    try:
        if canonical_after != canonical_before:
            contamination = RuntimeError("joint development run changed canonical artifacts")
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
