#!/usr/bin/env python3
"""Run the preregistered authoritative exact-live MLP0--2 factorial.

The runner consumes only the validated local FineWeb receipt, freezes the exact
ship realization before scoring, and evaluates the complete Boolean cube on the
registered discovery and heldout rows.  Exact residual restoration is a causal
ceiling, never a learned replacement or training license.
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
from typing import Any, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
TENSOR_ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "joint_early_mlp_oracle_factorial_authoritative_v3_preregistration.json"
RESULT = BQ / "joint_early_mlp_oracle_factorial_authoritative_v3_results.json"
MANIFEST = BQ / "joint_early_mlp_oracle_factorial_authoritative_v3_manifest.json"
LOCK = Path("/workspace/runs/.bilin18_joint_early_mlp_oracle_factorial_authoritative_v3.lock")
FROZEN_STATE = Path("/workspace/runs/bilin18_frozen_ship_v2.pt")
FROZEN_MANIFEST = Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json")
SHIP_SEED = 27182818
BOOTSTRAP_SEED = 27182819
BOOTSTRAP_DRAWS = 2000
PREREG_SHA256 = "41f3a614666ee911d028f0b559c9a4c469369bb8786aec61b3d6836b6d927419"
MODEL_SNAPSHOT = Path(
    "/workspace/.hf_home/hub/"
    "models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/"
    "snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240"
)
PINNED_INPUTS = {
    PREREG: PREREG_SHA256,
    BQ / ".rowcache/fineweb_oracle_v2_receipt.json": (
        "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16"
    ),
    HERE / "code_oracle_corpus_v2.pt": (
        "6750a72b4232d4d4687946bb379457210555a660c9ef2e0d4967a63ddfaf2d9d"
    ),
    HERE / "code_oracle_corpus_v2_manifest.json": (
        "a19def47d44a581a72a6cb8f0d91ac7ae0bb2121007f43964f4f1dcb526cb9ec"
    ),
    MODEL_SNAPSHOT / "config.json": (
        "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
    ),
    MODEL_SNAPSHOT / "pytorch_model.bin": (
        "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
    ),
    BQ / "mlp2_glue_params.pt": (
        "76148b072c22f3c0d0ccdcaa08d8a6ade89d7231d0dd5a328597e10f6a0a3ef4"
    ),
    BQ / "ship_error_attrib.py": (
        "f966110fe53c8997bd5eadaa3ea415130dcb860b3104d4295b8a72b4a5254c8b"
    ),
    HERE / "joint_early_mlp_oracle_factorial.py": (
        "ace57bbbd244a5c33ebb97a09d962e3e763ff985a19333a5c2624e2a413625ac"
    ),
    HERE / "frozen_ship_oracle_v2.py": (
        "d12deb10ed12485f8ae83897a58f4f5bf862df27ad6e4df894644c2da1d71df1"
    ),
    HERE / "prepare_fineweb_oracle_rows.py": (
        "67362c609eaef883b8f58e09c1fc9fc2b7503cee0e2a58b8933fcff8fa9236ca"
    ),
    HERE / "code_ood_oracle.py": (
        "13a8f771a7ff47ee7bb43f55bceec63cb3740f255d3894cd0983cf7385bb7397"
    ),
    HERE / "factorial_causal_attribution.py": (
        "f5a8a5f01342043280aa301dfb16c9e56118e253e93413cd3e53ce605ccc0538"
    ),
    HERE / "source_global_preflight.py": (
        "7b8e6b036eda14e7eb15e0ff33e2776664ab60a73ce5e905c59711aa5154bb2e"
    ),
    HERE / "oracle_authority.py": (
        "24a2940af0588ae75e0446c9239566ed69d324be5c745b40c94d9e4825267518"
    ),
    BQ / "bilin18_joint_removal.py": (
        "0ff3d3bd22b2819e60aa0e3ff82226c541735a3b5b43c82b032394eb519af594"
    ),
    BQ / "census_lib.py": (
        "f51c19e83f46dc363a2c5dad1887b55ab5dd9b3684294e940583a6814881cf1f"
    ),
}
ROW_ROLES = {
    "ship_fit": (480, 80),
    "basis": (96, 1200),
    "discovery": (192, 7000),
    "heldout": (192, 11000),
}
PREFIX257_SHA256 = {
    "ship_fit": "e7b646d5218d3dcc093cb6a37ef4adf17becb907ae87b1943025be9a0107fb7c",
    "basis": "821ce3ce4ecaf6ea63319270d40c56107c7eaafde8636539216b66198fae1d3e",
    "discovery": "4e69a0d692a20386b7d3d978f2d458c92315817df1ff5ee08d6447cd3999cb83",
    "heldout": "83e1b7db4b9d92c243d68ab1f7349664d1425ee4a315700d177c8d714471733a",
}
UNIQUE_DOCUMENT_COUNTS = {
    "ship_fit": 209, "basis": 33, "discovery": 79, "heldout": 105,
}
PROTECTED_EXISTING = (
    BQ / "ship_content_oracle_screen_results.json",
    BQ / "ship_content_oracle_screen_preliminary_results.json",
    BQ / "joint_early_mlp_oracle_factorial_curated_dev_v2_results.json",
    BQ / "joint_early_mlp_oracle_factorial_curated_dev_v2_manifest.json",
    BQ / "oracle_local_pca_strength_control_v1_results.json",
    BQ / "oracle_local_pca_strength_control_v1_manifest.json",
    BQ / "oracle_local_pca_strength_control_v1_scale_receipt.json",
    HERE / "code_ood_oracle_results.json",
    HERE / "joint_early_mlp_pca_composition_v1_preregistration.json",
)
SOURCE_CLOSURE = (
    Path(__file__),
    HERE / "test_joint_early_mlp_oracle_factorial_authoritative.py",
    HERE / "joint_early_mlp_oracle_factorial.py",
    HERE / "factorial_causal_attribution.py",
    HERE / "frozen_ship_oracle_v2.py",
    HERE / "prepare_fineweb_oracle_rows.py",
    HERE / "code_ood_oracle.py",
    HERE / "source_global_preflight.py",
    HERE / "oracle_authority.py",
    BQ / "ship_error_attrib.py",
    BQ / "bilin18_joint_removal.py",
    BQ / "census_lib.py",
)

sys.path.insert(0, str(HERE))
import code_ood_oracle as code_oracle  # noqa: E402
from factorial_causal_attribution import powerset  # noqa: E402
import frozen_ship_oracle_v2 as frozen  # noqa: E402
import joint_early_mlp_oracle_factorial as joint  # noqa: E402
import prepare_fineweb_oracle_rows as row_prep  # noqa: E402
import source_global_preflight  # noqa: E402


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


def protected_snapshot() -> dict[str, str | None]:
    paths = PROTECTED_EXISTING + tuple(PINNED_INPUTS)
    return {
        str(path): file_sha256(path) if path.exists() else None
        for path in paths
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


def frozen_lifecycle_receipt(row_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state_exists = FROZEN_STATE.is_file()
    manifest_exists = FROZEN_MANIFEST.is_file()
    output: dict[str, Any] = {
        "state_exists": state_exists,
        "manifest_exists": manifest_exists,
        "pair_consistent_by_existence": state_exists == manifest_exists,
        "state_sha256": file_sha256(FROZEN_STATE) if state_exists else None,
        "manifest_sha256": file_sha256(FROZEN_MANIFEST) if manifest_exists else None,
        "validated": False,
    }
    if state_exists and manifest_exists and row_receipt is not None:
        try:
            payload, manifest = frozen.validate_frozen_ship_pair(dict(row_receipt))
            output.update({
                "validated": True,
                "ship_realization_sha256": payload["ship_realization_sha256"],
                "artifact_sha256": manifest["artifact_sha256"],
            })
        except Exception as error:
            output["validation_error"] = f"{type(error).__name__}: {error}"
    return output


def component_tree_sha256(sa: Any, twall: Mapping[str, Any], all_attention: frozenset[int]) -> str:
    return code_oracle.tensor_tree_sha256({
        "TWALL": twall,
        "SHIP": sa.SHIP,
        "CORR": {key: sa.CORR[key] for key in ("on", "b", "U", "V")},
        "all_attention": sorted(all_attention),
    })


def require_inert_correction_state(sa: Any) -> None:
    if sa.CONTENT_CORR.get("on") is not False:
        raise RuntimeError("CONTENT_CORR must be inert before every factorial arm")
    if sa.ORACLE_CORR.get("capture") is not None:
        raise RuntimeError("ORACLE_CORR capture must be absent before every factorial arm")
    if sa.ORACLE_CORR.get("on") is not False:
        raise RuntimeError("ORACLE_CORR must be cleared before installing an arm")
    if sa.ORACLE_CORR.get("site") is not None or sa.ORACLE_CORR.get("corrections") is not None:
        raise RuntimeError("ORACLE_CORR retains a stale singleton or joint map")


@torch.no_grad()
def exact_patch_canary(sa: Any) -> dict[str, Any]:
    require_inert_correction_state(sa)
    values = torch.linspace(-1.0, 1.0, 2 * 3 * 1152, device=sa.DEV)
    output = {}
    try:
        for site in (0, 1, 2):
            block = sa.H[site]
            parameter_dtype = next(block.mlp.parameters()).dtype
            z = values.view(2, 3, 1152).to(parameter_dtype)
            original = block.mlp(z).float()
            deployed = torch.cos(values).view_as(original).float()
            joint.configure_oracle_corrections(
                sa.ORACLE_CORR, {site: {"basis": None, "scale": 1.0}}
            )
            patched = sa.add_oracle_correction(site, block, z, deployed).float()
            error = float((patched - original).abs().max())
            if error > 2e-5:
                raise RuntimeError(f"exact patch canary failed at MLP{site}: {error}")
            output[str(site)] = {"max_abs_float32_error": error, "tolerance": 2e-5}
            joint.clear_oracle_corrections(sa.ORACLE_CORR)
            require_inert_correction_state(sa)
    finally:
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
    return output


def tensor_prefix_sha256(tensor: torch.Tensor, width: int = 257) -> str:
    value = tensor[:, :width].detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def validate_document_provenance(
    receipt: Mapping[str, Any], rows: Mapping[tuple[int, int], torch.Tensor]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    provenance = receipt.get("document_provenance", {})
    if provenance.get("schema_version") != 1 or not isinstance(provenance.get("sets"), dict):
        raise RuntimeError("FineWeb receipt lacks row-aligned document provenance")
    document_ids: dict[str, list[str]] = {}
    receipts: dict[str, Any] = {}
    for role, spec in ROW_ROLES.items():
        tensor = rows[spec]
        key = row_prep.spec_key(*spec)
        records = provenance["sets"].get(key)
        if not isinstance(records, list) or len(records) != spec[0]:
            raise RuntimeError(f"invalid row-aligned provenance for {role}")
        ids = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("document_id"), str):
                raise RuntimeError(f"invalid provenance record {role}[{index}]")
            if not isinstance(record.get("chunk_id"), int) or not isinstance(
                record.get("token_start"), int
            ):
                raise RuntimeError(f"incomplete provenance record {role}[{index}]")
            ids.append(record["document_id"])
        prefix_hash = tensor_prefix_sha256(tensor)
        if prefix_hash != PREFIX257_SHA256[role]:
            raise RuntimeError(f"scored prefix hash changed for {role}")
        unique_count = len(set(ids))
        if unique_count != UNIQUE_DOCUMENT_COUNTS[role]:
            raise RuntimeError(f"unique document count changed for {role}")
        document_ids[role] = ids
        receipts[role] = {
            "request": {"n": spec[0], "skip": spec[1]},
            "shape_full": list(tensor.shape),
            "shape_scored_prefix": [spec[0], 257],
            "dtype": str(tensor.dtype),
            "tensor_full_raw_sha256": row_prep.tensor_sha256(tensor),
            "tensor_prefix257_raw_sha256": prefix_hash,
            "unique_document_count": unique_count,
            "document_ids_sha256": logical_json_sha256(ids),
            "provenance_records_sha256": logical_json_sha256(records),
        }
    for left_index, left in enumerate(ROW_ROLES):
        for right in tuple(ROW_ROLES)[left_index + 1:]:
            overlap = set(document_ids[left]).intersection(document_ids[right])
            if overlap:
                raise RuntimeError(
                    f"FineWeb document leakage between {left} and {right}: {len(overlap)}"
                )
    return document_ids, receipts


def _ci95(values: torch.Tensor) -> list[float]:
    return [
        float(torch.quantile(values, 0.025)),
        float(torch.quantile(values, 0.975)),
    ]


def paired_document_cluster_bootstrap(
    row_ce_by_arm: Mapping[tuple[str, ...], Sequence[float]],
    document_ids: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired document-cluster bootstrap retaining the row-weighted estimand."""

    arms = powerset(joint.EARLY_MLP_GROUPS)
    if set(row_ce_by_arm) != set(arms):
        raise ValueError("bootstrap requires the complete registered cube")
    values = torch.tensor(
        [[float(value) for value in row_ce_by_arm[arm]] for arm in arms],
        dtype=torch.float64,
    )
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError("bootstrap arms must contain positive paired row counts")
    if len(document_ids) != values.shape[1] or not all(
        isinstance(document, str) for document in document_ids
    ):
        raise ValueError("bootstrap document IDs must align one-to-one with rows")
    if not torch.isfinite(values).all():
        raise ValueError("bootstrap row CE must be finite")
    unique_documents = list(dict.fromkeys(document_ids))
    if len(unique_documents) < 2:
        raise ValueError("document-cluster bootstrap requires at least two documents")
    baseline = values[arms.index(())]
    gain_rows = baseline.unsqueeze(0) - values
    document_index = {document: index for index, document in enumerate(unique_documents)}
    cluster = torch.tensor(
        [document_index[document] for document in document_ids], dtype=torch.long
    )
    document_sums = torch.zeros(
        len(arms), len(unique_documents), dtype=torch.float64
    )
    document_sums.index_add_(1, cluster, gain_rows)
    document_row_counts = torch.bincount(
        cluster, minlength=len(unique_documents)
    ).to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    sampled_documents = torch.randint(
        len(unique_documents), (draws, len(unique_documents)), generator=generator
    )
    sampled_numerators = document_sums[:, sampled_documents].sum(dim=2).T
    sampled_denominators = document_row_counts[sampled_documents].sum(dim=1)
    boot_gain = sampled_numerators / sampled_denominators.unsqueeze(1)
    point_gain = gain_rows.mean(dim=1)
    arm_index = {arm: index for index, arm in enumerate(arms)}
    full = arm_index[joint.EARLY_MLP_GROUPS]
    upstream = arm_index[("mlp0", "mlp1")]
    singletons = [arm_index[(group,)] for group in joint.EARLY_MLP_GROUPS]
    joint_minus_best = boot_gain[:, full] - boot_gain[:, singletons].max(dim=1).values
    joint_minus_sum = boot_gain[:, full] - boot_gain[:, singletons].sum(dim=1)
    conditional_mlp2 = boot_gain[:, full] - boot_gain[:, upstream]
    mlp2_singleton = boot_gain[:, arm_index[("mlp2",)]]
    sign_flip = conditional_mlp2 - mlp2_singleton
    mobius_draws: dict[tuple[str, ...], torch.Tensor] = {}
    for arm in arms:
        total = torch.zeros(draws, dtype=torch.float64)
        for subset in powerset(arm):
            canonical = tuple(group for group in joint.EARLY_MLP_GROUPS if group in subset)
            total += ((-1.0) ** (len(arm) - len(subset))) * boot_gain[
                :, arm_index[canonical]
            ]
        mobius_draws[arm] = total

    def summary(point: float, bootstrap: torch.Tensor) -> dict[str, Any]:
        return {
            "point_estimate": float(point),
            "bootstrap_mean": float(bootstrap.mean()),
            "ci95": _ci95(bootstrap),
        }

    return {
        "draws": draws,
        "seed": seed,
        "resampling": (
            "sample unique documents with replacement; retain all chunks and use "
            "the sampled total row count as denominator; shared multiplicities across arms"
        ),
        "row_count": values.shape[1],
        "unique_document_count": len(unique_documents),
        "cluster_size_range": [
            int(document_row_counts.min()), int(document_row_counts.max())
        ],
        "arm_gain": {
            ("+".join(arm) if arm else "baseline"): summary(
                float(point_gain[index]), boot_gain[:, index]
            )
            for index, arm in enumerate(arms)
        },
        "joint_gain": summary(float(point_gain[full]), boot_gain[:, full]),
        "joint_minus_best_singleton": summary(
            float(point_gain[full] - point_gain[singletons].max()), joint_minus_best
        ),
        "joint_minus_singleton_sum": summary(
            float(point_gain[full] - point_gain[singletons].sum()), joint_minus_sum
        ),
        "mlp2_singleton_gain": summary(
            float(point_gain[arm_index[("mlp2",)]]), mlp2_singleton
        ),
        "mlp2_conditional_marginal_after_mlp0_mlp1": summary(
            float(point_gain[full] - point_gain[upstream]), conditional_mlp2
        ),
        "mlp2_sign_flip_contrast": summary(
            float(
                point_gain[full] - point_gain[upstream]
                - point_gain[arm_index[("mlp2",)]]
            ),
            sign_flip,
        ),
        "mobius": {
            ("+".join(arm) if arm else "baseline"): summary(
                float(
                    sum(
                        ((-1.0) ** (len(arm) - len(subset)))
                        * point_gain[
                            arm_index[tuple(
                                group for group in joint.EARLY_MLP_GROUPS
                                if group in subset
                            )]
                        ]
                        for subset in powerset(arm)
                    )
                ),
                mobius_draws[arm],
            )
            for arm in arms
        },
    }


def score_decisions(
    analyses: Mapping[str, Mapping[str, Any]],
    bootstraps: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    discovery = analyses["discovery"]
    heldout = analyses["heldout"]
    heldout_boot = bootstraps["heldout"]
    singleton_best = max(
        heldout["gain_by_arm"][group] for group in joint.EARLY_MLP_GROUPS
    )
    inputs = [
        heldout["joint_gain"],
        heldout["interaction_l1_fraction_of_joint_gain"],
        *(discovery["gain_by_arm"][group] for group in joint.EARLY_MLP_GROUPS),
        *(heldout["gain_by_arm"][group] for group in joint.EARLY_MLP_GROUPS),
        discovery["gain_by_arm"]["mlp2"],
        heldout["gain_by_arm"]["mlp2"],
        discovery["mlp2_conditional_marginal_after_mlp0_mlp1"],
        heldout["mlp2_conditional_marginal_after_mlp0_mlp1"],
        *(heldout_boot[name]["ci95"][bound]
          for name, bound in (
              ("joint_gain", 0),
              ("joint_minus_best_singleton", 0),
              ("mlp2_singleton_gain", 1),
              ("mlp2_conditional_marginal_after_mlp0_mlp1", 0),
              ("mlp2_sign_flip_contrast", 0),
              ("joint_minus_singleton_sum", 0),
          )),
    ]
    if not all(math.isfinite(float(value)) for value in inputs):
        raise ValueError("registered decision inputs must be finite")
    return {
        "pred_a_joint_positive_both_splits": (
            discovery["joint_gain"] > 0.0 and heldout["joint_gain"] > 0.0
        ),
        "pred_a_joint_heldout_ci95_lower_gt_zero": (
            heldout_boot["joint_gain"]["ci95"][0] > 0.0
        ),
        "pred_a_joint_exceeds_best_singleton_heldout": (
            heldout["joint_gain"] > singleton_best
        ),
        "pred_a_joint_minus_best_singleton_heldout_ci95_lower_gt_zero": (
            heldout_boot["joint_minus_best_singleton"]["ci95"][0] > 0.0
        ),
        "pred_b_mlp2_singleton_negative_both_splits": (
            discovery["gain_by_arm"]["mlp2"] < 0.0
            and heldout["gain_by_arm"]["mlp2"] < 0.0
        ),
        "pred_b_mlp2_conditional_after_mlp0_mlp1_positive_both_splits": (
            discovery["mlp2_conditional_marginal_after_mlp0_mlp1"] > 0.0
            and heldout["mlp2_conditional_marginal_after_mlp0_mlp1"] > 0.0
        ),
        "pred_b_mlp2_singleton_heldout_ci95_upper_lt_zero": (
            heldout_boot["mlp2_singleton_gain"]["ci95"][1] < 0.0
        ),
        "pred_b_mlp2_conditional_heldout_ci95_lower_gt_zero": (
            heldout_boot["mlp2_conditional_marginal_after_mlp0_mlp1"]["ci95"][0]
            > 0.0
        ),
        "pred_b_sign_flip_contrast_heldout_ci95_lower_gt_zero": (
            heldout_boot["mlp2_sign_flip_contrast"]["ci95"][0] > 0.0
        ),
        "pred_c_joint_minus_singleton_sum_positive_both_splits": (
            discovery["joint_minus_singleton_sum"] > 0.0
            and heldout["joint_minus_singleton_sum"] > 0.0
        ),
        "pred_c_joint_minus_singleton_sum_heldout_ci95_lower_gt_zero": (
            heldout_boot["joint_minus_singleton_sum"]["ci95"][0] > 0.0
        ),
    }


def mark_failed(error: BaseException, protected_after: Mapping[str, str | None]) -> None:
    try:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    except Exception:
        manifest = {}
    manifest.update({
        "schema_version": 1,
        "status": "failed_authoritative_factorial",
        "authority": "canonical_fineweb",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "training_license_sites": [],
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "protected_paths_after": dict(protected_after),
        "frozen_state_lifecycle": frozen_lifecycle_receipt(
            json.loads(row_prep.RECEIPT.read_text()) if row_prep.RECEIPT.is_file() else None
        ),
        "recovery": "Preserve all artifacts; diagnose and use a new versioned namespace for any retry.",
    })
    write_json_atomic(manifest, MANIFEST)
    if RESULT.exists():
        try:
            result = json.loads(RESULT.read_text())
        except Exception:
            result = {}
        result.update({
            "status": "failed_authoritative_factorial",
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "training_license_sites": [],
            "invalidated_by_failure": True,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        write_json_atomic(result, RESULT)


def run_claimed(protected_before: Mapping[str, str | None]) -> None:
    verify_pinned_inputs()
    source_hashes = verify_committed_source_closure()
    source_global_preflight.require_defined_globals([
        BQ / "ship_error_attrib.py",
        HERE / "joint_early_mlp_oracle_factorial.py",
        HERE / "frozen_ship_oracle_v2.py",
        HERE / "prepare_fineweb_oracle_rows.py",
        Path(__file__),
    ])
    row_receipt, frozen_rows = row_prep.validate_receipt()
    if row_receipt.get("authority") != "pinned_local_ordered_manifest":
        raise RuntimeError("FineWeb receipt lacks authoritative ordered-manifest status")
    if row_receipt.get("authorized_for_scored_experiments") is not True:
        raise RuntimeError("FineWeb receipt does not authorize scored experiments")
    if logical_json_sha256(row_receipt) != (
        "8c510aed5586b4d950f0688a0e575c7695e51525d69f80c6bc39817c1454e9cb"
    ):
        raise RuntimeError("FineWeb logical receipt hash changed")
    document_ids, split_receipts = validate_document_provenance(row_receipt, frozen_rows)
    code_rows, code_manifest = code_oracle.load_frozen_corpus()
    if FROZEN_STATE.exists() or FROZEN_MANIFEST.exists():
        raise RuntimeError("refusing to overwrite an existing frozen ship realization")
    if frozen.FROZEN_LOCK.exists():
        raise RuntimeError("canonical frozen-state lock already exists")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running_authoritative_factorial",
        "authority": "canonical_fineweb",
        "row_receipt_authorized_for_scored_experiments": True,
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "training_license_sites": [],
        "scope_guardrail": (
            "Exact-restoration oracle headroom and state-dependent causal nonadditivity "
            "on one fixed deployed ship and two pinned first-shard FineWeb blocks only; "
            "no corpus-wide, performance-ceiling, clean-model, transported-variable, "
            "predictor, compression, OOD, training, whole-model-fraction, or simplicity claim."
        ),
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": PREREG_SHA256,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=TENSOR_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_hashes": source_hashes,
        "pinned_input_hashes": {str(path): digest for path, digest in PINNED_INPUTS.items()},
        "fineweb_row_receipt_logical_sha256": logical_json_sha256(row_receipt),
        "row_splits": split_receipts,
        "code_sentinel_manifest_sha256": logical_json_sha256(code_manifest),
        "protected_paths_before": dict(protected_before),
    }
    write_json_atomic(manifest, MANIFEST)

    def registered_fineweb_rows(n: int = 120, skip: int = 0) -> torch.Tensor:
        spec = (n, skip)
        if spec not in frozen_rows:
            raise RuntimeError(f"unregistered FineWeb row request: {spec}")
        return frozen_rows[spec].clone()

    torch.manual_seed(SHIP_SEED)
    torch.cuda.manual_seed_all(SHIP_SEED)
    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    torch.manual_seed(SHIP_SEED)
    torch.cuda.manual_seed_all(SHIP_SEED)
    sa.cl.fineweb_rows = registered_fineweb_rows
    start_time = time.time()

    def factorial_callback(
        twall: dict, all_attention: frozenset[int], _: float
    ) -> None:
        realization_hash, frozen_manifest = frozen.freeze_ship_realization(
            sa, twall, all_attention, row_receipt, code_rows
        )
        component_before = component_tree_sha256(sa, twall, all_attention)
        if component_before != realization_hash:
            raise RuntimeError("fresh component tree differs from frozen realization")
        joint.clear_oracle_corrections(sa.ORACLE_CORR)
        sa.CONTENT_CORR["on"] = False
        require_inert_correction_state(sa)
        patch_canary = exact_patch_canary(sa)
        all_mlps = frozenset(range(18))
        discovery_rows = frozen_rows[(192, 7000)][:, :257].contiguous()
        heldout_rows = frozen_rows[(192, 11000)][:, :257].contiguous()
        rare_vocab = sa._token_masks(discovery_rows)
        evaluations: dict[str, Any] = {}
        analyses: dict[str, Any] = {}
        bootstraps: dict[str, Any] = {}
        partial: dict[str, Any] = {
            "schema_version": 1,
            "status": "running_authoritative_factorial",
            "authority": "canonical_fineweb",
            "row_receipt_authorized_for_scored_experiments": True,
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "training_license_sites": [],
            "preregistration_sha256": PREREG_SHA256,
            "ship_realization_sha256": realization_hash,
            "component_tree_before_sha256": component_before,
            "exact_patch_canary": patch_canary,
            "row_splits": split_receipts,
            "source_hashes": source_hashes,
            "evaluations": evaluations,
            "split_analyses": analyses,
            "paired_bootstraps": bootstraps,
        }
        write_json_atomic(partial, RESULT)
        try:
            for split_name, rows in (
                ("discovery", discovery_rows), ("heldout", heldout_rows)
            ):
                row_ce_by_arm: dict[tuple[str, ...], list[float]] = {}
                evaluations[split_name] = {}
                for arm in powerset(joint.EARLY_MLP_GROUPS):
                    joint.clear_oracle_corrections(sa.ORACLE_CORR)
                    require_inert_correction_state(sa)
                    correction_map = {
                        int(group[-1]): {"basis": None, "scale": 1.0}
                        for group in arm
                    }
                    joint.configure_oracle_corrections(sa.ORACLE_CORR, correction_map)
                    try:
                        scored = sa._score_content_rows(
                            rows, twall, all_attention, all_mlps,
                            rare_vocab=rare_vocab, retain_row_ce=True,
                        )
                    finally:
                        joint.clear_oracle_corrections(sa.ORACLE_CORR)
                    name = "+".join(arm) if arm else "baseline"
                    evaluations[split_name][name] = scored
                    row_ce_by_arm[arm] = scored["row_global_ce"]
                    write_json_atomic(partial, RESULT)
                    print(f"authoritative joint oracle {split_name} arm={name} done", flush=True)
                analyses[split_name] = joint.analyze_full_live_subset_rows(row_ce_by_arm)
                if abs(analyses[split_name]["shapley_closure_error"]) > 1e-10:
                    raise RuntimeError(f"Shapley closure failed on {split_name}")
                bootstraps[split_name] = paired_document_cluster_bootstrap(
                    row_ce_by_arm, document_ids[split_name]
                )
                write_json_atomic(partial, RESULT)
        finally:
            joint.clear_oracle_corrections(sa.ORACLE_CORR)

        require_inert_correction_state(sa)
        replay = sa._score_content_rows(
            heldout_rows, twall, all_attention, all_mlps,
            rare_vocab=rare_vocab, retain_row_ce=True,
        )
        first_baseline = evaluations["heldout"]["baseline"]["row_global_ce"]
        replay_difference = torch.tensor(replay["row_global_ce"], dtype=torch.float64) - torch.tensor(
            first_baseline, dtype=torch.float64
        )
        baseline_replay = {
            "max_abs_row_ce_difference": float(replay_difference.abs().max()),
            "mean_abs_row_ce_difference": float(replay_difference.abs().mean()),
            "max_tolerance": 1e-7,
            "mean_tolerance": 1e-8,
        }
        if (
            baseline_replay["max_abs_row_ce_difference"] > 1e-7
            or baseline_replay["mean_abs_row_ce_difference"] > 1e-8
        ):
            raise RuntimeError(f"heldout baseline replay changed: {baseline_replay}")
        component_after = component_tree_sha256(sa, twall, all_attention)
        if component_after != component_before:
            raise RuntimeError("component tree changed during factorial cube")
        decisions = score_decisions(analyses, bootstraps)
        result = {
            "schema_version": 1,
            "status": "scored_pending_integrity",
            "authority": "canonical_fineweb",
            "row_receipt_authorized_for_scored_experiments": True,
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "training_license_sites": [],
            "interpretation_guardrail": manifest["scope_guardrail"],
            "config": {
                "groups": list(joint.EARLY_MLP_GROUPS),
                "arms": ["+".join(arm) if arm else "baseline"
                         for arm in powerset(joint.EARLY_MLP_GROUPS)],
                "row_counts": {"discovery": 192, "heldout": 192},
                "row_requests": {"discovery": [192, 7000], "heldout": [192, 11000]},
                "rare_vocab": "derived once on discovery and frozen across splits/arms",
                "intervention": (
                    "exact live original-minus-deployed-approximation residual at every "
                    "selected site on that arm's current state; MLP2 approximation includes "
                    "the ridge plank and live rank-32 glue"
                ),
                "same_currency_residual_denominator": None,
                "denominator_rule": (
                    "No cross-run MLP0-2 denominator imported; recovery fraction is null."
                ),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "FineWeb source document cluster",
            },
            "ship_realization_sha256": realization_hash,
            "component_tree_before_sha256": component_before,
            "component_tree_after_sha256": component_after,
            "component_tree_unchanged": True,
            "exact_patch_canary": patch_canary,
            "heldout_baseline_replay": baseline_replay,
            "frozen_ship_artifact_sha256": frozen_manifest["artifact_sha256"],
            "frozen_ship_manifest_sha256": file_sha256(FROZEN_MANIFEST),
            "fineweb_row_receipt_file_sha256": file_sha256(row_prep.RECEIPT),
            "fineweb_row_receipt_logical_sha256": logical_json_sha256(row_receipt),
            "row_splits": split_receipts,
            "source_commit": manifest["source_commit"],
            "source_hashes": source_hashes,
            "evaluations": evaluations,
            "split_analyses": analyses,
            "paired_bootstraps": bootstraps,
            "registered_predictions": decisions,
            "runtime_s": round(time.time() - start_time, 1),
            "preregistration_sha256": PREREG_SHA256,
        }
        write_json_atomic(result, RESULT)
        manifest.update({
            "status": "scored_pending_integrity",
            "ship_realization_sha256": realization_hash,
            "frozen_ship_path": str(FROZEN_STATE),
            "frozen_ship_artifact_sha256": frozen_manifest["artifact_sha256"],
            "frozen_ship_manifest_path": str(FROZEN_MANIFEST),
            "frozen_ship_manifest_sha256": file_sha256(FROZEN_MANIFEST),
            "result_path": str(RESULT.resolve()),
            "result_sha256": file_sha256(RESULT),
            "registered_predictions": decisions,
            "component_tree_before_sha256": component_before,
            "component_tree_after_sha256": component_after,
            "component_tree_unchanged": True,
            "exact_patch_canary": patch_canary,
            "heldout_baseline_replay": baseline_replay,
            "frozen_state_lifecycle": frozen_lifecycle_receipt(row_receipt),
            "runtime_s": result["runtime_s"],
        })
        write_json_atomic(manifest, MANIFEST)
        print(json.dumps({
            "split_analyses": analyses,
            "paired_bootstraps": bootstraps,
            "registered_predictions": decisions,
            "training_license_sites": [],
        }, indent=2), flush=True)

    sa.run_oracle_content_screen = factorial_callback
    sa.main(oracle_content_screen=True)


def finalize_success(protected_after: Mapping[str, str | None]) -> None:
    result = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if result.get("status") != "scored_pending_integrity":
        raise RuntimeError("result is not ready for integrity finalization")
    if manifest.get("status") != "scored_pending_integrity":
        raise RuntimeError("manifest is not ready for integrity finalization")
    if manifest.get("result_sha256") != file_sha256(RESULT):
        raise RuntimeError("pending result hash differs from manifest")
    lifecycle = frozen_lifecycle_receipt(
        json.loads(row_prep.RECEIPT.read_text())
    )
    if not lifecycle.get("validated"):
        raise RuntimeError(f"frozen state does not validate at finalization: {lifecycle}")
    result.update({
        "status": "completed_authoritative_factorial",
        "authorized_for_scored_experiments": True,
        "protected_paths_after": dict(protected_after),
        "protected_paths_unchanged": True,
        "frozen_state_lifecycle": lifecycle,
        "integrity_finalized": True,
    })
    write_json_atomic(result, RESULT)
    manifest.update({
        "status": "completed_authoritative_factorial",
        "authorized_for_scored_experiments": True,
        "protected_paths_after": dict(protected_after),
        "protected_paths_unchanged": True,
        "frozen_state_lifecycle": lifecycle,
        "result_sha256": file_sha256(RESULT),
        "integrity_finalized": True,
    })
    write_json_atomic(manifest, MANIFEST)


def main() -> None:
    if RESULT.exists() or MANIFEST.exists():
        raise RuntimeError("refusing to overwrite authoritative factorial artifacts")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"authoritative factorial launch already claimed: {LOCK}") from error
    protected_before = protected_snapshot()
    run_error: BaseException | None = None
    try:
        run_claimed(protected_before)
    except BaseException as error:
        run_error = error
    protected_after = protected_snapshot()
    try:
        if protected_after != protected_before:
            contamination = RuntimeError(
                "authoritative factorial changed protected prior artifacts"
            )
            mark_failed(contamination, protected_after)
            raise contamination from run_error
        if run_error is not None:
            mark_failed(run_error, protected_after)
            raise run_error
        finalize_success(protected_after)
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
