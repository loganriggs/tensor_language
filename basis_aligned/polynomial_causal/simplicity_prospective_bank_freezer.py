#!/usr/bin/env python3
"""Rung445: freeze a rebuildable, whole-family prospective consequence bank."""

from __future__ import annotations

import hashlib
import json
import py_compile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
HERE = Path(__file__).resolve().parent
FEATURES = HERE / "simplicity_candidate_arm_features_v1.json"
BANK = HERE / "prospective_consequence_candidate_bank_v1.json"
RESULT = HERE / "prospective_consequence_candidate_bank_freezer_results.json"
FEATURE_SHA256 = "6d3e984f0ab5b343fabcbaae5b25a8b5b60284b1dc4b46ad6848d8cb72654738"

TEACHING_FAMILIES = {
    "vocabulary_factorization",
    "mixed104_mlp_pca",
    "mixed104_mlp0_context_input",
}
CONFIRMATION_FAMILIES = {"attention0_sparse_qk"}
EXPECTED_COUNTS = {
    "vocabulary_factorization": 23,
    "mixed104_mlp_pca": 7,
    "mixed104_mlp0_context_input": 5,
    "attention0_sparse_qk": 10,
}
FORBIDDEN_PARTS = {
    "ce", "loss", "damage", "error", "fidelity", "accuracy", "cosine",
    "correlation", "certificate", "outcome", "pred", "null",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def forbidden_keys(value: Any) -> list[str]:
    bad: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            parts = set(str(key).lower().replace("-", "_").split("_"))
            if parts & FORBIDDEN_PARTS:
                bad.add(str(key))
            bad.update(forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            bad.update(forbidden_keys(item))
    return sorted(bad)


def producer(candidate_id: str) -> tuple[str, str, list[str]]:
    """Return source, rebuild mode, and already-existing required artifacts."""
    if candidate_id.startswith("vocab_r300_"):
        return (
            "ops/joint_vocab_shared_code_screen.py", "deterministic_refit",
            [".rowcache/fineweb_n96_skip1200.pt", ".rowcache/fineweb_n96_skip80.pt"],
        )
    if candidate_id.startswith("vocab_r304_"):
        return (
            "ops/joint_vocab_sparse_rare_residual.py", "deterministic_refit",
            [".rowcache/fineweb_n480_skip80.pt", ".rowcache/fineweb_n192_skip7000.pt"],
        )
    if candidate_id.startswith("vocab_r305_"):
        return (
            "ops/joint_vocab_distributed_rank_frontier.py", "deterministic_refit",
            [".rowcache/fineweb_n480_skip80.pt", ".rowcache/fineweb_n192_skip11000.pt"],
        )
    if candidate_id.startswith("mlp_pca_grad"):
        return (
            "ops/mixed104_pca_certificate_gradient_hybrid.py", "deterministic_refit",
            [".rowcache/fineweb_n480_skip80.pt", "census_state_diverse.pt",
             "mixed104_online_cv0_results.json"],
        )
    if candidate_id in {"mlp_pca_p8_17_r384", "mlp_pca_p8_17_r512"}:
        return (
            "ops/mixed104_pca_fixed_pair_rank_frontier.py", "deterministic_refit",
            [".rowcache/fineweb_n480_skip80.pt", "census_state_diverse.pt"],
        )
    if candidate_id.startswith("mlp_pca_"):
        return (
            "ops/mixed104_pca_fixed_pair_frontier.py", "deterministic_refit",
            [".rowcache/fineweb_n480_skip80.pt", "census_state_diverse.pt"],
        )
    if candidate_id in {
        "mlp0_context_input_r256", "mlp0_context_input_r384", "mlp0_context_input_r448",
    }:
        return (
            "ops/mixed104_mlp0_context_metric_lower_rank_frontier.py", "deterministic_refit",
            [".rowcache/fineweb_n192_skip11000.pt", "census_state_diverse.pt"],
        )
    if candidate_id in {"mlp0_context_input_r512", "mlp0_context_input_r640"}:
        return (
            "ops/mixed104_mlp0_context_metric_input_frontier.py", "deterministic_refit",
            [".rowcache/fineweb_n192_skip11000.pt", "census_state_diverse.pt",
             "mlp0_context_metric_shared_input_frontier_results.json"],
        )
    if candidate_id.startswith("attention0_r426_"):
        return (
            "ops/attention0_cross_head_sparse_qk_vocabulary.py", "retained_bundle_and_deterministic_arm",
            ["attention0_cross_head_sparse_qk_vocabulary_bundle.pt"],
        )
    if candidate_id.startswith("attention0_r430_"):
        return (
            "ops/attention0_coupled_sparse_qk_score_product.py", "retained_bundle_and_deterministic_arm",
            ["attention0_coupled_sparse_qk_score_product_bundle.pt"],
        )
    raise KeyError(f"no frozen producer mapping for {candidate_id}")


def main() -> None:
    if sha256(FEATURES) != FEATURE_SHA256:
        raise RuntimeError("rung441 structural feature hash mismatch")
    payload = json.loads(FEATURES.read_text())
    rows = payload["rows"]
    ids = [str(row["candidate_id"]) for row in rows]
    if len(ids) != 45 or len(set(ids)) != 45:
        raise RuntimeError("expected45 unique rung441 candidate IDs")

    frozen_rows: list[dict[str, Any]] = []
    compile_status: dict[str, bool] = {}
    artifact_hashes: dict[str, str] = {}
    missing: list[str] = []
    for row in rows:
        family = str(row["program_family"])
        role = "teaching" if family in TEACHING_FAMILIES else "sealed_confirmation"
        if family not in TEACHING_FAMILIES | CONFIRMATION_FAMILIES:
            raise RuntimeError(f"unregistered family {family}")
        source_name, mode, artifacts = producer(str(row["candidate_id"]))
        source_path = BQ / source_name
        try:
            py_compile.compile(str(source_path), doraise=True)
            compile_status[source_name] = True
        except (OSError, py_compile.PyCompileError):
            compile_status[source_name] = False
            missing.append(source_name)
        required = []
        for artifact_name in artifacts:
            artifact_path = BQ / artifact_name
            if not artifact_path.exists():
                missing.append(artifact_name)
                continue
            artifact_hashes.setdefault(artifact_name, sha256(artifact_path))
            required.append({"path": artifact_name, "sha256": artifact_hashes[artifact_name]})
        copied = dict(row)
        copied.update({
            "family_role": role,
            "producer_source": source_name,
            "producer_source_sha256": sha256(source_path) if source_path.exists() else None,
            "rebuild_mode": mode,
            "required_artifacts": required,
        })
        frozen_rows.append(copied)

    bank = {
        "schema": "prospective_consequence_candidate_bank_v1",
        "source_feature_sha256": FEATURE_SHA256,
        "role_policy": {
            "teaching_families": sorted(TEACHING_FAMILIES),
            "sealed_confirmation_families": sorted(CONFIRMATION_FAMILIES),
        },
        "rows": frozen_rows,
    }
    bad = forbidden_keys(bank)
    write_json(BANK, bank)

    counts = Counter(str(row["program_family"]) for row in frozen_rows)
    teaching = [row for row in frozen_rows if row["family_role"] == "teaching"]
    sealed = [row for row in frozen_rows if row["family_role"] == "sealed_confirmation"]
    teaching_families = {str(row["program_family"]) for row in teaching}
    sealed_families = {str(row["program_family"]) for row in sealed}
    price_positive = all(
        (row.get("price_scalars") is not None and float(row["price_scalars"]) > 0)
        or (row.get("price_bytes") is not None and float(row["price_bytes"]) > 0)
        for row in frozen_rows
    )
    mapped = sum(
        bool(row.get("producer_source_sha256")) and bool(row["required_artifacts"])
        for row in frozen_rows
    )
    controls = sum(str(row["control_type"]) != "candidate" for row in sealed)
    noncontrols = len(sealed) - controls
    pred_a = bool(
        not bad and counts == Counter(EXPECTED_COUNTS) and len(frozen_rows) == 45
    )
    pred_b = bool(
        mapped == 45 and not missing and all(compile_status.values())
        and all(row["rebuild_mode"] in {"deterministic_refit", "retained_bundle_and_deterministic_arm"}
                for row in frozen_rows)
    )
    pred_c = bool(
        len(teaching) >= 30 and teaching_families == TEACHING_FAMILIES
        and min(Counter(str(row["program_family"]) for row in teaching).values()) >= 5
        and price_positive
    )
    pred_d = bool(
        len(sealed) >= 8 and sealed_families == CONFIRMATION_FAMILIES
        and not (teaching_families & sealed_families) and controls >= 2 and noncontrols >= 5
    )
    strong_null = bool(
        bad or (teaching_families & sealed_families) or mapped < 38
        or len(teaching) < 30 or len(sealed) < 8 or bool(missing)
    )
    result = {
        "status": "complete",
        "rung": 445,
        "claim_level": "prospective_dataset_instrument",
        "source_feature_sha256": FEATURE_SHA256,
        "bank_path": str(BANK),
        "bank_sha256": sha256(BANK),
        "candidate_count": len(frozen_rows),
        "family_counts": dict(sorted(counts.items())),
        "teaching_candidate_count": len(teaching),
        "teaching_family_count": len(teaching_families),
        "sealed_candidate_count": len(sealed),
        "sealed_family_count": len(sealed_families),
        "sealed_control_count": controls,
        "sealed_noncontrol_count": noncontrols,
        "mapped_rebuild_count": mapped,
        "unique_producer_source_count": len(compile_status),
        "unique_required_artifact_count": len(artifact_hashes),
        "compile_status": compile_status,
        "missing": sorted(set(missing)),
        "forbidden_structural_keys": bad,
        "consequence_file_loaded": False,
        "model_or_row_role_loaded": False,
        "pred_a_separation_and_identity": pred_a,
        "pred_b_rebuild_mapping": pred_b,
        "pred_c_teaching_support": pred_c,
        "pred_d_sealed_support": pred_d,
        "strong_null_bank_not_viable": strong_null,
        "next_step": (
            "preregister_teaching_family_removal_and_composition_generation"
            if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "repair_bank_under_new_registration_before_any_consequence_generation"
        ),
        "literal_deployed_model_values": 0,
        "native_model_calls": 0,
    }
    write_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

