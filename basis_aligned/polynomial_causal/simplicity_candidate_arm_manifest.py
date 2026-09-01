#!/usr/bin/env python3
"""Rung 441: hand-reviewed one-row-per-candidate archive manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
HERE = Path(__file__).resolve().parent
FEATURE_PATH = HERE / "simplicity_candidate_arm_features_v1.json"
LABEL_PATH = HERE / "simplicity_candidate_arm_consequences_v1.json"
RESULT_PATH = HERE / "simplicity_candidate_arm_manifest_results.json"


def load(name: str) -> dict[str, Any]:
    return json.loads((BQ / name).read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(*names: str) -> list[dict[str, str]]:
    return [
        {"receipt": name, "sha256": digest(BQ / name)}
        for name in names
    ]


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def feature_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Family 1: vocabulary programs. The mapping explicitly expands shared and matched-independent arms.
    r300_name = "joint_vocab_shared_code_screen_results.json"
    r300 = load(r300_name)
    for arm_name, arm in r300["arms"].items():
        for kind in ("shared", "independent"):
            rank_key = "shared_residual_rank" if kind == "shared" else "independent_matched_rank"
            price_key = "shared_price_scalars" if kind == "shared" else "independent_price_scalars"
            rows.append(
                {
                    "candidate_id": f"vocab_r300_{kind}_{arm_name}",
                    "program_family": "vocabulary_factorization",
                    "grammar_family": "shared_low_rank" if kind == "shared" else "independent_low_rank",
                    "control_type": "candidate" if kind == "shared" else "price_matched_baseline",
                    "sources": source_record(r300_name),
                    "price_scalars": arm[price_key],
                    "price_bytes": None,
                    "rank": arm[rank_key],
                    "atom_count": None,
                    "sparse_rows": 0,
                    "shares_parameters_across_input_output": kind == "shared",
                }
            )

    r304_name = "joint_vocab_sparse_rare_residual_results.json"
    r304 = load(r304_name)
    for arm_name in r304["arms"]:
        rows.append(
            {
                "candidate_id": f"vocab_r304_{arm_name}",
                "program_family": "vocabulary_factorization",
                "grammar_family": "shared_low_rank_plus_indexed_rows",
                "control_type": "candidate" if arm_name == "fisher_rare" else "selector_control",
                "sources": source_record(r304_name),
                "price_scalars": r304["price"]["hybrid_scalars"],
                "price_bytes": None,
                "rank": 512,
                "atom_count": None,
                "sparse_rows": r304["price"]["selected_rows"],
                "shares_parameters_across_input_output": True,
            }
        )

    r305_name = "joint_vocab_distributed_rank_frontier_results.json"
    r305 = load(r305_name)
    for arm_name, arm in r305["arms"].items():
        for kind in ("shared", "independent"):
            rows.append(
                {
                    "candidate_id": f"vocab_r305_{arm_name}_{kind}",
                    "program_family": "vocabulary_factorization",
                    "grammar_family": f"{kind}_weighted_low_rank",
                    "control_type": "candidate" if kind == "shared" else "price_matched_baseline",
                    "sources": source_record(r305_name),
                    "price_scalars": arm[f"{kind}_scalars"],
                    "price_bytes": None,
                    "rank": arm[f"{kind}_rank"],
                    "atom_count": None,
                    "sparse_rows": 0,
                    "shares_parameters_across_input_output": kind == "shared",
                }
            )

    # Family 2: whole-program MLP PCA. The p8+17 rank256 candidate is represented once.
    r311_name = "mixed104_pca_fixed_pair_frontier_results.json"
    r311 = load(r311_name)
    for arm_name in r311["arms"]:
        rows.append(
            {
                "candidate_id": f"mlp_pca_{arm_name}_r256",
                "program_family": "mixed104_mlp_pca",
                "grammar_family": "activation_pca_pair",
                "control_type": "candidate",
                "sources": source_record(r311_name),
                "price_scalars": r311["price"]["literal_standalone_scalars_each"],
                "price_bytes": r311["price"]["literal_raw_tensor_bytes_each"],
                "rank": 256,
                "atom_count": None,
                "sparse_rows": 0,
                "shares_parameters_across_input_output": False,
            }
        )
    r312_name = "mixed104_pca_fixed_pair_rank_frontier_results.json"
    r312 = load(r312_name)
    for arm_name in ("384", "512"):
        arm = r312["arms"][arm_name]
        rows.append(
            {
                "candidate_id": f"mlp_pca_p8_17_r{arm_name}",
                "program_family": "mixed104_mlp_pca",
                "grammar_family": "activation_pca_pair",
                "control_type": "candidate",
                "sources": source_record(r312_name),
                "price_scalars": arm["literal_standalone_scalars"],
                "price_bytes": arm["literal_raw_tensor_bytes"],
                "rank": arm["rank"],
                "atom_count": None,
                "sparse_rows": 0,
                "shares_parameters_across_input_output": False,
            }
        )
    r313_name = "mixed104_pca_certificate_gradient_hybrid_results.json"
    r313 = load(r313_name)
    for arm_name in ("grad32", "grad64"):
        rows.append(
            {
                "candidate_id": f"mlp_pca_{arm_name}",
                "program_family": "mixed104_mlp_pca",
                "grammar_family": "gradient_plus_activation_pca",
                "control_type": "candidate",
                "sources": source_record(r313_name),
                "price_scalars": r313["price_each"]["literal_standalone_scalars"],
                "price_bytes": r313["price_each"]["literal_raw_tensor_bytes"],
                "rank": int(arm_name.removeprefix("grad")),
                "atom_count": None,
                "sparse_rows": 0,
                "shares_parameters_across_input_output": False,
            }
        )

    # Family 3: MLP0 context-metric input ranks, merging primary/OOD/intervention follow-ups by rank.
    high_name = "mixed104_mlp0_context_metric_input_frontier_results.json"
    low_name = "mixed104_mlp0_context_metric_lower_rank_frontier_results.json"
    high, low = load(high_name), load(low_name)
    for arm_name, arm in {**low["arms"], **high["arms"]}.items():
        source_names = [high_name] if arm_name in high["arms"] else [low_name]
        rows.append(
            {
                "candidate_id": f"mlp0_context_input_r{arm_name}",
                "program_family": "mixed104_mlp0_context_input",
                "grammar_family": "context_metric_shared_input_low_rank",
                "control_type": "candidate",
                "sources": source_record(*source_names),
                "price_scalars": arm["literal_standalone_scalars"],
                "price_bytes": arm["literal_raw_tensor_bytes"],
                "rank": arm["rank"],
                "atom_count": None,
                "sparse_rows": 0,
                "shares_parameters_across_input_output": True,
            }
        )

    # Family 4: attention0 sparse Q/K generator programs and controls.
    r426_name = "attention0_cross_head_sparse_qk_vocabulary_results.json"
    r426 = load(r426_name)
    for arm_name in ("I72", "G54", "G72", "D54"):
        bill_key = arm_name if arm_name in r426["literal_raw_tensor_bytes"] else "G54"
        rows.append(
            {
                "candidate_id": f"attention0_r426_{arm_name}",
                "program_family": "attention0_sparse_qk",
                "grammar_family": "sparse_qk_dictionary",
                "control_type": "deranged_control" if arm_name == "D54" else "candidate",
                "sources": source_record(r426_name),
                "price_scalars": None,
                "price_bytes": r426["literal_raw_tensor_bytes"][bill_key],
                "rank": None,
                "atom_count": 512,
                "sparse_rows": 72 if arm_name.endswith("72") else 54,
                "shares_parameters_across_input_output": arm_name.startswith("G"),
            }
        )
    r430_name = "attention0_coupled_sparse_qk_score_product_results.json"
    r430 = load(r430_name)
    for arm_name in ("SQ54", "SC54", "CP54", "CP72", "PP54", "WH54"):
        rows.append(
            {
                "candidate_id": f"attention0_r430_{arm_name}",
                "program_family": "attention0_sparse_qk",
                "grammar_family": "coupled_sparse_qk_dictionary",
                "control_type": "relation_control" if arm_name in ("PP54", "WH54") else "candidate",
                "sources": source_record(r430_name),
                "price_scalars": None,
                "price_bytes": r430["literal_raw_tensor_bytes"]["CP72" if arm_name == "CP72" else "CP54"],
                "rank": None,
                "atom_count": 512,
                "sparse_rows": 72 if arm_name.endswith("72") else 54,
                "shares_parameters_across_input_output": True,
            }
        )
    return rows


def blank_label(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "consequence_sources": [],
        "validation_ce_damage": None,
        "ood_ce_damage": None,
        "census_ce_damage": None,
        "certificates_valid": None,
        "intervention_effect_cosine": None,
        "intervention_normalized_error": None,
        "intervention_collateral_spearman": None,
        "composition_ratio": None,
        "composition_prediction_error": None,
        "local_relative_squared_error": None,
        "attention_write_relative_squared_error": None,
        "downstream_reader_relative_squared_error": None,
    }


def consequence_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {row["candidate_id"]: blank_label(row["candidate_id"]) for row in features}
    r300 = load("joint_vocab_shared_code_screen_results.json")
    for arm_name, arm in r300["arms"].items():
        for kind in ("shared", "independent"):
            row = labels[f"vocab_r300_{kind}_{arm_name}"]
            row["consequence_sources"] = source_record("joint_vocab_shared_code_screen_results.json")
            row["validation_ce_damage"] = arm[f"{kind}_fineweb"]["damage"]
            row["ood_ce_damage"] = arm[f"{kind}_wikitext"]["damage"]
            row["local_relative_squared_error"] = arm[f"{kind}_output_weight_relative_mse"]
    r304 = load("joint_vocab_sparse_rare_residual_results.json")
    for arm_name, arm in r304["arms"].items():
        row = labels[f"vocab_r304_{arm_name}"]
        row["consequence_sources"] = source_record("joint_vocab_sparse_rare_residual_results.json")
        row["validation_ce_damage"] = arm["fineweb"]["damage"]
        row["ood_ce_damage"] = arm["wikitext"]["damage"]
    r305 = load("joint_vocab_distributed_rank_frontier_results.json")
    for arm_name, arm in r305["arms"].items():
        for kind in ("shared", "independent"):
            row = labels[f"vocab_r305_{arm_name}_{kind}"]
            row["consequence_sources"] = source_record("joint_vocab_distributed_rank_frontier_results.json")
            row["validation_ce_damage"] = arm[f"{kind}_fineweb"]["damage"]
            row["ood_ce_damage"] = arm[f"{kind}_wikitext"]["damage"]

    r311 = load("mixed104_pca_fixed_pair_frontier_results.json")
    for arm_name, arm in r311["arms"].items():
        row = labels[f"mlp_pca_{arm_name}_r256"]
        row["consequence_sources"] = source_record("mixed104_pca_fixed_pair_frontier_results.json")
        row["census_ce_damage"] = arm["census_damage"]
        row["certificates_valid"] = arm["certificates_valid"]
    r312 = load("mixed104_pca_fixed_pair_rank_frontier_results.json")
    for arm_name in ("384", "512"):
        arm = r312["arms"][arm_name]
        row = labels[f"mlp_pca_p8_17_r{arm_name}"]
        row["consequence_sources"] = source_record("mixed104_pca_fixed_pair_rank_frontier_results.json")
        row["census_ce_damage"] = arm["census_damage"]
        row["certificates_valid"] = arm["certificates_valid"]
    r313 = load("mixed104_pca_certificate_gradient_hybrid_results.json")
    for arm_name in ("grad32", "grad64"):
        arm = r313["arms"][arm_name]
        row = labels[f"mlp_pca_{arm_name}"]
        row["consequence_sources"] = source_record("mixed104_pca_certificate_gradient_hybrid_results.json")
        row["census_ce_damage"] = arm["census_damage"]
        row["certificates_valid"] = arm["row_excluded_full_certificates_valid"]

    high = load("mixed104_mlp0_context_metric_input_frontier_results.json")
    low = load("mixed104_mlp0_context_metric_lower_rank_frontier_results.json")
    high_ood = load("mixed104_mlp0_context_metric_input_frontier_ood_results.json")
    low_ood = load("mixed104_mlp0_context_metric_lower_rank_ood_results.json")
    intervention = load("a16_transfer_mixed104_mlp0_context_metric_frontier_results.json")
    for arm_name, arm in {**low["arms"], **high["arms"]}.items():
        row = labels[f"mlp0_context_input_r{arm_name}"]
        row["census_ce_damage"] = arm["census_damage"]
        row["certificates_valid"] = arm["certificates_valid"]
        ood = (high_ood if arm_name in high_ood["arms"] else low_ood)["arms"][arm_name]
        source_names = [
            "mixed104_mlp0_context_metric_input_frontier_results.json" if arm_name in high["arms"] else "mixed104_mlp0_context_metric_lower_rank_frontier_results.json",
            "mixed104_mlp0_context_metric_input_frontier_ood_results.json" if arm_name in high_ood["arms"] else "mixed104_mlp0_context_metric_lower_rank_ood_results.json",
        ]
        if arm_name in intervention["arms"]:
            source_names.append("a16_transfer_mixed104_mlp0_context_metric_frontier_results.json")
        row["consequence_sources"] = source_record(*source_names)
        row["validation_ce_damage"] = arm["census_damage"]
        row["ood_ce_damage"] = ood["shifted_damage_mean"]
        if arm_name in intervention["arms"]:
            ir = intervention["arms"][arm_name]
            row["intervention_effect_cosine"] = ir["effect_cosine"]
            row["intervention_normalized_error"] = ir["effect_normalized_error"]
            row["intervention_collateral_spearman"] = ir["collateral_spearman"]

    for receipt_name, prefix, arms in (
        ("attention0_cross_head_sparse_qk_vocabulary_results.json", "attention0_r426_", ("I72", "G54", "G72", "D54")),
        ("attention0_coupled_sparse_qk_score_product_results.json", "attention0_r430_", ("SQ54", "SC54", "CP54", "CP72", "PP54", "WH54")),
    ):
        value = load(receipt_name)
        for arm_name in arms:
            row = labels[prefix + arm_name]
            row["consequence_sources"] = source_record(receipt_name)
            row["validation_ce_damage"] = value["select_document_metrics"]["ce"][arm_name]["damage"]
            row["local_relative_squared_error"] = value["select_random_pair_score_metrics"][arm_name]["complete_pattern_relative_squared_error"]
            row["attention_write_relative_squared_error"] = value["select_document_metrics"]["full_attention0_write_relative_squared_error"][arm_name]
            row["downstream_reader_relative_squared_error"] = value["select_document_metrics"]["mean_consumer_relative_squared_error"][arm_name]
    return [labels[key] for key in sorted(labels)]


FORBIDDEN_PARTS = {"ce", "damage", "error", "fidelity", "accuracy", "cosine", "correlation", "certificate", "outcome", "pred", "null"}


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


def command_features() -> None:
    rows = feature_rows()
    output = {"schema": "simplicity_candidate_arm_features_v1", "rows": rows}
    bad = forbidden_keys(output)
    if bad:
        raise RuntimeError(f"forbidden structural keys: {bad}")
    write(FEATURE_PATH, output)
    print(json.dumps({"feature_sha256": digest(FEATURE_PATH), "candidate_count": len(rows)}))


def command_labels(feature_sha: str) -> None:
    if digest(FEATURE_PATH) != feature_sha:
        raise RuntimeError("feature hash mismatch")
    features = json.loads(FEATURE_PATH.read_text())["rows"]
    labels = consequence_rows(features)
    write(LABEL_PATH, {"schema": "simplicity_candidate_arm_consequences_v1", "frozen_feature_sha256": feature_sha, "rows": labels})
    print(json.dumps({"consequence_sha256": digest(LABEL_PATH), "candidate_count": len(labels)}))


def command_audit() -> None:
    feature_doc = json.loads(FEATURE_PATH.read_text())
    label_doc = json.loads(LABEL_PATH.read_text())
    features, labels = feature_doc["rows"], label_doc["rows"]
    fids = [row["candidate_id"] for row in features]
    lids = [row["candidate_id"] for row in labels]
    f_by_id = {row["candidate_id"]: row for row in features}
    families = sorted({row["program_family"] for row in features})
    price_complete = all((row["price_scalars"] or 0) > 0 or (row["price_bytes"] or 0) > 0 for row in features)
    sources_complete = all(row["sources"] and all(s["sha256"] for s in row["sources"]) for row in features)
    consequence_sources_complete = all(row["consequence_sources"] and all(s["sha256"] for s in row["consequence_sources"]) for row in labels)
    coverage: dict[str, dict[str, Any]] = {}
    fields = {
        "ood": "ood_ce_damage",
        "extraction": "certificates_valid",
        "removal": "intervention_effect_cosine",
        "composition": "composition_ratio",
    }
    for name, field in fields.items():
        present = [row for row in labels if row[field] is not None]
        present_families = sorted({f_by_id[row["candidate_id"]]["program_family"] for row in present})
        coverage[name] = {
            "candidates": len(present),
            "families": present_families,
            "family_count": len(present_families),
            "historical_fit_licensed": len(present) >= 20 and len(present_families) >= 3,
        }
    pred_a = len(features) >= 40 and len(families) == 4 and len(fids) == len(set(fids)) and price_complete and sources_complete and consequence_sources_complete and not forbidden_keys(feature_doc)
    pred_b = coverage["ood"]["candidates"] >= 25 and coverage["ood"]["family_count"] >= 2
    pred_c = coverage["extraction"]["candidates"] >= 10 and coverage["extraction"]["family_count"] >= 2
    pred_d = (
        coverage["removal"]["candidates"] >= 10
        and coverage["removal"]["family_count"] >= 2
        and coverage["composition"]["candidates"] >= 10
        and coverage["composition"]["family_count"] >= 2
    )
    strong_null = len(features) < 30 or not any(v["candidates"] >= 20 and v["family_count"] >= 2 for v in coverage.values())
    licensed = [name for name, value in coverage.items() if value["historical_fit_licensed"]]
    result = {
        "status": "complete",
        "rung": 441,
        "claim_level": "dataset_instrument",
        "candidate_count": len(features),
        "program_families": families,
        "duplicate_candidate_ids": len(fids) - len(set(fids)),
        "join_keys_match": sorted(fids) == sorted(lids),
        "price_complete": price_complete,
        "source_hashes_complete": sources_complete,
        "consequence_source_hashes_complete": consequence_sources_complete,
        "forbidden_structural_keys": forbidden_keys(feature_doc),
        "feature_sha256": digest(FEATURE_PATH),
        "consequence_sha256": digest(LABEL_PATH),
        "coverage": coverage,
        "historical_fit_licensed_consequences": licensed,
        "pred_a_unique_priced_arm_manifest": pred_a,
        "pred_b_ood_coverage": pred_b,
        "pred_c_extraction_coverage": pred_c,
        "pred_d_removal_and_composition_coverage": pred_d,
        "strong_null_no_salvageable_historical_slice": strong_null,
        "routing": "do_not_fit_generate_prospective_removal_and_composition_families" if not licensed else "fit_only_licensed_consequences",
        "literal_deployed_model_values": 0,
        "native_model_calls": 0,
    }
    write(RESULT_PATH, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features")
    label_parser = sub.add_parser("labels")
    label_parser.add_argument("--feature-sha", required=True)
    sub.add_parser("audit")
    args = parser.parse_args()
    if args.command == "features":
        command_features()
    elif args.command == "labels":
        command_labels(args.feature_sha)
    else:
        command_audit()


if __name__ == "__main__":
    main()
