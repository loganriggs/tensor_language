#!/usr/bin/env python3
"""Assemble a denominator-safe north-star balance sheet for bilin18.

This is deliberately an aggregator, not a new evaluator.  It makes the existing
results comparable by recording their scope and currency while refusing to average
quantities which answer different questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TENSOR_ROOT = HERE.parents[1]

DEFAULT_SOURCES = {
    "causal_coverage": TENSOR_ROOT / "basis_aligned/qk_mdl/qk_coverage_ledger.json",
    "analytic_substitution": TENSOR_ROOT / "basis_aligned/qk_mdl/qk_wholemodel_substitutable.json",
    "named_understanding": TENSOR_ROOT / "basis_aligned/bilinear_quotient/whole_model_understanding_multidraw_results.json",
    "composed_replacement": TENSOR_ROOT / "basis_aligned/bilinear_quotient/whole_model_greedy_results.json",
    "content_passthrough": TENSOR_ROOT / "basis_aligned/bilinear_quotient/whole_model_content_passthrough_results.json",
    "upstream_interface": TENSOR_ROOT / "basis_aligned/bilinear_quotient/whole_model_upstream_results.json",
    "question_channel": HERE / "question_channel_ledger_results.json",
    "output_slice": HERE / "output_slice_audit_results.json",
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, atol: float = 5e-4) -> bool:
    return abs(a - b) <= atol


def registry_inventory(theseus_root: Path | None) -> dict[str, Any]:
    if theseus_root is None:
        return {"available": False, "reason": "no TheseusBench root supplied"}
    reg = theseus_root / "registry"
    paths = {
        "circuits": reg / "circuits.json",
        "frontier": reg / "frontier_seed.json",
        "fidelity": reg / "fidelity_seed.json",
        "anchor_meta": theseus_root / "bench/anchors/_meta.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return {"available": False, "missing": missing}

    circuits = load_json(paths["circuits"])
    frontier = load_json(paths["frontier"])
    fidelity = load_json(paths["fidelity"])
    anchor = load_json(paths["anchor_meta"])
    circuit_rows = circuits.get("certified", {})
    frontier_rows = {k: v for k, v in frontier.items() if not k.startswith("_")}
    fidelity_rows = {k: v for k, v in fidelity.items() if not k.startswith("_")}
    ship = frontier.get("_composite_ship", {})
    pareto = ship.get("pareto", [])
    fidelity_point = min(pareto, key=lambda row: row["ce"]) if pareto else None
    unique_heads: set[str] = set()
    for row in circuit_rows.values():
        heads = row.get("heads", [])
        if isinstance(heads, list):
            unique_heads.update(map(str, heads))
    return {
        "available": True,
        "scope": "inventory only; counts are not a fraction of model computation",
        "certified_behavior_entries": len(circuit_rows),
        "unique_listed_heads": len(unique_heads),
        "components_with_frontier_candidates": len(frontier_rows),
        "components_with_best_fidelity_seed": len(fidelity_rows),
        "frozen_anchor": {
            "version": anchor["anchor_version"],
            "clean_ce": anchor["clean_ce"],
            "n_components": anchor["n_components"],
            "partial": anchor["partial"],
            "data_budget": anchor["data_budget"],
        },
        "current_composite": {
            "top_level_targets_replaced": "36/36",
            "clean_ce": anchor["clean_ce"],
            "composite_ce": fidelity_point["ce"] if fidelity_point else None,
            "delta_ce": round(fidelity_point["ce"] - anchor["clean_ce"], 4) if fidelity_point else None,
            "approximate_gbit": fidelity_point.get("gbit") if fidelity_point else None,
            "ce_error": fidelity_point.get("err") if fidelity_point else None,
            "behavioral_certification": ship.get("behavioral_certification"),
            "pricing_caveat": "ship price is manually approximated and is not yet produced by bench/complexity.py",
        },
        "sources": {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()},
    }


def build_balance_sheet(sources: dict[str, Path], theseus_root: Path | None = None) -> dict[str, Any]:
    data = {name: load_json(path) for name, path in sources.items()}
    coverage = data["causal_coverage"]["ledger"]
    subst = data["analytic_substitution"]
    named = data["named_understanding"]
    composed = data["composed_replacement"]
    content = data["content_passthrough"]
    upstream = data["upstream_interface"]
    question = data["question_channel"]
    output = data["output_slice"]
    inventory = registry_inventory(theseus_root)

    subst_recovery = 1.0 - subst["chain_pca"]["dCE"] / subst["available_headroom_to_floor"]
    composed_dce = composed["ce_standin_all"] - composed["ce_full"]
    composed_recovery = composed["whole_model_recovery"]
    named_fraction = coverage["coverage_fractions_of_headroom"]["named_ext_found"]["frac"]

    # Catch stale or silently redefined source metrics before emitting a headline.
    assert close(subst_recovery, subst["frac_of_floor_captured"]), "substitution denominator changed"
    recomputed_composed = 1.0 - composed_dce / (composed["ce_meanablate_all"] - composed["ce_full"])
    assert close(recomputed_composed, composed_recovery), "composition denominator changed"
    assert close(named_fraction, 1.0 - coverage["not_found_fraction_of_headroom"]["ext"]), "coverage does not close"
    assert close(content["recovery_vs_1070_denominator"], 1.0 - (content["ce_standin_scaffold"] - content["ce_full"]) / (composed["ce_meanablate_all"] - composed["ce_full"])), "content passthrough denominator changed"

    ledgers = {
        "representation": {
            "status": "exact_by_architecture_and_verified_elsewhere",
            "value": None,
            "currency": "tensor-network algebraic identity",
            "scope": "all model layers",
            "claim": "The network is exactly writable as typed tensor operations; this alone does not compress or interpret it.",
        },
        "analytic_interface_substitutability": {
            "value": round(subst_recovery, 6),
            "residual_delta_ce": subst["chain_pca"]["dCE"],
            "residual_delta_ce_se": subst["chain_pca"]["SE"],
            "currency": "1 - replacement delta-CE / joint-MLP mean-floor delta-CE",
            "denominator_delta_ce": subst["available_headroom_to_floor"],
            "scope": "whole-model PCA/head bottleneck plus composed analytic fold",
            "claim": "A compressed analytic interface can preserve nearly all floor-relative behavior.",
            "caveat": "This is interface fidelity, not a short human-legible program and not the composed named-stand-in result.",
        },
        "named_variable_understanding": {
            "value": named["token+topic+prev"]["mean"],
            "std_across_draws": named["token+topic+prev"]["std"],
            "range": [named["token+topic+prev"]["min"], named["token+topic+prev"]["max"]],
            "currency": "recovery of module behavior by frozen token+topic+previous-token variables",
            "scope": "four independent FineWeb draws; local/module replacement ledger",
            "claim": "Simple named covariates explain a stable minority of module behavior.",
            "caveat": "Not directly comparable to path-ablation headroom or whole-model composed recovery.",
        },
        "causal_path_coverage": {
            "value": named_fraction,
            "unnamed_fraction": coverage["not_found_fraction_of_headroom"]["ext"],
            "named_joint_delta_ce": coverage["named_ext_joint"]["dCE"],
            "full_headroom_delta_ce": coverage["full_headroom"]["dCE"],
            "non_axis_aligned_fraction": coverage["coverage_fractions_of_headroom"]["non_axis_aligned_residual"]["frac"],
            "hard_fraction_of_unnamed_axis_paths": coverage["unnamed_easy_vs_hard"]["global"]["hard_fraction"],
            "currency": "joint mean-ablation delta-CE / all-head+all-MLP mean-ablation delta-CE",
            "scope": "26 named head/top-SVD paths on held FW[448:600,:128]",
            "claim": "The verified named paths cover about one ninth of global ablation headroom.",
            "caveat": "Mean-ablation interactions are large: joint-234 is 2.87x the sum of positive single paths.",
        },
        "legacy_composition_stress_test": {
            "value": composed_recovery,
            "residual_fraction": round(1.0 - composed_recovery, 6),
            "standin_delta_ce": round(composed_dce, 4),
            "currency": "1 - all-stand-in delta-CE / all-module mean-ablation delta-CE",
            "scope": "36 stand-ins installed together in the running model",
            "claim": "This older controlled arc demonstrates severe composition failure among individually useful stand-ins.",
            "caveat": "This is a diagnostic predecessor, not the current shipped composite. It uses a different surrogate family and denominator from analytic-interface substitutability.",
        },
        "content_passthrough_diagnostic": {
            "value": content["recovery_vs_1070_denominator"],
            "scaffold_local_denominator_value": content["scaffold_recovery_content_passthrough"],
            "currency": "recovery with middle content preserved, reported against the all-36 mean-ablation denominator",
            "scope": "26 scaffold replacements with selected content live",
            "claim": "Preserving content triples whole-model recovery from 0.124 to 0.392, localizing much of the gap to contextual content and compounding.",
        },
        "smooth_interface_predictability": {
            "value": upstream["upstream_smoothmap_understanding"],
            "shuffled_null": upstream["shuffled_feature_null"],
            "currency": "recovery by a low-rank upstream smooth-map feature interface",
            "scope": "separate upstream-interface benchmark",
            "claim": "A distributed learned interface is strongly predictive even when named covariates are not.",
            "caveat": "This measures predictive compression, not semantic legibility.",
        },
        "causal_instrument_validation": {
            "question_pairwise_frozen_normalized_error": question["frozen"]["comparisons"]["question_true_raw"]["pairwise_normalized_error"],
            "question_additive_frozen_normalized_error": question["frozen"]["comparisons"]["question_true_raw"]["additive_normalized_error"],
            "output_basis_recall": output["summary"]["evaluation_recall"],
            "output_basis_random_recall": output["summary"]["random_recall"],
            "output_basis_damage_fraction_of_oracle": output["summary"]["winner_class_rise"] / output["summary"]["oracle_class_rise"],
            "currency": "held-out intervention prediction and circuit-localization diagnostics",
            "scope": "question circuit and disjoint output classes",
            "claim": "Polynomial pairwise terms predict interventions; a weights-only output basis locates circuits but is not yet a sufficient causal control basis.",
        },
    }

    current_ship = inventory.get("current_composite", {})
    current_delta = current_ship.get("delta_ce")
    return {
        "schema_version": 1,
        "model": "bilin18",
        "north_star": "a smaller typed tensor program that predicts natural/OOD behavior and held-out interventions, composes, and supports selective edits",
        "reporting_rule": "Never average or substitute ledger values across currencies. Every percentage must retain its scope, null, and denominator.",
        "metrics_not_averagable": True,
        "ledgers": ledgers,
        "current_bottleneck": {
            "metric": "current_36_of_36_composite_delta_ce",
            "top_level_replacement_scope": current_ship.get("top_level_targets_replaced"),
            "target_global_delta_ce": 0.02,
            "observed_global_delta_ce": current_delta,
            "composite_ce": current_ship.get("composite_ce"),
            "clean_ce": current_ship.get("clean_ce"),
            "interpretation": "The tensor structure supplies exact identities and useful interfaces, and every top-level module has a replacement, but the certified composite still loses about 0.93 nats. Replacement scope is not reverse-engineering completeness.",
        },
        "ranked_actions": [
            {
                "priority": 1,
                "action": "Build and evaluate replacements against the residual of the current whole-model composite, with content live and shared factors priced once.",
                "why": "Directly attacks the current 36/36 ship's roughly 0.93-nat gap and avoids duplicating already-substitutable interfaces.",
            },
            {
                "priority": 2,
                "action": "Localize that residual by layer, token/output class, and held-out intervention family.",
                "why": "The causal ledger says 89.08% of ablation headroom is unnamed, including a 36.37% non-axis-aligned residual.",
            },
            {
                "priority": 3,
                "action": "Compile joint vector-valued quadratic fragments with shared projection dictionaries between RMSNorm boundaries.",
                "why": "The scalar question slice proves exact multiplicative simplification is possible; sharing is required for whole-program savings.",
            },
            {
                "priority": 4,
                "action": "Use behavior-agnostic output bases for discovery only, then learn/test a distinct causal control basis on disjoint classes.",
                "why": "The current output basis has 40.4% circuit recall versus 7.5% random but only 13.5% of oracle removal damage.",
            },
            {
                "priority": 5,
                "action": "Preregister selective edit/transplant tests for every fragment admitted to the composite.",
                "why": "Editing and held-out intervention transport, not reconstruction alone, are the final reverse-engineering criterion.",
            },
        ],
        "registry_inventory": inventory,
        "sources": {name: {"path": str(path), "sha256": digest(path)} for name, path in sources.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "whole_model_balance_sheet.json")
    parser.add_argument("--theseus-root", type=Path, default=Path("/workspace/theseus-bench"))
    args = parser.parse_args()
    sheet = build_balance_sheet(DEFAULT_SOURCES, args.theseus_root)
    args.output.write_text(json.dumps(sheet, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps({
        "analytic_interface_substitutability": sheet["ledgers"]["analytic_interface_substitutability"]["value"],
        "named_variable_understanding": sheet["ledgers"]["named_variable_understanding"]["value"],
        "causal_path_coverage": sheet["ledgers"]["causal_path_coverage"]["value"],
        "legacy_composition_stress_test": sheet["ledgers"]["legacy_composition_stress_test"]["value"],
    }, indent=2))


if __name__ == "__main__":
    main()
