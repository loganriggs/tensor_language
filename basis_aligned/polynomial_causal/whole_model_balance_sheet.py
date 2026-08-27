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
    "ship_error_cells": TENSOR_ROOT / "basis_aligned/bilinear_quotient/ship_error_mine_results.json",
    "ship_error_groups": TENSOR_ROOT / "basis_aligned/bilinear_quotient/ship_error_attrib_results.json",
    "ship_error_factorial": TENSOR_ROOT / "basis_aligned/bilinear_quotient/ship_error_factorial_results.json",
    "ship_behavior_state": TENSOR_ROOT / "basis_aligned/bilinear_quotient/state_in_full_ship_results.json",
    "mlp_product_rank": HERE / "mlp_product_rank_audit_results.json",
    "question_one_product": HERE / "question_one_product_results.json",
    "content_product_frontier": HERE / "content_product_frontier_results.json",
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
    ship_cells = data["ship_error_cells"]
    ship_groups = data["ship_error_groups"]
    ship_factorial = data["ship_error_factorial"]
    ship_state = data["ship_behavior_state"]
    product_rank = data["mlp_product_rank"]
    question_product = data["question_one_product"]
    content_frontier = data["content_product_frontier"]
    factorial_heldout = ship_factorial["splits"]["heldout"]
    factorial_primary = factorial_heldout["primary"]
    factorial_cells = factorial_primary["cells"]
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
    assert close(sum(ship_cells["cell_shares"].values()), 1.0, atol=2e-3), "ship error cells do not close"
    assert close(sum(ship_groups["shares"].values()), 1.0, atol=2e-3), "ship group attribution does not close"

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
        "current_composite_residual_localization": {
            "global_clean_ce_same_run": ship_state["clean"]["global"],
            "global_ship_ce_same_run": ship_state["full_ship"]["global"],
            "legacy_target_cell_damage_shares": ship_cells["cell_shares"],
            "top100_most_damaged_token_type_damage_share": ship_cells["top100_damage_share"],
            "legacy_sequential_top100_most_frequent_group_shares": ship_groups["shares"],
            "factorial_heldout_cell_damage_shares": factorial_primary["cell_damage_shares"],
            "factorial_heldout_weighted_group_shapley_nats": factorial_primary["weighted_shapley"],
            "factorial_heldout_cell_group_shapley_nats": {
                cell: row["shapley"] for cell, row in factorial_cells.items()
            },
            "factorial_heldout_cell_interaction_l1_fraction": {
                cell: row["interaction_l1_fraction_of_total"] for cell, row in factorial_cells.items()
            },
            "top100_most_frequent_token_damage_share": factorial_heldout["frequency"]["cell_damage_shares"]["top100"],
            "currency": "signed nats and shares of current K=3072 ship CE damage within explicitly named partitions",
            "scope": "the 2^3 factorial crosses attention, MLP0-2, and deep replacements over global copy/novel-frequency cells and the separate most-frequent-token partition",
            "claim": "Held-out MLP0-2 Shapley effect is 0.728 of 0.873 global ship nats and 1.078 of 1.176 novel-rare nats; the early group is the stable dominant residual source.",
            "caveat": "The prior 0.500 result selects the 100 most-damaged token types, whereas the prior 0.499 group share and factorial 0.319 partition use the 100 most-frequent token types. They cannot be multiplied. Interactions consume 43-64% of each cell's total effect, so the Shapley localization licenses only a joint-ship correction, not an independent module claim.",
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
        "arithmetic_complexity_bounds": {
            "audited_layers": product_rank["config"]["layers"],
            "full_vector_numerical_product_lower": min(row["conservative_numerical_lower_rtol_1e-6"] for row in product_rank["layers"].values()),
            "native_product_upper": min(row["explicit_products_upper"] for row in product_rank["layers"].values()),
            "question_scalar_exact_products": 1,
            "observed_sigma_min_over_max_range": [
                min(sketch["sigma_1152_over_max"] for row in product_rank["layers"].values() for sketch in row["sketches"]),
                max(sketch["sigma_1152_over_max"] for row in product_rank["layers"].values() for sketch in row["sketches"]),
            ],
            "currency": "scalar multiplication gates in the frozen sum_i c_i(a_i.x)(b_i.x) grammar",
            "scope": "randomized numerical coefficient-space output-flattening bounds for full MLPs 0,1,2,11,17; exact inertia certificate for the question scalar slice",
            "claim": "All audited full-vector maps lack near-exact coefficient degeneracy at registered tolerances through 1e-4, while the selected scalar slice has an exact one-product certificate.",
            "caveat": "The 1152 result is a two-sketch floating-point diagnostic whose thresholds all lie below the observed spectral tails. It is neither a symbolic rank proof nor a natural-activation incompressibility result, and the practical coefficient-space knee is unmeasured.",
        },
        "matched_cost_causal_compiler": {
            "pair_fp32_max_scalar_relative_rmse": max(row["pair_fp32"]["scalar_relative_rmse"] for row in question_product["evaluations"].values()),
            "pair_bf16_max_scalar_relative_rmse": max(row["pair_bf16"]["scalar_relative_rmse"] for row in question_product["evaluations"].values()),
            "square_scalar_relative_rmse": {
                split: row["square"]["scalar_relative_rmse"] for split, row in question_product["evaluations"].items()
            },
            "square_question_kl": {
                split: row["square"]["question_kl"] for split, row in question_product["evaluations"].items()
            },
            "square_question_kl_fraction_of_zero_rank2": question_product["square_question_kl_fraction_of_zero_rank2"],
            "registered_predictions": question_product["predictions"],
            "currency": "matched-one-product question-slice replacement fidelity on natural activations",
            "scope": "mlp11 selected positive/negative question-unembedding eigenpair; discovery and untouched held-out FineWeb rows",
            "claim": "The exact paired gate is stable even in bf16, but exact product geometry did not earn causal preference over the best one-square program: the registered held-out KL gate failed.",
            "caveat": "This is a rank-2 slice correction with the orthogonal MLP output live, not an MLP replacement. The square's held-out question KL is only 0.39% of zeroing the slice despite 35.4% scalar error, so reconstruction error badly overstates behavioral necessity here.",
        },
        "local_content_compiler_frontier": {
            "heldout_r2": {
                f"mlp{site}": {
                    arm: row[arm]["heldout_r2"]
                    for arm in ("linear", "native_selected", "random_products", "learned_paired")
                }
                for site, row in content_frontier["sites"].items()
            },
            "best_site": content_frontier["best_paired_site"],
            "registered_predictions": content_frontier["predictions"],
            "mlp0_native_fraction_of_linear_r2": content_frontier["sites"]["0"]["native_selected"]["heldout_r2"] / content_frontier["sites"]["0"]["linear"]["heldout_r2"],
            "mlp0_native_amortized_r2_per_parameter_advantage": (
                content_frontier["sites"]["0"]["native_selected"]["heldout_r2"]
                / content_frontier["sites"]["0"]["native_selected"]["amortized_new_parameters"]
                / (content_frontier["sites"]["0"]["linear"]["heldout_r2"]
                   / content_frontier["sites"]["0"]["linear"]["parameters"])
            ),
            "pricing_rule": "Standalone parameters are primary. Native factor projections may use the amortized-new price only after independent admission and payment as a shared library; provenance in the original model does not make them free.",
            "currency": "held-out whitened R2 at a frozen 64-dimensional local content-output API",
            "scope": "clean-model MLP0-2 outputs; 32 products; sequence-disjoint 96/48/48 train/validation/test rows; not installed in the ship",
            "claim": "At standalone price, learned paired and selected native products are dominated by a smaller linear map at all three early MLPs; tensor-specific compilation is rejected at this interface. Native products are a conditional amortized frontier only if their factors are legitimately shared.",
            "caveat": "The MLP0 native arm retains 69% of linear R2 for 1/35 the new parameters, but that price excludes its two factor projections. The linear winner predicts clean local outputs rather than the current ship residual, so it licenses a ship-correction test only after group x token-cell attribution, not whole-model admission.",
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
                "action": "Fit a content-restricted live-activation correction at MLP0-2 inside the complete ship and compare it with a matched random output basis and the already-installed generic rank-32 MLP2 glue.",
                "why": "The factorial assigns 0.728/0.873 held-out global nats and 1.078/1.176 novel-rare nats to MLP0-2, but 43-64% interaction forbids a marginal correction. This directly tests whether the clean-model content API is the missing composable interface rather than merely a predictive local basis.",
            },
            {
                "priority": 2,
                "action": "Extend the same frozen attention x MLP0-2 x deep factorial to powered output slices and held-out intervention families.",
                "why": "Token CE localizes the current residual but cannot establish that the simplified program transports causal behavior or selective edits; these are still the largest missing whole-model interfaces.",
            },
            {
                "priority": 3,
                "action": "Factor the licensed MLP0-2 correction internally only after it passes whole-ship CE and matched-random controls, preserving standalone and amortized prices.",
                "why": "The clean local product frontier rejected paired products in favor of linear maps. Tensor/product structure becomes useful only if it compresses a causally licensed live-ship correction without losing its held-out gain.",
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
