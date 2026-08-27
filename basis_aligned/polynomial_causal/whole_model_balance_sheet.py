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
    "sequence_hankel": HERE / "hankel_rank_audit_results.json",
    "content_ood": TENSOR_ROOT / "basis_aligned/bilinear_quotient/content_ood_code_results.json",
    "ood_bands": TENSOR_ROOT / "basis_aligned/bilinear_quotient/ood_band_importance_results.json",
    "headgrain_control": TENSOR_ROOT / "basis_aligned/bilinear_quotient/headgrain_control2_results.json",
    "writer_floor_question": TENSOR_ROOT / "basis_aligned/bilinear_quotient/writer_floor_question_results.json",
    "writer_floor_pronouns": TENSOR_ROOT / "basis_aligned/bilinear_quotient/writer_floor_pronouns_results.json",
    "writer_floor_absmass": TENSOR_ROOT / "basis_aligned/bilinear_quotient/writer_floor_absmass_results.json",
    "extraction_rank": TENSOR_ROOT / "basis_aligned/bilinear_quotient/extraction_rank_results.json",
    "extraction_question": TENSOR_ROOT / "basis_aligned/bilinear_quotient/extraction_bg_results.json",
    "extraction_pronouns": TENSOR_ROOT / "basis_aligned/bilinear_quotient/extraction_bg_p_results.json",
    "compression_rank": TENSOR_ROOT / "basis_aligned/bilinear_quotient/compression_rank3_results.json",
    "local_ship_oracle_curated_dev_v2": TENSOR_ROOT / "basis_aligned/bilinear_quotient/ship_content_oracle_curated_dev_v2_results.json",
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
    sequence_hankel = data["sequence_hankel"]
    content_ood = data["content_ood"]
    ood_bands = data["ood_bands"]
    headgrain = data["headgrain_control"]
    writer_floor = data["writer_floor_question"]
    writer_floor_pronouns = data["writer_floor_pronouns"]
    writer_floor_absmass = data["writer_floor_absmass"]
    extraction_rank = data["extraction_rank"]
    extraction_question = data["extraction_question"]
    extraction_pronouns = data["extraction_pronouns"]
    compression_rank = data["compression_rank"]
    local_oracle = data["local_ship_oracle_curated_dev_v2"]
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
            "caveat": "The prior 0.500 result selects the 100 most-damaged token types, whereas the prior 0.499 group share and factorial 0.319 partition use the 100 most-frequent token types. They cannot be multiplied. Interactions consume 43-64% of each cell's total effect, so the Shapley localization licenses only a joint-ship correction, not an independent module claim. The v1 copy mask covers distances 2-65 and its rare vocabulary is split-specific; global group allocations remain valid but token-cell labels require a corrected frozen-stratum replication.",
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
        "sequence_state_program_test": {
            "value": sequence_hankel["heldout"]["suffix_logprob"]["best_improvement"],
            "heldout_rank90": sequence_hankel["heldout"]["suffix_logprob"]["rank90"],
            "heldout_stable_rank": sequence_hankel["heldout"]["suffix_logprob"]["stable_rank"],
            "heldout_splice_ce_excess": sequence_hankel["heldout"]["splice_ce_excess"],
            "registered_predictions": sequence_hankel["predictions"],
            "currency": "best held-out low-rank RMSE improvement over row-plus-column additive prefix/suffix baseline",
            "scope": "48 prefixes x 48 suffixes, 64-token prefixes and 8-token suffixes; discovery and held-out spliced FineWeb grids",
            "claim": "The registered compact Hankel-state hypothesis is rejected at this interface: low rank does not beat the additive baseline and the apparent ranks replicate near 19-24 rather than collapsing.",
            "caveat": "Splicing is strongly distribution-shifting (held-out +3.54 CE), so this rejects the present probe, not every finite-state or predictive-state representation of natural continuations.",
        },
        "ood_content_interface": {
            "value": content_ood["code_frac_of_own_captured_by_prose"],
            "code_variance_retained_by_prose_basis": content_ood["retained"]["code_by_prose_subspace"],
            "code_variance_retained_by_code_basis": content_ood["retained"]["code_by_own_top64"],
            "code_variance_retained_by_random_basis": content_ood["retained"]["code_by_random64"],
            "prose_code_subspace_overlap": content_ood["prose_code_subspace_overlap"],
            "content_generalization_prediction": content_ood["pred_a_register_general"],
            "grammar_band_holds_on_code": ood_bands["grammar_holds_on_code"],
            "currency": "code contextual-variation capture by the prose content basis divided by code's own top-64 capture",
            "scope": "150 code and 150 prose sequences at residual layers 8, 10, and 12",
            "claim": "The current prose-derived content basis is not a universal OOD interface: it captures only 16.6% of code variation, or 32.2% of the code-local top-64 ceiling.",
            "caveat": "It remains 3.0x above a random 64D basis on code, so it contains transferable signal; a universal compiler must test a transported or typed mixture of content coordinates rather than treating this basis as domain invariant.",
        },
        "controlled_tensor_head_grain": {
            "question_lambda_to_random_median_ratio": headgrain["cells"]["question@mlp11"]["ratio_lambda_over_random"],
            "pronoun_lambda_to_random_median_ratio": headgrain["cells"]["pronouns@mlp17"]["ratio_lambda_over_random"],
            "registered_predictions": headgrain["predictions"],
            "currency": "published |eigenvalue|-slice head-read concentration divided by matched-rank random-subspace concentration",
            "scope": "question@MLP11 rank 2 and pronouns@MLP17 rank 8 over three disjoint 96-row samples",
            "claim": "The tensor slice identifies sharply concentrated reader heads at the registered circuit layers, but the whole-stack median separation gate fails.",
            "caveat": "This is a controlled local wiring law, not yet a replacement, CE recovery, or proof that the same grain composes across behaviors and layers.",
        },
        "tensor_writer_specificity": {
            "question_positive_only_lambda_mean_top4_share": writer_floor["mean_share"]["lambda"],
            "question_positive_only_random_mean_top4_share": writer_floor["mean_share"]["random"],
            "question_positive_only_consensus_overlap_of4": writer_floor["overlap_lambda_random_of4"],
            "question_positive_only_lambda_consensus": writer_floor["consensus"]["lambda"],
            "question_positive_only_random_consensus": writer_floor["consensus"]["random"],
            "question_positive_only_floor_corrected_top4": writer_floor["lambda_top4_floor_corrected"],
            "question_class_counts": [row["class_n"] for row in writer_floor["arms"]["lambda"]],
            "question_circuit_clean_samples_positive_only": writer_floor["circuit_clean_samples"],
            "question_registered_predictions_positive_only": writer_floor["predictions"],
            "pronoun_positive_only_lambda_mean_top6_share": writer_floor_pronouns["mean_share"]["lambda"],
            "pronoun_positive_only_random_mean_top6_share": writer_floor_pronouns["mean_share"]["random"],
            "pronoun_registered_predictions_degenerate": writer_floor_pronouns["predictions"],
            "question_absolute_mass_lambda_mean_top4_share": writer_floor_absmass["cells"]["question@mlp11"]["mean_share"]["lambda"],
            "question_absolute_mass_random_mean_top4_share": writer_floor_absmass["cells"]["question@mlp11"]["mean_share"]["random"],
            "question_absolute_mass_share_gap_lambda_minus_random": writer_floor_absmass["cells"]["question@mlp11"]["mean_share"]["lambda"] - writer_floor_absmass["cells"]["question@mlp11"]["mean_share"]["random"],
            "question_absolute_mass_lambda_members_absent_from_random": writer_floor_absmass["cells"]["question@mlp11"]["absent_from_random"],
            "pronoun_absolute_mass_lambda_mean_top6_share": writer_floor_absmass["cells"]["pronouns@mlp17"]["mean_share"]["lambda"],
            "pronoun_absolute_mass_random_mean_top6_share": writer_floor_absmass["cells"]["pronouns@mlp17"]["mean_share"]["random"],
            "pronoun_absolute_mass_share_gap_lambda_minus_random": writer_floor_absmass["cells"]["pronouns@mlp17"]["mean_share"]["lambda"] - writer_floor_absmass["cells"]["pronouns@mlp17"]["mean_share"]["random"],
            "pronoun_absolute_mass_lambda_members_absent_from_random": writer_floor_absmass["cells"]["pronouns@mlp17"]["absent_from_random"],
            "absolute_mass_registered_predictions": writer_floor_absmass["predictions"],
            "currency": "matched-rank within-run absolute attribution-mass top-k share and component membership at question@MLP11 and pronouns@MLP17",
            "measured_currency": "top-k share and membership over positive signed component contributions",
            "original_currency": "top-k share and membership over absolute component attribution mass",
            "currency_matches_original_writer_claims": True,
            "original_absolute_mass_null_tested": True,
            "positive_only_audit_status": "noncommensurate with the published statistic; question within-run comparison narrow, pronoun measurement saturated; cross-references withdrawn",
            "status": "absolute-mass null is cell-dependent: question slice is more concentrated than random, pronoun slice is substantially less concentrated than random",
            "scope": "three disjoint 333-row chunks from curated_rows.pt; question counts 105/127/102 and pronoun counts 434/408/399; local census rows, not fresh oracle rows",
            "claim": "With the published absolute-mass statistic, question@MLP11 concentrates above its matched-rank null (.5563 versus .4489), while pronouns@MLP17 concentrates far below its null (.5846 versus .7295). Concentration has slice-specific sign; matched-rank null subtraction, not raw share, is the informative quantity.",
            "caveat": "The published .718/.482 values used different rows and are not numerically comparable to this local corpus. The within-run null gaps are valid, but they establish a local structural diagnostic—not causal sufficiency, generalization, selective editability, or whole-model programmability.",
        },
        "compression_selectivity_boundary": {
            "question_class_function_kept_rank32": 1.0 - extraction_question["res"]["bg_class_rise"] / extraction_question["res"]["const_class_rise_ref"],
            "pronoun_class_function_kept_rank32": 1.0 - extraction_pronouns["res"]["bg_class_rise"] / extraction_pronouns["res"]["const_class_rise_ref"],
            "exact_circuit_marginal_recovery_in_compressed_background": {
                "question": extraction_question["res"]["rec_in_bg"],
                "pronouns": extraction_pronouns["res"]["rec_in_bg"],
            },
            "class_to_global_damage_ratio_by_rank": {
                rank: row["class_rise"] / row["global_rise"]
                for rank, row in extraction_rank["res"]["by_rank"].items()
            },
            "rank_sweep_predictions": {
                key: extraction_rank[key]
                for key in ("pred_a_monotone", "pred_b_r8_half", "pred_c_knee_16")
            },
            "unit_truncation_class_rise_spearman": compression_rank["spearman_class_rise"],
            "unit_truncation_selectivity_within_2x": compression_rank["sel_within_2x"],
            "currency": "class-specific versus global CE damage under rank compression, plus circuit-head marginal recovery inside the compressed background",
            "scope": "question rank sweep r4/r8/r16/r32; rank-32 question/pronoun backgrounds; eight class-specific top-unit truncation controls",
            "claim": "Compression is a useful self-consistent replacement background, but not a circuit-selective operator: class and global damage move together at every tested rank, and restoring exact named heads does not recover function inside that background.",
            "caveat": "This rejects generic compression as an extraction or selective-edit primitive at the tested interfaces. It does not reject compression as a priced fidelity component, nor behavior-specific gauges that pass direct intervention-transport tests.",
        },
        "early_mlp_live_correction_oracle_exploratory": {
            "authority": local_oracle["config"]["authority"],
            "authorized_for_scored_experiments": local_oracle["config"]["authorized_for_scored_experiments"],
            "development_candidate_sites": local_oracle["development_candidate_sites"],
            "training_license_sites": local_oracle["training_license_sites"],
            "projection_rank": local_oracle["config"]["projection_rank"],
            "matched_nulls_per_site": local_oracle["config"]["matched_nulls_per_site"],
            "heldout_global_gain_nats": {
                str(site): {
                    arm: local_oracle["paired_gains"][str(site)]["heldout"][arm]["global"]["mean"]
                    for arm in ("full", "content", "local_pca")
                }
                for site in (0, 1, 2)
            },
            "site_decisions": local_oracle["site_decisions"],
            "currency": "paired held-out CE gain from exact live original-minus-plank correction versus frozen ship, with exact one-sided 20-null gate for the rank-64 content arm",
            "scope": "one frozen same-realization local-curated ship; whole-document-disjoint ship-fit/basis/discovery/heldout rows, but frozen content-factor source-document provenance is unavailable",
            "claim": "The unrestricted live correction is strongly beneficial at MLP0 and MLP1 (+0.116 and +0.153 nats), while the proposed rank-64 content arm recovers only 8.1% and 1.9% of those gains and loses to every matched null. At MLP2 the unrestricted correction is harmful (-0.211 nats), so the content arm's +0.031-nat null win is a regularizing sign rather than a faithful replacement. No site passes the complete candidate gate.",
            "caveat": "This run has authority none and licenses no site: the content factor may overlap the curated corpus, so it is exploratory within-realization evidence only. Local-PCA arms are not compared to the content-matched nulls because their correction RMS differs. The result prunes the current prose-content factorization; it does not prune a fresh authoritative live oracle or a differently typed causal interface.",
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
                "action": "Finish the pinned FineWeb shard, prove every registered row tensor against the canonical file/config order, freeze the complete rowcache, and rerun the same-realization live-correction oracle at MLP0-2 with authoritative provenance.",
                "why": "The isolated local run found strong unrestricted correction at MLP0/1 but decisively falsified the current content arm there; MLP2 was sign-unstable. A canonical rerun is the highest-information test of whether those causal signs survive fresh data and which site, if any, can license downstream composition.",
            },
            {
                "priority": 2,
                "action": "If and only if the authoritative oracle licenses a site, repeat the optimizer-free singleton screen on code using frozen prose, code-local, transported, and matched-null bases before fitting a predictor.",
                "why": "The local oracle now joins the earlier code result in rejecting prose content as a universal API: it loses to all nulls at MLP0/1, while prose retains only 16.6% of code variation versus 51.5% for code-local coordinates. Conditional OOD screening distinguishes a typed interface from a non-content residual without spending training compute.",
            },
            {
                "priority": 3,
                "action": "Extend the frozen group factorial to powered output slices and held-out intervention families, with corrected fixed token strata and alternate ship backgrounds.",
                "why": "Token CE localizes the deployed residual but cannot establish causal transport or selective edits; large negative interactions also require checking whether a fragment's intervention response changes with its replacement background.",
            },
            {
                "priority": 4,
                "action": "Execute the preregistered no-teacher-forcing L8-to-L11-to-L14 gauge-transport triangle, first gating destination-subspace sufficiency, then direct response prediction, chain composition, and alternate-background transfer.",
                "why": "Local bases often locate circuits, but affine donor transport has near-zero median fidelity, repeated content patching teacher-forces every clamp, and the output basis preserves only 13.5% of oracle removal damage. The commuting triangle separates a locator failure from transport failure and transport failure from composition failure while pricing the physical map, coordinate field, and repeated interfaces.",
            },
            {
                "priority": 5,
                "action": "Only after oracle and transport gates pass, fit the residual predictor and factor it through linear, native-product, paired-product, and tensor-head-grain programs at standalone and amortized prices.",
                "why": "The early product frontier and Hankel probe rejected premature structural compression, while the compression rank sweep shows class and global behavior degrade together rather than exposing a privileged circuit core. Tensor structure should compete only at a causally licensed, OOD-scoped interface where simplicity can be tested for composition and editability.",
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
