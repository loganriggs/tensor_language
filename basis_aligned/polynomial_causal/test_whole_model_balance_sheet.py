import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("whole_model_balance_sheet.py")
SPEC = importlib.util.spec_from_file_location("whole_model_balance_sheet", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_balance_sheet_keeps_distinct_currencies():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, Path("/workspace/theseus-bench"))
    ledgers = sheet["ledgers"]
    assert sheet["metrics_not_averagable"] is True
    assert ledgers["analytic_interface_substitutability"]["value"] > 0.99
    assert 0.25 < ledgers["named_variable_understanding"]["value"] < 0.5
    assert 0.05 < ledgers["causal_path_coverage"]["value"] < 0.2
    assert ledgers["legacy_composition_stress_test"]["value"] < 0.2
    assert len({row["currency"] for row in ledgers.values()}) == len(ledgers)


def test_balance_sheet_closes_primary_denominators():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    causal = sheet["ledgers"]["causal_path_coverage"]
    composed = sheet["ledgers"]["legacy_composition_stress_test"]
    assert abs(causal["value"] + causal["unnamed_fraction"] - 1.0) < 1e-6
    assert abs(composed["value"] + composed["residual_fraction"] - 1.0) < 1e-6
    assert sheet["registry_inventory"]["available"] is False


def test_current_composite_uses_frozen_registry():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, Path("/workspace/theseus-bench"))
    ship = sheet["registry_inventory"]["current_composite"]
    assert ship["top_level_targets_replaced"] == "36/36"
    assert abs(ship["delta_ce"] - (ship["composite_ce"] - ship["clean_ce"])) < 1e-6
    assert sheet["current_bottleneck"]["observed_global_delta_ce"] == ship["delta_ce"]


def test_output_basis_is_locator_not_controller():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    instrument = sheet["ledgers"]["causal_instrument_validation"]
    assert instrument["output_basis_recall"] > 2 * instrument["output_basis_random_recall"]
    assert instrument["output_basis_damage_fraction_of_oracle"] < 0.5


def test_current_ship_residual_partitions_close_without_conflation():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    residual = sheet["ledgers"]["current_composite_residual_localization"]
    assert abs(sum(residual["legacy_target_cell_damage_shares"].values()) - 1.0) < 0.002
    assert abs(sum(residual["legacy_sequential_top100_most_frequent_group_shares"].values()) - 1.0) < 0.002
    assert abs(sum(residual["factorial_heldout_cell_damage_shares"].values()) - 1.0) < 1e-9
    shapley = residual["factorial_heldout_weighted_group_shapley_nats"]
    assert abs(sum(shapley.values()) - 0.8727067046695285) < 1e-6
    assert shapley["mlp012"] > 0.7
    assert residual["factorial_heldout_cell_group_shapley_nats"]["novel_rare"]["mlp012"] > 1.0
    assert min(residual["factorial_heldout_cell_interaction_l1_fraction"].values()) > 0.4
    assert residual["top100_most_damaged_token_type_damage_share"] != residual["top100_most_frequent_token_damage_share"]
    assert "cannot be multiplied" in residual["caveat"]


def test_full_vector_and_scalar_complexity_are_separate_scopes():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    arithmetic = sheet["ledgers"]["arithmetic_complexity_bounds"]
    assert arithmetic["full_vector_numerical_product_lower"] == 1152
    assert arithmetic["native_product_upper"] == 4608
    assert arithmetic["observed_sigma_min_over_max_range"][0] > 1e-4
    assert arithmetic["observed_sigma_min_over_max_range"][1] < 1e-2
    assert "symbolic rank proof" in arithmetic["caveat"]
    assert "natural-activation" in arithmetic["caveat"]
    assert arithmetic["question_scalar_exact_products"] == 1


def test_matched_product_geometry_is_not_promoted_after_causal_failure():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    matched = sheet["ledgers"]["matched_cost_causal_compiler"]
    assert matched["registered_predictions"]["A_pair_fp32_exact"] is True
    assert matched["registered_predictions"]["B_pair_bf16_stable"] is True
    assert matched["registered_predictions"]["C_pair_geometry_beats_best_square"] is False
    assert matched["square_scalar_relative_rmse"]["heldout"] > 0.35
    assert matched["square_question_kl"]["heldout"] < 1e-4
    assert matched["square_question_kl_fraction_of_zero_rank2"]["heldout"] < 0.01


def test_failed_early_product_frontier_promotes_linear_only_as_local_candidate():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    frontier = sheet["ledgers"]["local_content_compiler_frontier"]
    assert frontier["registered_predictions"]["A_paired_beats_native_and_linear"] is False
    assert frontier["registered_predictions"]["B_paired_reaches_r2_0p60"] is False
    for site in ("mlp0", "mlp1", "mlp2"):
        assert frontier["heldout_r2"][site]["linear"] > frontier["heldout_r2"][site]["learned_paired"]
        assert frontier["heldout_r2"][site]["learned_paired"] > frontier["heldout_r2"][site]["native_selected"]
    assert "rather than the current ship residual" in frontier["caveat"]
    assert 0.68 < frontier["mlp0_native_fraction_of_linear_r2"] < 0.70
    assert frontier["mlp0_native_amortized_r2_per_parameter_advantage"] > 20
    assert "does not make them free" in frontier["pricing_rule"]


def test_failed_hankel_and_ood_content_claims_prune_the_strategy():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    hankel = sheet["ledgers"]["sequence_state_program_test"]
    ood = sheet["ledgers"]["ood_content_interface"]
    assert hankel["registered_predictions"]["low_rank_beats_additive"] is False
    assert hankel["heldout_splice_ce_excess"] > 3.5
    assert ood["content_generalization_prediction"] is False
    assert ood["code_variance_retained_by_prose_basis"] < 0.2
    assert ood["code_variance_retained_by_code_basis"] > 0.5
    assert ood["value"] < 0.35


def test_head_grain_is_kept_as_local_law_not_whole_model_claim():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    grain = sheet["ledgers"]["controlled_tensor_head_grain"]
    assert grain["registered_predictions"]["pred_a_pron_attn9_is_9.6_ratio_ge4_2of3"] is True
    assert grain["registered_predictions"]["pred_b_ques_attn10_is_10.5_ratio_ge10_2of3"] is True
    assert grain["registered_predictions"]["pred_c_lambda_beats_random_1.5x_both_cells"] is False
    assert "not yet a replacement" in grain["caveat"]


def test_compression_is_priced_fidelity_not_circuit_selectivity():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    compression = sheet["ledgers"]["compression_selectivity_boundary"]
    assert compression["question_class_function_kept_rank32"] > 0.96
    assert compression["pronoun_class_function_kept_rank32"] > 0.92
    ratios = compression["class_to_global_damage_ratio_by_rank"]
    assert min(ratios.values()) > 0.9
    assert max(ratios.values()) < 1.2
    assert max(compression["exact_circuit_marginal_recovery_in_compressed_background"].values()) < 0
    assert "not a circuit-selective operator" in compression["claim"]
