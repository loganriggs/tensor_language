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
    assert ship["same_run"]["paired_clean"] is True
    assert ship["same_run"]["clean_ce"] == 2.9455
    assert ship["same_run"]["composite_ce"] == 3.8431
    assert ship["same_run"]["delta_ce"] == 0.8976
    assert ship["cross_row_certificate"]["paired_clean"] is False
    assert ship["cross_row_certificate"]["composite_ce_mean"] == 3.8801
    assert ship["registry_pareto_display"]["paired_clean"] is False
    assert "may not be subtracted" in ship["registry_pareto_display"]["rule"]
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


def test_writer_null_preserves_metric_mismatch_and_measurement_failure():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    writer = sheet["ledgers"]["tensor_writer_specificity"]
    assert writer["currency_matches_original_writer_claims"] is True
    assert writer["original_absolute_mass_null_tested"] is True
    assert writer["question_registered_predictions_positive_only"]["pred_b_circuit_rule_specific_2of3"] is False
    assert writer["question_positive_only_consensus_overlap_of4"] == 1
    assert writer["question_circuit_clean_samples_positive_only"] == 0
    assert min(writer["question_class_counts"]) >= 100
    assert writer["question_positive_only_random_mean_top4_share"] > writer["question_positive_only_lambda_mean_top4_share"]
    assert writer["pronoun_positive_only_lambda_mean_top6_share"] > 0.99
    assert writer["question_absolute_mass_share_gap_lambda_minus_random"] > 0.10
    assert writer["pronoun_absolute_mass_share_gap_lambda_minus_random"] < -0.12
    assert writer["question_absolute_mass_lambda_members_absent_from_random"] == [2, 2, 2]
    assert writer["pronoun_absolute_mass_lambda_members_absent_from_random"] == [3, 3, 3]
    assert all(value is False for value in writer["absolute_mass_registered_predictions"].values())
    assert writer["null_share_natural_n_spearman"]["rho"] > 0.60
    assert writer["null_share_natural_n_spearman"]["p"] < 0.05
    assert writer["null_share_equalized_n_range"] > 0.16
    assert writer["null_share_equalized_n_range"] > 0.9 * writer["null_share_natural_n_range"]
    assert writer["null_share_n_control_predictions"]["pred_b_controlled_range_lt05"] is False
    assert "cell-dependent" in writer["status"]
    assert "matched-class" in writer["claim"]
    assert "withdraws cheap null predictors" in writer["caveat"]


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


def test_local_live_correction_oracle_prunes_content_without_licensing_science():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    oracle = sheet["ledgers"]["early_mlp_live_correction_oracle_exploratory"]
    assert oracle["authority"] == "none"
    assert oracle["authorized_for_scored_experiments"] is False
    assert oracle["development_candidate_sites"] == []
    assert oracle["training_license_sites"] == []
    assert oracle["heldout_global_gain_nats"]["0"]["full"] > 0.10
    assert oracle["heldout_global_gain_nats"]["1"]["full"] > 0.14
    assert oracle["heldout_global_gain_nats"]["2"]["full"] < -0.20
    assert oracle["site_decisions"]["0"]["exact_twenty_null_test"]["exact_one_sided_p"] == 1.0
    assert oracle["site_decisions"]["1"]["exact_twenty_null_test"]["exact_one_sided_p"] == 1.0
    assert oracle["site_decisions"]["2"]["exact_twenty_null_test"]["passes_5pct"] is True
    assert "regularizing sign" in oracle["claim"]
    assert "authority none" in oracle["caveat"]


def test_joint_early_oracle_requires_coupled_program_and_same_currency_denominator():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    oracle = sheet["ledgers"]["early_mlp_joint_live_oracle_exploratory"]
    assert oracle["authority"] == "none"
    assert oracle["authorized_for_scored_experiments"] is False
    assert oracle["training_license_sites"] == []
    heldout = oracle["heldout"]
    assert heldout["joint_gain"] > 0.50
    assert heldout["singleton_gain_sum"] < 0.06
    assert heldout["mlp2_conditional_marginal_after_mlp0_mlp1"] > 0.11
    assert heldout["interaction_l1_fraction_of_joint_gain"] > 1.7
    assert heldout["joint_gain_fraction_of_mlp012_residual"] is None
    assert abs(heldout["shapley_closure_error"]) < 1e-12
    assert all(oracle["registered_predictions"].values())
    assert "coupled causal program" in oracle["claim"]
    assert "deliberately null" in oracle["caveat"]


def test_authoritative_joint_oracle_replication_is_cluster_robust_but_not_a_program():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    oracle = sheet["ledgers"]["early_mlp_joint_live_oracle_authoritative"]
    assert oracle["authority"] == "canonical_fineweb"
    assert oracle["authorized_for_scored_experiments"] is True
    assert oracle["payload_self_authorized"] is False
    assert oracle["authorized_for_training"] is False
    assert oracle["training_license_sites"] == []
    heldout = oracle["heldout"]
    bootstrap = oracle["heldout_cluster_bootstrap"]
    assert heldout["joint_gain"] > 0.51
    assert bootstrap["joint_gain"]["ci95"][0] > 0.48
    assert heldout["joint_minus_singleton_sum"] > 0.45
    assert bootstrap["joint_minus_singleton_sum"]["ci95"][0] > 0.43
    assert bootstrap["mlp2_singleton_gain"]["ci95"][1] < -0.20
    assert bootstrap["mlp2_conditional_marginal_after_mlp0_mlp1"]["ci95"][0] > 0.09
    assert all(oracle["registered_predictions"].values())
    assert heldout["joint_gain_fraction_of_mlp012_residual"] is None
    assert oracle["state_integrity"]["component_tree_unchanged"] is True
    assert oracle["state_integrity"]["heldout_baseline_replay"][
        "max_abs_row_ce_difference"
    ] == 0.0
    assert "not corpus-wide" in oracle["caveat"]


def test_local_pca_strength_controls_license_only_an_oracle_subspace():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    oracle = sheet["ledgers"]["early_mlp_local_pca_strength_control_exploratory"]
    assert oracle["authority"] == "none"
    assert oracle["authorized_for_scored_experiments"] is False
    assert oracle["training_license_sites"] == []
    assert oracle["code_ood_licensed"] is False
    assert oracle["projection_rank"] == 64
    for site in ("0", "1"):
        assert oracle["site_decisions"][site]["passes_both_strength_controls"] is True
        for control in ("downstream_kl", "raw_rms"):
            decision = oracle["site_decisions"][site][control]
            assert decision["nulls_at_least_candidate"] == 0
            assert decision["exact_one_sided_p"] == 1 / 21
            assert decision["decision"]["passes"] is True
    assert oracle["heldout_fraction_of_full_oracle"]["0"] > 0.79
    assert oracle["heldout_fraction_of_full_oracle"]["1"] > 0.51
    assert "oracle-selected" in oracle["caveat"]


def test_authoritative_mixed_pca_composes_but_remains_an_oracle_subspace():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    oracle = sheet["ledgers"]["early_mlp_mixed_pca_oracle_authoritative"]
    assert oracle["authority"] == "canonical_fineweb"
    assert oracle["authorized_for_scored_experiments"] is True
    assert oracle["payload_self_authorized"] is False
    assert oracle["authorized_for_training"] is False
    assert oracle["training_license_sites"] == []
    assert oracle["projection_rank_per_site"] == 64
    assert all(oracle["registered_predictions"].values())
    assert len(oracle["registered_decisions"]["no_free_rider"]) == 12
    assert len(oracle["registered_decisions"]["same_background_40pct_margin"]) == 12
    assert all(oracle["registered_decisions"]["no_free_rider"].values())
    assert all(oracle["registered_decisions"]["same_background_40pct_margin"].values())
    assert oracle["heldout_projected_upstream_fraction_of_exact"] > 0.56
    assert oracle["heldout_projected_upstream_fraction_of_exact_with_mlp2_fixed"] > 0.63
    assert oracle["heldout_full_projected_package_fraction_of_exact_joint"] < 0.49
    assert oracle["state_integrity"]["component_tree_unchanged"] is True
    assert max(oracle["state_integrity"]["exact_v4_row_reproduction"]["heldout"].values()) == 0.0
    assert oracle["state_integrity"]["heldout_baseline_replay"][
        "max_abs_row_ce_difference"
    ] == 0.0
    assert "still calls the missing original MLP" in oracle["caveat"]


def test_authoritative_affine_compiler_failure_gets_zero_recovery_credit():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    compiler = sheet["ledgers"]["early_mlp_affine_compiler_authoritative_failure"]
    assert compiler["stage"] == "completed_authoritative_descriptive_failure_with_erratum"
    assert compiler["authority"] == "isolated_compiler_experiment"
    assert compiler["preregistration_status"] == "preregistered_not_run"
    assert compiler["row_receipt_status"] == "frozen_before_predictor_fit"
    assert compiler["authorized_for_scored_experiments"] is True
    assert compiler["payload_self_authorized"] is False
    assert compiler["fit_was_isolated_and_licensed"] is True
    assert compiler["post_result_authorized_for_training"] is False
    assert compiler["post_result_training_license_sites"] == []
    assert all(compiler["disjointness_gates"].values())
    assert compiler["scored_executable_result_available"] is True
    assert compiler["credited_executable_recovery_nats"] == 0.0
    assert compiler["whole_model_recovery_fraction"] is None
    assert 0.0100 < compiler["arm_gains_nats"]["QNN"] < 0.0102
    assert -0.0983 < compiler["arm_gains_nats"]["NQN"] < -0.0980
    assert -0.0507 < compiler["arm_gains_nats"]["QQN"] < -0.0504
    assert compiler["control_contrasts_nats"]["QQN_beats_mean"] < -0.043
    assert compiler["control_contrasts_nats"]["QQN_beats_shuffle"] < -0.046
    assert 0.16 < compiler["local_validation_r2_centered"]["mlp0"] < 0.17
    assert 0.34 < compiler["local_validation_r2_centered"]["mlp1"] < 0.35
    assert compiler["collateral_worsening_nats"]["copy"] > 0.019
    assert compiler["collateral_worsening_nats"]["novel_freq"] > 0.049
    assert all(value is False for value in compiler["statistical_decisions"].values())
    assert compiler["registered_decisions"]["integrity"] is True
    assert compiler["registered_decisions"]["gauge_replay"] is True
    assert compiler["registered_decisions"]["all_registered_gates"] is False
    assert compiler["preregistration_conformant"] is False
    assert compiler["erratum_status"] == "post_authority_preregistration_erratum"
    assert "literal gauge_replay=true" in compiler["withdrawn_labels"]
    assert len(compiler["deviations"]) == 5
    integrity = compiler["state_integrity"]
    assert integrity["atomic_authority_result_sha256"].startswith("f189cd4f")
    assert integrity["atomic_authority_manifest_sha256"].startswith("8ed7ce44")
    assert integrity["atomic_authority_program_sha256"].startswith("165b656a")
    assert integrity["atomic_authority_program_receipt_sha256"].startswith("9ed63cd0")
    assert integrity["atomic_authority_receipt_sha256"].startswith("2c1ad6ca")
    assert "authoritative descriptive evidence" in compiler["claim"]
    assert "zero executable recovery credit" in compiler["caveat"]
    assert sheet["ledgers"]["early_mlp_mixed_pca_oracle_authoritative"][
        "heldout_projected_upstream_fraction_of_exact"
    ] > 0.56
    assert sheet["ranked_actions"][0]["priority"] == 1
    assert "native bilinear-product" in sheet["ranked_actions"][0]["action"]
    assert "MLP1 response bottleneck" in sheet["ranked_actions"][0]["action"]
    assert "c(z,mo)=p(z)-B^T mo" in sheet["ranked_actions"][0]["action"]
    assert sheet["ranked_actions"][1]["priority"] == 2
    assert "macro factorial" in sheet["ranked_actions"][1]["action"]
    assert "early_mlp_affine_compiler_v1_contract" in sheet["sources"]
    assert "early_mlp_affine_compiler_v1_program" in sheet["sources"]


def test_writer_null_predictors_are_withdrawn_after_disjoint_replication():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    writer = sheet["ledgers"]["tensor_writer_specificity"]
    replication = writer["disjoint_class_replication"]
    assert abs(replication["n_vs_share"]["rho"]) < 0.02
    assert abs(replication["snr_vs_share"]["rho"]) < 0.08
    assert replication["inv_sqrt_n_vs_shuffled"]["rho"] > 0.98
    assert writer["disjoint_class_registered_predictions"]["pred_a_snr_replicates_ge60"] is False
    assert "withdraws cheap null predictors" in writer["caveat"]
