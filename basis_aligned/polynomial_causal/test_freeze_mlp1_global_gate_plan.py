from __future__ import annotations

import json

import freeze_mlp1_global_gate_plan as freezer


def test_plan_is_deterministic_and_binds_admitted_parents() -> None:
    first, second = freezer.build_plan(), freezer.build_plan()
    assert first == second
    assert first["status"] == "frozen_cpu_plan_no_gpu_authority"
    assert first["plan_fingerprint"] == freezer.canonical_sha256({
        key: value for key, value in first.items() if key != "plan_fingerprint"
    })
    assert first["parents"]["hashes"] == {
        "rows_receipt": freezer.EXPECTED_ROWS_RECEIPT_SHA256,
        "rows_file": freezer.EXPECTED_ROWS_FILE_SHA256,
        "row_use_authority": freezer.EXPECTED_ROW_USE_AUTHORITY_SHA256,
        "rank640_predictive": freezer.EXPECTED_RANK640_PREDICTIVE_SHA256,
        "rank640_causal": freezer.EXPECTED_RANK640_CAUSAL_SHA256,
        "program_authority": freezer.EXPECTED_PROGRAM_AUTHORITY_SHA256,
    }


def test_fit_and_validation_are_document_and_row_disjoint() -> None:
    plan = freezer.build_plan()
    fit, validation = plan["cohorts"]["fit"], plan["cohorts"]["validation"]
    assert fit["contexts"] == validation["contexts"] == 16
    assert fit["model_input_shape"] == validation["model_input_shape"] == [16, 256]
    assert len(set(fit["document_ids"])) == len(set(validation["document_ids"])) == 16
    assert not set(fit["document_ids"]) & set(validation["document_ids"])
    assert not set(fit["row_indices"]) & set(validation["row_indices"])
    assert fit["row_indices"] == list(range(16))
    assert validation["row_indices"] == list(range(16, 32))
    assert plan["cohorts"]["registry_wide_disjoint_from_every_prior_role"] is True
    assert plan["cohorts"]["validation_is_fresh_fineweb_not_cross_corpus_ood"] is True


def test_probe_halves_are_new_disjoint_and_exactly_price_the_run() -> None:
    plan = freezer.build_plan()
    first = plan["probe_halves"]["first"]["probe_seeds"]
    second = plan["probe_halves"]["second"]["probe_seeds"]
    assert len(first) == len(second) == 32
    assert not set(first) & set(second)
    assert min(first) > 2026083032
    assert plan["operator"]["backward_passes_at_batch4"] == 512


def test_candidate_bundle_and_promotion_rules_are_outcome_blind_and_complete() -> None:
    plan = freezer.build_plan()
    selectors, decision = plan["selectors"], plan["decision"]
    assert selectors["budgets"] == [32, 128, 512]
    assert selectors["target_rank_by_budget"] == {"32": 16, "128": 64, "512": 256}
    assert "fit documents/probe-half first only" in selectors["candidate_bundle"]
    assert set(selectors["controls"]) == {
        "response_energy", "activation_down", "factor_product_derangement", "hash_random",
    }
    assert decision["support_jaccard_minimum"] == 0.5
    assert decision["css_and_all_on_must_both_pass"] is True
    assert decision["both_validation_probe_halves_must_pass"] is True
    assert decision["consequence_stage_authorized"] is False
    assert plan["finite_followup_if_promoted"]["authorized_by_this_plan"] is False
    assert plan["linear_solver"]["precision"] == "CPU torch.float64"
    assert plan["linear_solver"]["same_solver_for_primary_and_every_control"] is True
    assert plan["metrics"]["bootstrap"]["simultaneous_family_size"] == 48
    assert plan["metrics"]["bootstrap"]["repetitions"] == 20_000
    assert plan["metrics"]["bootstrap"]["method"].endswith("max-error bootstrap")
    assert "Spearman" in plan["metrics"]["score_rank_stability"]
    assert selectors["factor_product_derangement_shift"].startswith("+1")
    assert selectors["factor_product_derangement_scale_sign_gauge_invariant"] is True
    assert decision["best_control_selection"].startswith("none")


def test_serialized_plan_equals_builder() -> None:
    assert json.loads(freezer.OUT.read_text()) == freezer.build_plan()
