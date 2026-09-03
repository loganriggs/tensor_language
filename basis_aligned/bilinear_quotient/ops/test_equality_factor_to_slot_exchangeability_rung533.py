#!/usr/bin/env python3

import math

import torch

import bilin18_observed_model_facade as facade
import equality_factor_to_slot_exchangeability_rung533 as rung533


def test_registered_arm_set_has_four_matched_pairs_and_exact_price():
    assert len(rung533.MAPPINGS) == 4
    assert len(rung533.ARMS) == 11
    assert set(rung533.MAPPINGS).issubset(rung533.ARMS)
    assert set(rung533.CONTROL_BY_MAPPING.values()).issubset(rung533.ARMS)
    assert rung533.FORWARDS_PER_BATCH == 23
    assert rung533.FORWARDS == 2208


def test_replacement_patterns_use_the_requested_source_and_native_companion():
    a = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
    b = torch.tensor([[[11.0, 13.0], [17.0, 19.0]]])
    c = torch.tensor([[[23.0, 29.0], [31.0, 37.0]]])
    d = torch.tensor([[[41.0, 43.0], [47.0, 53.0]]])
    assert torch.equal(rung533.replacement_pattern("native", a, b, c, d), c * d)
    assert torch.equal(rung533.replacement_pattern("absent", a, b, c, d), torch.zeros_like(c))
    assert torch.equal(
        rung533.replacement_pattern("product_control", a, b, c, d),
        rung533.parent.GAMMA * a * b)
    for mapping, scale in rung533.math_contract.SCALES.items():
        source = a if "source_first" in mapping else b
        companion = d if "target_first" in mapping else c
        assert torch.equal(
            rung533.replacement_pattern(mapping, a, b, c, d), scale * source * companion)
        assert not torch.equal(
            rung533.replacement_pattern(mapping, a, b, c, d),
            rung533.replacement_pattern(
                rung533.CONTROL_BY_MAPPING[mapping], a, b, c, d))


def _arm_report(cosine=0.95, error=0.2, recovery=1.0, off=0.001):
    return {
        "positive_document_effect": {"cosine": cosine, "relative_error": error},
        "positive_task_recovery": recovery,
        "matched_negative_abs_mean_ce_change_from_native": off,
        "off_target_abs_mean_ce_change_from_native": off,
    }


def _passing_context():
    arms = {arm: _arm_report() for arm in rung533.ARMS}
    for control in rung533.CONTROL_BY_MAPPING.values():
        arms[control] = _arm_report(cosine=0.20, error=1.0, recovery=0.0, off=0.02)
    return {"arms": arms}


def _diagnostics():
    return {
        "native_replay_logit_max_abs": 0.0,
        "factor_reconstruction_max": 1e-14,
        "branch_product_max_abs": 0.0,
        "minimum_donor_edit_rms": 1.0,
        "minimum_target_edit_rms": 1.0,
        "zero_intended_edits": 0,
        "calls_exact": True,
        "all_cell_supports_live": True,
        "all_positive_document_supports_live": True,
    }


def test_score_accepts_a_complete_four_way_family_and_rejects_one_bad_mapping():
    contexts = [_passing_context() for _ in range(8)]
    stability = {str(index): {"cosine": 0.95} for index in range(16)}
    predictions, checks = rung533.score(
        contexts, stability, _diagnostics(), facade.WEIGHTS_SHA256)
    assert all(predictions.values())
    assert checks["mapping_contexts_passing"] == {
        mapping: 8 for mapping in rung533.MAPPINGS}
    contexts[3]["arms"]["source_first_to_target_first"] = _arm_report(cosine=0.80)
    predictions, checks = rung533.score(
        contexts, stability, _diagnostics(), facade.WEIGHTS_SHA256)
    assert predictions["pred_a_valid_physical_instrument"] is True
    assert predictions["pred_b_product_level_positive_control"] is True
    assert predictions["pred_c_both_source_factors_fill_target_first"] is False
    assert predictions["pred_e_branch_exchangeable_downstream_family"] is False
    assert checks["mapping_contexts_passing"]["source_first_to_target_first"] == 7


def test_validate_inputs_has_support_in_both_halves_and_roles():
    payloads, metadata = rung533.validate_inputs()
    assert set(payloads) == set(rung533.ROLES)
    for role in rung533.ROLES:
        assert metadata[role]["positive_documents"][0] >= 50
        assert metadata[role]["positive_documents"][1] >= 50
        assert min(value for counts in metadata[role]["half_support"].values() for value in counts) > 0
