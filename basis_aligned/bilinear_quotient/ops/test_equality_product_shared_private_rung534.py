#!/usr/bin/env python3

import torch

import bilin18_observed_model_facade as facade
import equality_product_shared_private_rung534 as rung534


def test_registered_arm_set_and_exact_price():
    assert rung534.ARMS == (
        "native", "absent", "shared", "private", "shared_key_control",
        "private_key_control", "private_sign_control")
    assert rung534.FORWARDS_PER_BATCH == 15
    assert rung534.FORWARDS == 1440


def test_score_patterns_recompose_and_controls_are_nontrivial():
    generator = torch.Generator().manual_seed(12)
    tensors = [torch.randn(2, 9, 9, generator=generator) for _ in range(4)]
    native, shared, private = rung534.split_patterns(*tensors)
    assert torch.allclose(native, shared + private, atol=2e-6, rtol=0)
    assert torch.equal(rung534.replacement_pattern("native", *tensors), native)
    assert torch.equal(rung534.replacement_pattern("shared", *tensors), shared)
    assert torch.equal(rung534.replacement_pattern("private", *tensors), private)
    assert torch.equal(rung534.replacement_pattern("private_sign_control", *tensors), -private)
    assert not torch.equal(rung534.replacement_pattern("shared_key_control", *tensors), shared)
    assert not torch.equal(rung534.replacement_pattern("private_key_control", *tensors), private)


def _metric(cosine=0.95, error=0.2):
    return {"cosine": cosine, "relative_error": error}


def _context():
    cells = {}
    for cell in rung534.TASK_CELLS:
        cells[cell] = {
            "arms_vs_native_effect": {
                arm: _metric() for arm in rung534.ARMS},
            "private_vs_marginal": {
                "private": _metric(),
                "private_key_control": _metric(cosine=0.1, error=1.0),
                "private_sign_control": _metric(cosine=-0.9, error=2.0),
            },
        }
    cells["positive"]["arms_vs_native_effect"]["shared_key_control"] = _metric(
        cosine=0.1, error=1.0)
    return {
        "cells": cells,
        "positive_recovery": {arm: 1.0 for arm in rung534.ARMS},
        "shared_matched_negative_abs_mean_ce_change_from_native": 0.025,
    }


def _diagnostics():
    return {
        "native_replay_logit_max_abs": 0.0,
        "factor_reconstruction_max": 1e-14,
        "branch_product_max_abs": 0.0,
        "score_recomposition_max_abs": 0.0,
        "minimum_donor_edit_rms": 1.0,
        "minimum_target_edit_rms": 1.0,
        "zero_intended_edits": 0,
        "calls_exact": True,
        "all_document_supports_live": True,
    }


def test_score_accepts_autonomous_private_correction_and_catches_interaction_only():
    reports = {
        f"{role}/{background}/half{half}": _context()
        for role in rung534.ROLES for background in rung534.BACKGROUNDS for half in range(2)
    }
    predictions, checks = rung534.score(
        reports, list(reports.values()), _diagnostics(), facade.WEIGHTS_SHA256)
    assert all(predictions.values())
    assert checks["private_autonomy_code_absent_halves_passing"] == 2
    for half in range(2):
        reports[f"ood_code/donor_absent/half{half}"]["cells"]["matched_negative"] \
            ["private_vs_marginal"]["private"] = _metric(cosine=0.2, error=1.0)
    predictions, checks = rung534.score(
        reports, list(reports.values()), _diagnostics(), facade.WEIGHTS_SHA256)
    assert predictions["pred_a_exact_live_instrument"] is True
    assert predictions["pred_b_shared_signal_premise_reproduces"] is True
    assert predictions["pred_c_private_correction_autonomous_on_code"] is False
    assert checks["private_autonomy_code_absent_halves_passing"] == 0


def test_validate_inputs_reuses_only_frozen_parent_rows():
    payloads, metadata = rung534.validate_inputs()
    assert set(payloads) == set(rung534.ROLES)
    assert metadata["rung533_result_sha256"] == rung534.HASHES[rung534.PARENT_RESULT]
