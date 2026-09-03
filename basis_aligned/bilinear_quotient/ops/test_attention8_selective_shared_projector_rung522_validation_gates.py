"""Focused tests for corrected provisional rung-522 VALIDATION decisions."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))
PATH = OPS / "attention8_selective_shared_projector_rung522_validation_gates.py"
SPEC = importlib.util.spec_from_file_location("rung522_validation_gates", PATH)
GATES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATES
SPEC.loader.exec_module(GATES)


FOLDS = ("omit202", "omit211", "omit221")
SEEDS = tuple(range(52_200, 52_205))
CELLS = GATES.VALIDATION_CELLS


def _cell(*, member=0.5, control=0.05, cosine=0.90, recovery=0.60):
    return {
        "signed_cosine": cosine,
        "relative_residual": 0.20,
        "aligned_recovery": recovery,
        "member_rms": member,
        "control_rms": control,
        "full_attention8_concentration": 5.0,
        "fourfold_margin_lower95": 0.10,
        "bounded_selectivity_improvement_lower95": 0.05,
        "exact_token_tier0_or1": None,
    }


def _family(cell_factory=_cell):
    return {
        fold: {
            seed: {
                "healthy": True,
                "cells": {cell: dict(cell_factory()) for cell in CELLS},
            }
            for seed in SEEDS
        }
        for fold in FOLDS
    }


def _inputs():
    real = _family()
    recovery = _family(lambda: _cell(member=0.30, control=0.06, cosine=0.92, recovery=0.55))
    oracle = _family(lambda: _cell(member=0.50, control=0.05, cosine=0.90, recovery=0.80))
    basis = torch.eye(8, dtype=torch.float64)
    real_frames = {
        seed: {fold: basis[:, :2] for fold in FOLDS}
        for seed in SEEDS
    }
    null_frames = {
        seed: {
            FOLDS[0]: basis[:, :2],
            FOLDS[1]: basis[:, 2:4],
            FOLDS[2]: basis[:, 4:6],
        }
        for seed in range(52_300, 52_316)
    }
    return {
        "real": real,
        "recovery_only": recovery,
        "oracles": oracle,
        "haar_joint": {fold: [0.10] * 20 for fold in FOLDS},
        "label_null_joint": {fold: [0.20] * 16 for fold in FOLDS},
        "real_frames": real_frames,
        "label_null_frames": null_frames,
    }


def test_all_corrected_validation_a_and_b_clauses_pass_on_planted_metrics():
    result = GATES.evaluate_provisional_validation_gates(**_inputs())
    assert result.oracle_liveness_passes
    assert result.real_fit_health_passes
    assert result.recovery_only_fit_health_passes
    assert result.prediction_a_passes
    assert result.matched_stability.passes_four_of_five
    assert all(fold.passing_seed_count == 5 for fold in result.prediction_a_folds)
    assert result.prediction_b_clauses_pass_without_a
    assert result.prediction_b_passes
    for fold in result.prediction_b_folds:
        assert fold.recovery_comparison.passing_seed_count == 5
        assert fold.recovery_comparison.sign_flip.strictly_exceeds_q95
        assert fold.joint_comparison.passing_seed_count == 5


def test_same_seed_oracle_half_recovery_is_not_replaced_by_average_or_best_oracle():
    inputs = _inputs()
    fold = FOLDS[0]
    for seed in SEEDS[:2]:
        for cell in CELLS:
            inputs["oracles"][fold][seed]["cells"][cell]["aligned_recovery"] = 1.40
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    scored = result.prediction_a_folds[0]
    assert scored.passing_seed_count == 3
    assert not scored.passes_four_of_five
    assert "recovery_below_half_same_seed_oracle" in scored.seeds[0].cells[0].failures
    assert not result.prediction_a_passes


def test_oracle_liveness_is_global_and_cannot_make_half_oracle_gate_easy():
    inputs = _inputs()
    inputs["oracles"][FOLDS[1]][SEEDS[0]]["cells"][CELLS[2]]["aligned_recovery"] = 0.049
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    assert not result.oracle_liveness_passes
    assert not result.prediction_a_passes
    assert "oracle_recovery_below_0.05" in result.prediction_a_folds[1].seeds[0].cells[2].failures


def test_a_requires_four_of_five_seeds_in_every_omitted_fold_and_stability():
    inputs = _inputs()
    for seed in SEEDS[:2]:
        inputs["real"][FOLDS[2]][seed]["cells"][CELLS[0]]["fourfold_margin_lower95"] = 0
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    assert result.prediction_a_folds[2].passing_seed_count == 3
    assert not result.prediction_a_passes

    inputs = _inputs()
    basis = torch.eye(8, dtype=torch.float64)
    inputs["label_null_frames"] = {
        seed: {fold: basis[:, :2] for fold in FOLDS}
        for seed in range(52_300, 52_316)
    }
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    assert not result.matched_stability.passes_four_of_five
    assert not result.prediction_a_passes


def test_powered_exact_token_control_is_part_of_each_a_cell_gate():
    inputs = _inputs()
    for seed in SEEDS[:2]:
        inputs["real"][FOLDS[0]][seed]["cells"][CELLS[0]][
            "exact_token_tier0_or1"
        ] = {"pair_count": 40, "passes": False}
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    fold = result.prediction_a_folds[0]
    assert fold.passing_seed_count == 3
    assert "powered_exact_token_specificity_failed" in fold.seeds[0].cells[0].failures
    assert not result.prediction_a_passes


def test_any_unhealthy_required_real_or_recovery_fit_fails_the_global_precondition():
    inputs = _inputs()
    inputs["real"][FOLDS[0]][SEEDS[0]]["healthy"] = False
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    assert result.prediction_a_folds[0].passing_seed_count == 4
    assert not result.real_fit_health_passes
    assert not result.prediction_a_passes
    assert not result.prediction_b_clauses_pass_without_a

    inputs = _inputs()
    inputs["recovery_only"][FOLDS[2]][SEEDS[4]]["healthy"] = False
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    assert not result.recovery_only_fit_health_passes
    assert not result.prediction_b_clauses_pass_without_a


def test_recovery_comparison_uses_min_concentration_every_cell_cosine_and_bootstrap():
    inputs = _inputs()
    fold = FOLDS[0]
    # Two seeds fail three different clauses; the fold has only 3/5 passes.
    inputs["real"][fold][SEEDS[0]]["cells"][CELLS[0]]["member_rms"] = 0.315
    inputs["real"][fold][SEEDS[0]]["cells"][CELLS[0]]["control_rms"] = 0.06
    inputs["recovery_only"][fold][SEEDS[1]]["cells"][CELLS[1]]["signed_cosine"] = 0.951
    inputs["real"][fold][SEEDS[1]]["cells"][CELLS[2]][
        "bounded_selectivity_improvement_lower95"
    ] = 0
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    scored = result.prediction_b_folds[0].recovery_comparison
    assert scored.passing_seed_count == 3
    assert "minimum_concentration_improvement_below_0.5" in scored.seeds[0].failures
    assert "signed_cosine_loss_above_0.05" in scored.seeds[1].failures
    assert "cell_bootstrap_selectivity_improvement_not_positive" in scored.seeds[1].failures
    assert not scored.passes


def test_recovery_fold_requires_exact_five_seed_sign_flip_beyond_four_individual_passes():
    inputs = _inputs()
    fold = FOLDS[1]
    # Four values just meet 0.5 and one fails; their exact sign-flip q95 equals
    # the observed mean because the zero/tied values prevent strict separation.
    for seed in SEEDS[:4]:
        for cell in CELLS:
            inputs["real"][fold][seed]["cells"][cell]["member_rms"] = 0.33
            inputs["real"][fold][seed]["cells"][cell]["control_rms"] = 0.06
    for cell in CELLS:
        inputs["real"][fold][SEEDS[4]]["cells"][cell]["member_rms"] = 0.30
        inputs["real"][fold][SEEDS[4]]["cells"][cell]["control_rms"] = 0.06
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    scored = result.prediction_b_folds[1].recovery_comparison
    assert scored.passing_seed_count == 4
    recomputed = GATES.protocol.exact_five_pair_sign_flip_null(
        [seed.minimum_concentration_improvement for seed in scored.seeds]
    )
    assert scored.sign_flip == recomputed
    assert len(scored.sign_flip.null_means) == 32
    assert not scored.sign_flip.strictly_exceeds_q95
    assert not scored.passes


def test_joint_b_is_bounded_and_requires_four_real_seeds_strictly_above_both_controls():
    inputs = _inputs()
    fold = FOLDS[2]
    # Raising the null above two real seeds makes the fold exactly 3/5.
    baseline = GATES.evaluate_provisional_validation_gates(**inputs)
    real_joint = baseline.prediction_b_folds[2].joint_comparison.seeds[0].real_joint_statistic
    for index in range(16):
        inputs["label_null_joint"][fold][index] = real_joint
    result = GATES.evaluate_provisional_validation_gates(**inputs)
    joint = result.prediction_b_folds[2].joint_comparison
    assert joint.passing_seed_count == 0
    assert not joint.passes_four_of_five
    assert not result.prediction_b_passes
    assert all(seed.real_joint_statistic == pytest.approx(real_joint) for seed in joint.seeds)
    assert all(not seed.strictly_beats_both for seed in joint.seeds)  # equality is not enough


def test_validation_aggregator_fails_closed_on_missing_cells_seeds_and_controls():
    inputs = _inputs()
    del inputs["real"][FOLDS[0]][SEEDS[0]]["cells"][CELLS[0]]
    with pytest.raises(ValueError, match="exactly the four"):
        GATES.evaluate_provisional_validation_gates(**inputs)
    inputs = _inputs()
    inputs["haar_joint"][FOLDS[0]] = [0.1] * 19
    with pytest.raises(ValueError, match="exactly 20"):
        GATES.evaluate_provisional_validation_gates(**inputs)


def test_module_has_no_model_data_or_cuda_imports():
    tree = ast.parse(PATH.read_text())
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported_roots <= {
        "__future__", "dataclasses", "math", "operator", "typing",
        "attention8_selective_shared_projector_rung522_protocol",
    }
