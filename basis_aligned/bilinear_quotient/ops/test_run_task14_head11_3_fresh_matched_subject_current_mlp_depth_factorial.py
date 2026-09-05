#!/usr/bin/env python3
"""Focused CPU tests for the Task14 upstream MLP depth factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial as run


def _exact(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "state_sum_max_absolute_error": value,
        "normalized_state_max_absolute_error": value,
        "source_term_sum_max_absolute_error": value,
        "all_donor_current_head_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_head_max_absolute_error": value,
        "recipient_grouped_M_max_absolute_error": value,
        "all_donor_grouped_M_max_absolute_error": value,
    }


def _evidence(effects=None):
    effects = effects or {
        "recipient_G012": 0.0,
        "opposite_G0": .1, "opposite_G1": .1, "opposite_G2": .8,
        "opposite_G01": .2, "opposite_G02": .9, "opposite_G12": .9,
        "opposite_G012": 1.0,
        "lexical_G0": .05, "lexical_G1": .05, "lexical_G2": .05,
        "lexical_G012": .05,
    }
    return [{
        "row_id": row["row_id"],
        "cell_id": f"{row['direction_id']}__{row['template_id']}",
        "condition": condition,
        "target_margin_improvement": effect,
        "target_CE_improvement": effect,
    } for row in run.build_rows() for condition, effect in effects.items()]


def test_plan_is_balanced_parent_bound_and_small():
    plan = run.compile_plan()
    assert plan["row_count"] == 16
    assert plan["groups"] == {"G0": [0, 1, 2, 3],
                              "G1": [4, 5, 6, 7], "G2": [8, 9, 10]}
    assert plan["conditions"] == list(run.CONDITIONS)
    assert len(plan["conditions"]) == 12
    assert plan["price"] == {"model_forwards": 4, "example_evaluations": 480,
                             "backwards": 0, "parameter_updates": 0}


def test_no_model_dry_run_is_exact_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_preflight_hashes_fail_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.MLPDepthFactorialError, match="prior-art receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLPDepthFactorialError, match="parent result changed"):
        run.validate_preflight()


def test_group_remainder_and_order_reproduce_total():
    torch = pytest.importorskip("torch")
    g0 = torch.tensor([10000.0], dtype=torch.float32)
    g1 = torch.tensor([-10000.0], dtype=torch.float32)
    g2 = torch.tensor([0.00146484375], dtype=torch.float32)
    native_m = torch.tensor([0.0], dtype=torch.float32)
    grouped = (g0 + g1) + g2
    assert not torch.equal(grouped, native_m)
    remainder = native_m - grouped
    assert torch.equal(grouped + remainder, native_m)


def test_current_uses_group_remainder_and_parent_state_remainder():
    torch = pytest.importorskip("torch")

    class Linear:
        def __init__(self, weight):
            self.weight = weight

    class Attention:
        pass

    attention = Attention()
    attention.c_v = Linear(torch.eye(18))
    attention.lamb = torch.tensor(.2)
    zero = torch.zeros(1, 1, 18)
    E = torch.arange(18, dtype=torch.float32).reshape(1, 1, 18)
    G0, G1, G2 = zero + 1, zero + 2, zero + 3
    MR, R = zero.clone(), zero.clone()
    MR[..., 6] = 4
    R[..., 7] = 5
    observed = run._current_from_groups(
        E, zero, R, G0, G1, G2, MR, attention, torch, torch.nn.functional)
    state = (E + zero + (((G0 + G1) + G2) + MR)) + R
    expected = .8 * torch.nn.functional.rms_norm(state, (18,)).reshape(
        1, 1, 9, 2)[:, :, 3]
    assert torch.allclose(observed, expected)


def test_score_identifies_only_G2_and_number_specificity():
    predictions = run.score(_evidence(), _exact())["predictions"]
    assert predictions == {
        "pred_a_instrument_live": True,
        "pred_b_G0_carries_task": False,
        "pred_c_G1_carries_task": False,
        "pred_d_G2_carries_task": True,
        "pred_e_distributed_across_depth_groups": False,
        "pred_f_interaction_is_needed": False,
        "pred_g_number_specific": True,
        "pred_h_lexical_collateral": False,
    }


def test_score_requires_both_margin_and_CE_recovery():
    evidence = _evidence()
    for item in evidence:
        if item["condition"] == "opposite_G2":
            item["target_CE_improvement"] = .1
    assert not run.score(evidence, _exact())["predictions"]["pred_d_G2_carries_task"]


def test_each_new_closure_gate_can_invalidate_instrument():
    for key in ("recipient_grouped_M_max_absolute_error",
                "all_donor_grouped_M_max_absolute_error"):
        exactness = _exact()
        exactness[key] = 1.0
        assert not run.score(_evidence(), exactness)["predictions"]["pred_a_instrument_live"]


def test_evidence_shape_and_finiteness_fail_closed():
    with pytest.raises(run.MLPDepthFactorialError, match="exact licensed factorial"):
        run.score(_evidence()[:-1], _exact())
    evidence = _evidence()
    evidence[0]["target_CE_improvement"] = float("nan")
    with pytest.raises(run.MLPDepthFactorialError, match="non-finite"):
        run.score(evidence, _exact())
