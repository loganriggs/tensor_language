#!/usr/bin/env python3
"""Focused CPU tests for the Task14 within-MLP8 polarized response screen."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial as run


def _exact(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "state_sum_max_absolute_error": value,
        "normalized_state_max_absolute_error": value,
        "source_term_sum_max_absolute_error": value,
        "product_closure_max_absolute_error": value,
        "output_closure_max_absolute_error": value,
        "propagated_recipient_MLP8_max_absolute_error": value,
        "propagated_source_MLP8_max_absolute_error": value,
        "gauge_invariance_max_absolute_error": value,
        "parent_head_endpoint_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_head_max_absolute_error": value,
    }


def _condition_values(cross=.8, quadratic=.1, full=1.0, lexical=.02):
    values = {}
    for background, base in (("standalone", 0.0), ("conditional", .2)):
        values[f"{background}_recipient"] = base
        values[f"{background}_cross"] = base + cross
        values[f"{background}_quadratic"] = base + quadratic
        values[f"{background}_full"] = base + full
    values.update({
        "lexical_recipient": 0.0,
        "lexical_cross": lexical / 2,
        "lexical_quadratic": lexical / 2,
        "lexical_full": lexical,
    })
    return values


def _evidence(values=None):
    values = values or _condition_values()
    return [{
        "row_id": row["row_id"],
        "cell_id": f"{row['direction_id']}__{row['template_id']}",
        "condition": condition,
        "target_margin_improvement": effect,
        "target_CE_improvement": effect,
    } for row in run.build_rows() for condition, effect in values.items()]


def test_plan_is_parent_bound_subject_position_and_minimal():
    plan = run.compile_plan()
    assert plan["row_count"] == 16
    assert plan["subject_position"] == 8
    assert plan["mlp_layer"] == 8
    assert len(plan["conditions"]) == 12
    assert all("left" not in condition.lower() and "right" not in condition.lower()
               for condition in plan["conditions"])
    assert plan["price"] == {
        "model_forwards": 4, "example_evaluations": 480,
        "causal_interventions": 192, "backwards": 0, "parameter_updates": 0,
    }


def test_no_model_dry_run_is_exact_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_preflight_hashes_fail_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.MLP8PolarizedResponseError, match="prior-art receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8PolarizedResponseError, match="parent result changed"):
        run.validate_preflight()


def test_exact_polarization_closes_and_is_gauge_invariant():
    torch = pytest.importorskip("torch")

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Left = torch.nn.Linear(5, 7, bias=False)
            self.Right = torch.nn.Linear(5, 7, bias=False)
            self.Down = torch.nn.Linear(7, 5, bias=False)
            self.Down_bias = torch.nn.Parameter(torch.randn(5))

    torch.manual_seed(8)
    mlp = MLP()
    recipient = torch.randn(3, 9, 5)
    source = torch.randn(3, 9, 5)
    products, diagnostics = run._polarized_products(mlp, recipient, source, torch)
    assert torch.allclose(products["full"], mlp.Left(source) * mlp.Right(source))
    assert torch.allclose(
        products["full"], products["cross"] + products["quadratic"]
        - products["recipient"], atol=2e-6, rtol=2e-6)
    assert diagnostics["product_closure_max_absolute_error"] < 2e-6
    assert diagnostics["gauge_invariance_max_absolute_error"] < 2e-6


def test_affine_encoder_is_rejected_before_dx_is_evaluated():
    torch = pytest.importorskip("torch")

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Left = torch.nn.Linear(3, 4, bias=True)
            self.Right = torch.nn.Linear(3, 4, bias=False)
            self.Down = torch.nn.Linear(4, 3, bias=False)
            self.Down_bias = torch.nn.Parameter(torch.zeros(3))

    x = torch.randn(1, 9, 3)
    with pytest.raises(run.MLP8PolarizedResponseError, match="bias-free"):
        run._polarized_products(MLP(), x, x + 1, torch)


def test_propagated_slot_changes_only_subject_and_bias_is_once():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Down = torch.nn.Linear(4, 3, bias=False)
            self.Down_bias = torch.nn.Parameter(torch.randn(3))

    torch.manual_seed(9)
    mlp = MLP()
    products = {name: torch.randn(2, 9, 4) for name in run.COMPONENTS}
    recipient_slot = torch.randn(2, 9, 3)
    slots, outputs = run._propagated_slots(
        mlp, products, recipient_slot, torch.tensor(.25), F)
    for name in run.COMPONENTS:
        assert torch.equal(slots[name][:, :8], recipient_slot[:, :8])
        assert torch.allclose(slots[name][:, 8], .25 * outputs[name][:, 8])
        expected = F.linear(products[name], mlp.Down.weight) + mlp.Down_bias
        assert torch.allclose(outputs[name], expected)


def test_score_selects_cross_dominance_and_number_specificity():
    scored = run.score(_evidence(), _exact())
    assert scored["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_cross_dominant": True,
        "pred_c_quadratic_dominant": False,
        "pred_d_distributed": False,
        "pred_e_downstream_interaction_needed": False,
        "pred_f_background_stable": True,
        "pred_g_number_specific": True,
        "pred_h_lexical_collateral": False,
    }


def test_score_can_select_quadratic_or_distributed_interaction():
    quadratic = run.score(_evidence(_condition_values(cross=.1, quadratic=.8)), _exact())
    assert quadratic["predictions"]["pred_c_quadratic_dominant"]
    assert not quadratic["predictions"]["pred_b_cross_dominant"]
    distributed = run.score(
        _evidence(_condition_values(cross=.3, quadratic=.3, full=1.0)), _exact())
    assert distributed["predictions"]["pred_d_distributed"]
    assert distributed["predictions"]["pred_e_downstream_interaction_needed"]


def test_background_difference_and_lexical_controls_are_live():
    values = _condition_values()
    values["conditional_cross"] = .2 + .3
    unstable = run.score(_evidence(values), _exact())
    assert not unstable["predictions"]["pred_f_background_stable"]
    lexical = run.score(_evidence(_condition_values(lexical=.6)), _exact())
    assert not lexical["predictions"]["pred_g_number_specific"]
    assert lexical["predictions"]["pred_h_lexical_collateral"]


def test_recipient_corner_uses_same_opposite_answer_orientation_within_task_factorial():
    torch = pytest.importorskip("torch")
    row = run.build_rows()[0]
    opposite = row["endpoints"]["opposite_same_lemma"]
    recipient = row["endpoints"]["recipient"]
    size = max(int(opposite["answer_id"]), int(opposite["foil_id"]),
               int(recipient["answer_id"]), int(recipient["foil_id"])) + 1
    logits = torch.zeros(size)
    logits[int(opposite["answer_id"])] = 3.0
    task_base = run._metrics(logits, row, "standalone_recipient", torch)
    lexical_base = run._metrics(logits, row, "lexical_recipient", torch)
    assert task_base["target_margin"] == pytest.approx(3.0)
    assert lexical_base["target_margin"] == pytest.approx(-3.0)


def test_every_exactness_gate_can_invalidate_instrument():
    for key in _exact():
        exactness = _exact(); exactness[key] = 1.0
        assert not run.score(_evidence(), exactness)["predictions"]["pred_a_instrument_live"]


def test_evidence_shape_and_finiteness_fail_closed():
    with pytest.raises(run.MLP8PolarizedResponseError, match="exact licensed screen"):
        run.score(_evidence()[:-1], _exact())
    evidence = _evidence(); evidence[0]["target_CE_improvement"] = float("nan")
    with pytest.raises(run.MLP8PolarizedResponseError, match="non-finite"):
        run.score(evidence, _exact())
