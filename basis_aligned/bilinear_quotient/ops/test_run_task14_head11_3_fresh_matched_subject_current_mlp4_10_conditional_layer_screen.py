#!/usr/bin/env python3
"""Focused CPU tests for the Task14 MLP4--10 conditional layer screen."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen as run


def _exact(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "state_sum_max_absolute_error": value,
        "normalized_state_max_absolute_error": value,
        "source_term_sum_max_absolute_error": value,
        "full_donor_current_head_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_head_max_absolute_error": value,
        "recipient_high_group_max_absolute_error": value,
        "donor_high_group_max_absolute_error": value,
    }


def _effects():
    values = {condition: 0.05 for condition in run.CONDITIONS}
    values["recipient_M4_10"] = 0.0
    values["opposite_full_M4_10"] = 1.0
    for layer in run.LAYERS:
        values[f"opposite_except_M{layer}"] = .95
    values["opposite_M8"] = .4
    values["opposite_except_M8"] = .6
    return values


def _evidence(effects=None):
    effects = effects or _effects()
    return [{
        "row_id": row["row_id"],
        "cell_id": f"{row['direction_id']}__{row['template_id']}",
        "condition": condition,
        "target_margin_improvement": effect,
        "target_CE_improvement": effect,
    } for row in run.build_rows() for condition, effect in effects.items()]


def test_plan_is_parent_bound_and_minimal():
    plan = run.compile_plan()
    assert plan["row_count"] == 16
    assert plan["layers"] == list(range(4, 11))
    assert len(plan["conditions"]) == 24
    assert plan["price"] == {"model_forwards": 4, "example_evaluations": 864,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["layer_boundary_limit"] == \
        "native MLP layers are localization handles, not final semantic units"


def test_condition_choices_encode_singletons_leave_one_and_controls():
    choices = run._condition_choices()
    assert set(choices) == set(run.CONDITIONS)
    assert choices["recipient_M4_10"] == "rrrrrrr"
    assert choices["opposite_full_M4_10"] == "ooooooo"
    assert choices["opposite_M4"] == "orrrrrr"
    assert choices["opposite_M10"] == "rrrrrro"
    assert choices["opposite_except_M4"] == "roooooo"
    assert choices["opposite_except_M10"] == "oooooor"
    assert choices["lexical_full_M4_10"] == "lllllll"


def test_no_model_dry_run_is_exact_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_preflight_hashes_fail_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.ConditionalLayerScreenError, match="prior-art receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.ConditionalLayerScreenError, match="parent result changed"):
        run.validate_preflight()


def test_high_remainder_follows_M4_choice():
    torch = pytest.importorskip("torch")
    shape = (1, 1, 1)

    def role(offset, remainder):
        return {**{f"M{i}": torch.full(shape, offset + i) for i in run.LAYERS},
                "HR": torch.full(shape, remainder)}

    roles = {"r": role(0, .1), "o": role(10, .2), "l": role(20, .3)}
    recipient = run._high_from_choices(roles, "rrrrrrr")
    donor_M4 = run._high_from_choices(roles, "orrrrrr")
    donor_M5 = run._high_from_choices(roles, "rorrrrr")
    assert torch.allclose(donor_M4 - recipient, torch.tensor([[[10.1]]]))
    assert torch.allclose(donor_M5 - recipient, torch.tensor([[[10.0]]]))


def test_score_finds_same_standalone_and_conditional_layer():
    scored = run.score(_evidence(), _exact())
    assert scored["layer_predictions"]["8"] == {"standalone": True, "conditional": True}
    assert all(not values["standalone"] and not values["conditional"]
               for layer, values in scored["layer_predictions"].items() if layer != "8")
    assert scored["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_at_least_one_standalone_layer": True,
        "pred_c_at_least_one_conditional_layer": True,
        "pred_d_same_layer_is_stable": True,
        "pred_e_context_dependence": False,
        "pred_f_number_specific": True,
        "pred_g_lexical_collateral": False,
    }


def test_context_dependence_uses_full_minus_leave_one():
    effects = _effects()
    effects["opposite_M8"] = .1
    effects["opposite_except_M8"] = .5
    scored = run.score(_evidence(effects), _exact())
    assert not scored["layer_predictions"]["8"]["standalone"]
    assert scored["layer_predictions"]["8"]["conditional"]
    assert scored["predictions"]["pred_e_context_dependence"]


def test_layer_use_requires_margin_and_CE():
    evidence = _evidence()
    for item in evidence:
        if item["condition"] == "opposite_M8":
            item["target_CE_improvement"] = .1
    scored = run.score(evidence, _exact())
    assert not scored["layer_predictions"]["8"]["standalone"]
    assert scored["layer_predictions"]["8"]["conditional"]


def test_new_high_group_closure_gates_instrument():
    for key in ("recipient_high_group_max_absolute_error",
                "donor_high_group_max_absolute_error"):
        exactness = _exact(); exactness[key] = 1.0
        assert not run.score(_evidence(), exactness)["predictions"]["pred_a_instrument_live"]


def test_evidence_shape_and_finiteness_fail_closed():
    with pytest.raises(run.ConditionalLayerScreenError, match="exact licensed screen"):
        run.score(_evidence()[:-1], _exact())
    evidence = _evidence(); evidence[0]["target_margin_improvement"] = float("nan")
    with pytest.raises(run.ConditionalLayerScreenError, match="non-finite"):
        run.score(evidence, _exact())
