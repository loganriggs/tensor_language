#!/usr/bin/env python3
"""Focused CPU tests for the Task14 subject current/cache value factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial as run


def _exact(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "source_term_sum_max_absolute_error": value,
        "value_branch_sum_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_head_max_absolute_error": value,
        "complete_head_vector_max_absolute_error": value,
    }


def _evidence(mode="current"):
    patterns = {
        "current": (0.8, 0.1, 1.0, 0.05),
        "cached": (0.1, 0.8, 1.0, 0.05),
        "interaction": (0.2, 0.2, 1.0, 0.05),
        "leakage": (0.8, 0.1, 1.0, 0.6),
    }
    current, cached, joint, lexical = patterns[mode]
    effects = {
        "native_value": 0.0,
        "opposite_current_only": current,
        "opposite_cached_only": cached,
        "opposite_both": joint,
        "lexical_current_only": lexical,
        "lexical_cached_only": lexical,
        "lexical_both": lexical,
        "complete_opposite_head": 1.1,
    }
    output = []
    for row in run.build_rows():
        cell = f"{row['direction_id']}__{row['template_id']}"
        sign = 1.0 if row["direction_id"] == "singular_to_plural" else -1.0
        for condition, effect in effects.items():
            output.append({
                "row_id": row["row_id"], "cell_id": cell, "condition": condition,
                "target_margin_improvement": effect,
                "target_CE_improvement": effect,
                "fixed_are_minus_is_change": sign * effect,
            })
    return output


def test_candidate_specific_license_holdout_and_price():
    plan = run.compile_plan()
    assert plan["preflight_validated"] is True
    assert plan["candidate_id"] == run.CANDIDATE_ID
    assert plan["license_sha256"] == run.LICENSE_SHA256
    assert plan["row_count"] == 16 and plan["split"] == "LICENSED_HOLDOUT"
    assert plan["price"] == {"model_forwards": 4, "example_evaluations": 352,
                             "backwards": 0, "parameter_updates": 0}
    assert {row["group_number"] for row in run.build_rows()} == set(range(8, 16))


def test_dry_run_validates_without_model_access():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_wrong_derivative_license_fails_closed(monkeypatch):
    monkeypatch.setattr(run, "LICENSE_SHA256", "0" * 64)
    with pytest.raises(run.licensing.CapabilityLicenseError, match="license hash changed"):
        run.validate_preflight()


def test_projected_branches_sum_to_effective_value():
    torch = pytest.importorskip("torch")

    class Linear:
        def __init__(self, weight):
            self.weight = weight

    class Attention:
        pass

    attention = Attention()
    attention.c_v = Linear(torch.eye(18))
    attention.c_proj = Linear(torch.arange(18 * 18, dtype=torch.float32).reshape(18, 18) / 1000)
    attention.lamb = torch.tensor(0.3)
    state = torch.arange(2 * 4 * 18, dtype=torch.float32).reshape(2, 4, 18) / 100
    bus = torch.flip(state.reshape(2, 4, 9, 2), dims=(1,))
    current, cached = run._projected_value_branches(
        state, bus, attention, torch, torch.nn.functional)
    head_slice = attention.c_proj.weight[:, 6:8]
    expected = torch.nn.functional.linear(
        ((1 - attention.lamb) * state.reshape(2, 4, 9, 2)[:, :, 3]
         + attention.lamb * bus[:, :, 3]), head_slice)
    assert torch.allclose(current + cached, expected, atol=1e-6, rtol=1e-6)


def test_exact_branch_compiler_and_native_reinstall():
    torch = pytest.importorskip("torch")
    rows = run.build_rows(); n, width = len(rows), 5

    def side(offset):
        p = torch.arange(n * 9, dtype=torch.float32).reshape(n, 9) / 100 + 1 + offset
        current = torch.arange(n * 9 * width, dtype=torch.float32).reshape(n, 9, width) / 100 + offset
        cached = torch.ones_like(current) * (0.3 + offset)
        u = current + cached
        return {"p": p, "u": u, "current": current, "cached": cached,
                "head": torch.einsum("bk,bkd->bd", p, u)}

    recipient, opposite, lexical = side(0), side(1), side(2)
    tokens = torch.tensor([row["endpoints"]["recipient"]["ids"] for row in rows])
    patch = run._compile(tokens, recipient, opposite, lexical, rows, torch)
    assert len(patch["specs"]) == 128
    assert int(patch["native_reinstall_mask"].sum()) == 16
    for index, (row_index, condition, _) in enumerate(patch["specs"]):
        assert torch.allclose(patch["replacement_heads"][index],
                              patch["heads"][condition][row_index])
        assert bool(patch["native_reinstall_mask"][index]) == (condition == "native_value")


@pytest.mark.parametrize("mode,held", [
    ("current", "pred_b_current_branch_carries_task"),
    ("cached", "pred_c_cached_branch_carries_task"),
    ("interaction", "pred_d_interaction_is_needed"),
])
def test_opposing_branch_predictions(mode, held):
    predictions = run.score(_evidence(mode), _exact())["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert predictions[held]
    assert predictions["pred_f_number_specific"]
    assert not predictions["pred_e_lexical_leakage"]


def test_lexical_leakage_is_an_opposing_result():
    predictions = run.score(_evidence("leakage"), _exact())["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert predictions["pred_e_lexical_leakage"]
    assert not predictions["pred_f_number_specific"]


def test_interaction_requires_row_level_margin_and_CE_signs():
    evidence = _evidence("interaction")
    counters = {}
    for item in evidence:
        if item["condition"] != "opposite_both":
            continue
        index = counters.get(item["cell_id"], 0)
        counters[item["cell_id"]] = index + 1
        # With both single-branch effects at .2, these joint values give
        # interactions [-.2, -.2, 1.4, 1.4]: positive mean/recovery, but only
        # half the rows have a helpful factorial interaction.
        value = .2 if index < 2 else 1.8
        item["target_margin_improvement"] = value
        item["target_CE_improvement"] = value
    scored = run.score(evidence, _exact())
    assert scored["predictions"]["pred_a_instrument_live"]
    assert not scored["predictions"]["pred_d_interaction_is_needed"]
    cell = scored["cells"]["plural_to_singular__across_beside"]["derived"]
    assert cell["interaction_mean_margin"] == pytest.approx(.6)
    assert sum(value > 0 for value in cell["interaction_margin_values"]) == 2


def test_exactness_and_live_controls_can_fail():
    assert not run.score(_evidence(), _exact(1.0))["predictions"]["pred_a_instrument_live"]
    evidence = _evidence()
    for item in evidence:
        if item["condition"] in {"opposite_both", "complete_opposite_head"}:
            item["target_margin_improvement"] = 0.0
            item["target_CE_improvement"] = 0.0
    assert not run.score(evidence, _exact())["predictions"]["pred_a_instrument_live"]


def test_missing_arm_and_nonfinite_metric_fail_closed():
    with pytest.raises(run.CurrentCachedFactorialError, match="exact licensed factorial"):
        run.score(_evidence()[:-1], _exact())
    evidence = _evidence()
    evidence[0]["target_CE_improvement"] = float("nan")
    with pytest.raises(run.CurrentCachedFactorialError, match="non-finite"):
        run.score(evidence, _exact())
