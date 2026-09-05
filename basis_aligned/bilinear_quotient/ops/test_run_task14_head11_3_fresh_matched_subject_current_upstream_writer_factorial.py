#!/usr/bin/env python3
"""Focused CPU tests for the Task14 L11H3 upstream-writer factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial as run


def _exact(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "state_sum_max_absolute_error": value,
        "normalized_state_max_absolute_error": value,
        "source_term_sum_max_absolute_error": value,
        "all_donor_current_head_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_head_max_absolute_error": value,
    }


def _evidence(mode="embedding"):
    main = {
        "embedding": (.8, .1, .1),
        "attention": (.1, .8, .1),
        "MLP": (.1, .1, .8),
        "distributed": (.4, .35, .25),
        "interaction": (.1, .1, .1),
        "collateral": (.8, .1, .1),
    }[mode]
    e, a, m = main
    all_effect = 1.0
    effects = {
        "recipient_EAM": 0.0,
        "opposite_E": e, "opposite_A": a, "opposite_M": m,
        "opposite_EA": e + a, "opposite_EM": e + m, "opposite_AM": a + m,
        "opposite_EAM": all_effect,
        "lexical_E": .6 if mode == "collateral" else .05,
        "lexical_A": .05, "lexical_M": .05,
        "lexical_EAM": .6 if mode == "collateral" else .05,
    }
    output = []
    for row in run.build_rows():
        for condition, effect in effects.items():
            output.append({
                "row_id": row["row_id"],
                "cell_id": f"{row['direction_id']}__{row['template_id']}",
                "condition": condition,
                "target_margin_improvement": effect,
                "target_CE_improvement": effect,
            })
    return output


def test_candidate_specific_license_holdout_and_price():
    plan = run.compile_plan()
    assert plan["preflight_validated"] is True
    assert plan["candidate_id"] == run.CANDIDATE_ID
    assert plan["split"] == "LICENSED_HOLDOUT" and plan["row_count"] == 16
    assert plan["conditions"] == list(run.CONDITIONS)
    assert plan["price"] == {"model_forwards": 4, "example_evaluations": 480,
                             "backwards": 0, "parameter_updates": 0}


def test_dry_run_validates_without_model_access():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_wrong_receipt_and_license_fail_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.UpstreamWriterFactorialError, match="receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "LICENSE_SHA256", "0" * 64)
    with pytest.raises(run.licensing.CapabilityLicenseError, match="license hash changed"):
        run.validate_preflight()


def test_E_A_M_corner_compiler_uses_exact_named_sources():
    torch = pytest.importorskip("torch")
    rows = run.build_rows(); n, length, width = len(rows), 9, 18
    head_width, output_width = width // 9, 5

    class Linear:
        def __init__(self, weight):
            self.weight = weight

    class Attention:
        pass

    attention = Attention()
    attention.c_v = Linear(torch.eye(width))
    attention.lamb = torch.tensor(.2)
    projection = torch.arange(output_width * head_width, dtype=torch.float32).reshape(
        output_width, head_width) / 10

    def side(offset):
        E = torch.ones(n, length, width) * (1 + offset)
        A = torch.ones(n, length, width) * (2 + offset)
        M = torch.ones(n, length, width) * (3 + offset)
        current = run._current_from_state(E, A, M, attention, torch, torch.nn.functional)
        cached = torch.ones_like(current) * .3
        u = torch.nn.functional.linear(current + cached, projection)
        p = torch.ones(n, length) + offset / 10
        return {"E": E, "A": A, "M": M, "p": p, "u": u,
                "current_pre": current, "cached_pre": cached,
                "head": torch.einsum("bk,bkd->bd", p, u)}

    recipient, opposite, lexical = side(0), side(1), side(2)
    tokens = torch.tensor([row["endpoints"]["recipient"]["ids"] for row in rows])
    patch = run._compile(tokens, recipient, opposite, lexical, attention, projection,
                         rows, torch, torch.nn.functional)
    assert len(patch["specs"]) == 16 * len(run.CONDITIONS)
    assert int(patch["native_reinstall_mask"].sum()) == 16
    expected_EA = run._current_from_state(
        opposite["E"], opposite["A"], recipient["M"],
        attention, torch, torch.nn.functional)
    assert torch.equal(patch["current"]["opposite_EA"], expected_EA)
    expected_lexical_M = run._current_from_state(
        recipient["E"], recipient["A"], lexical["M"],
        attention, torch, torch.nn.functional)
    assert torch.equal(patch["current"]["lexical_M"], expected_lexical_M)


@pytest.mark.parametrize("mode,key", [
    ("embedding", "pred_b_embedding_carries_task"),
    ("attention", "pred_c_attention_carries_task"),
    ("MLP", "pred_d_MLP_carries_task"),
    ("distributed", "pred_e_distributed_across_writer_families"),
    ("interaction", "pred_f_interaction_is_needed"),
])
def test_frozen_opposing_writer_predictions(mode, key):
    predictions = run.score(_evidence(mode), _exact())["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert predictions[key]
    assert predictions["pred_g_number_specific"]
    assert not predictions["pred_h_lexical_collateral"]


def test_lexical_collateral_invalidates_number_specificity():
    predictions = run.score(_evidence("collateral"), _exact())["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert predictions["pred_h_lexical_collateral"]
    assert not predictions["pred_g_number_specific"]


def test_exactness_and_all_donor_task_controls_can_fail():
    assert not run.score(_evidence(), _exact(1.0))["predictions"]["pred_a_instrument_live"]
    evidence = _evidence()
    for item in evidence:
        if item["condition"] == "opposite_EAM":
            item["target_margin_improvement"] = 0.0
            item["target_CE_improvement"] = 0.0
    assert not run.score(evidence, _exact())["predictions"]["pred_a_instrument_live"]


def test_missing_arm_and_nonfinite_metric_fail_closed():
    with pytest.raises(run.UpstreamWriterFactorialError, match="exact licensed factorial"):
        run.score(_evidence()[:-1], _exact())
    evidence = _evidence()
    evidence[0]["target_CE_improvement"] = float("nan")
    with pytest.raises(run.UpstreamWriterFactorialError, match="non-finite"):
        run.score(evidence, _exact())
