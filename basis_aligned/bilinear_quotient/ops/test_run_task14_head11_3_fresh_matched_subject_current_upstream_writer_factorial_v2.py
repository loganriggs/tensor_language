#!/usr/bin/env python3
"""Focused CPU tests for the Task14 E/A/M float-remainder repair."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2 as run


def _exact(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "uncorrected_state_max_absolute_error": .00146484375,
        "state_sum_max_absolute_error": value,
        "normalized_state_max_absolute_error": value,
        "source_term_sum_max_absolute_error": value,
        "all_donor_current_head_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_head_max_absolute_error": value,
    }


def _evidence():
    effects = {
        "recipient_EAM": 0.0,
        "opposite_E": .8, "opposite_A": .1, "opposite_M": .1,
        "opposite_EA": .9, "opposite_EM": .9, "opposite_AM": .2,
        "opposite_EAM": 1.0,
        "lexical_E": .05, "lexical_A": .05, "lexical_M": .05, "lexical_EAM": .05,
    }
    return [{
        "row_id": row["row_id"],
        "cell_id": f"{row['direction_id']}__{row['template_id']}",
        "condition": condition,
        "target_margin_improvement": effect,
        "target_CE_improvement": effect,
    } for row in run.build_rows() for condition, effect in effects.items()]


def test_v2_changes_only_remainder_accounting_and_candidate_binding():
    plan = run.compile_plan()
    assert plan["numerical_repair_only"] is True
    assert plan["candidate_id"].endswith("factorial_v2")
    assert plan["conditions"] == list(run.v1.CONDITIONS)
    assert plan["bars"] == run.v1.BARS
    assert plan["price"] == {"model_forwards": 4, "example_evaluations": 480,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["outcomes"] == ["answer_directed_target_margin_improvement",
                                 "target_full_vocab_CE_improvement"]


def test_dry_run_validates_without_model_access():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_wrong_repair_receipt_and_license_fail_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.UpstreamWriterFactorialV2Error, match="repair receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "LICENSE_SHA256", "0" * 64)
    with pytest.raises(run.licensing.CapabilityLicenseError, match="license hash changed"):
        run.validate_preflight()


def test_explicit_remainder_repairs_a_real_float32_regrouping_drift():
    torch = pytest.importorskip("torch")
    # This is the same failure class as v1: a sequential native path and a
    # regrouped family sum need not agree in float32.
    E = torch.tensor([10000.0], dtype=torch.float32)
    A = torch.tensor([-10000.0], dtype=torch.float32)
    M = torch.tensor([0.00146484375], dtype=torch.float32)
    regrouped = E + A + M
    native_reference = torch.tensor([0.0], dtype=torch.float32)
    assert float((regrouped - native_reference).abs().max()) == .00146484375
    R = native_reference - regrouped
    corrected = regrouped + R
    assert torch.equal(corrected, native_reference)


def test_current_state_uses_remainder_after_E_A_M_sum():
    torch = pytest.importorskip("torch")

    class Linear:
        def __init__(self, weight):
            self.weight = weight

    class Attention:
        pass

    attention = Attention()
    attention.c_v = Linear(torch.eye(18))
    attention.lamb = torch.tensor(.2)
    E = torch.arange(18, dtype=torch.float32).reshape(1, 1, 18)
    A = torch.ones_like(E)
    M = torch.ones_like(E) * 2
    R = torch.zeros_like(E)
    R[..., 6] = 4
    observed = run._current_from_state(E, A, M, R, attention, torch, torch.nn.functional)
    normalized = torch.nn.functional.rms_norm((E + A + M) + R, (18,))
    expected = .8 * normalized.reshape(1, 1, 9, 2)[:, :, 3]
    assert torch.allclose(observed, expected)
    without_remainder = run._current_from_state(
        E, A, M, torch.zeros_like(R), attention, torch, torch.nn.functional)
    assert not torch.equal(observed, without_remainder)


def test_v2_score_keeps_v1_science_and_gates_corrected_closure():
    predictions = run.score(_evidence(), _exact())["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert predictions["pred_b_embedding_carries_task"]
    exactness = _exact()
    exactness["state_sum_max_absolute_error"] = 1.0
    assert not run.score(_evidence(), exactness)["predictions"]["pred_a_instrument_live"]


def test_uncorrected_remainder_diagnostic_is_required_but_not_loosened_into_gate():
    exactness = _exact()
    del exactness["uncorrected_state_max_absolute_error"]
    with pytest.raises(run.UpstreamWriterFactorialV2Error, match="remainder diagnostic is missing"):
        run.score(_evidence(), exactness)
    exactness = _exact()
    exactness["uncorrected_state_max_absolute_error"] = 100.0
    assert run.score(_evidence(), exactness)["predictions"]["pred_a_instrument_live"]
