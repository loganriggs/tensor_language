#!/usr/bin/env python3
"""Focused CPU tests for the Task14 subject-term/complement factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_subject_term_vs_complement_factorial as run


def _exact(value=0.0):
    return {"native_replay_max_absolute_logit_error": value,
            "source_term_sum_max_absolute_error": value,
            "same_batch_native_noop_endpoint_max_absolute_error": value,
            "installed_head_max_absolute_error": value,
            "complete_head_vector_max_absolute_error": value}


def _evidence(*, asymmetric=False):
    output = []
    for row in run.build_rows():
        p2s = row["direction_id"] == "plural_to_singular"
        subject = -.2 if p2s else .3
        complement = (-.3 if p2s else .5) if asymmetric else (.7 if p2s else .8)
        complete = 1.0
        values = {"native_neither": (0.0, 0.0),
                  "opposite_subject_only": (subject, subject),
                  "opposite_complement_only": (complement, complement),
                  "complete_opposite_head": (complete, complete)}
        for condition, (margin, ce) in values.items():
            output.append({"row_id": row["row_id"],
                "cell_id": f"{row['direction_id']}__{row['template_id']}",
                "condition": condition, "donor_margin_improvement": margin,
                "donor_CE_improvement": ce})
    return output


def test_preflight_is_candidate_specific_and_holdout_only():
    plan = run.compile_plan()
    assert plan["preflight_validated"] is True
    assert plan["candidate_id"] == run.CANDIDATE_ID
    assert plan["license_sha256"] == run.LICENSE_SHA256
    assert plan["row_count"] == 16 and plan["split"] == "LICENSED_HOLDOUT"
    assert plan["price"] == {"model_forwards": 4, "example_evaluations": 192,
                             "backwards": 0, "parameter_updates": 0}
    assert {row["group_number"] for row in run.build_rows()} == set(range(8, 16))


def test_no_model_dry_run_validates_license_without_loading_model():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_preflight_fails_closed_for_wrong_derivative_license(monkeypatch):
    monkeypatch.setattr(run, "LICENSE_SHA256", "0"*64)
    with pytest.raises(run.licensing.CapabilityLicenseError, match="license hash changed"):
        run.validate_preflight()


def test_exact_partition_compiler_and_same_batch_noop_mask():
    torch = pytest.importorskip("torch")
    rows = run.build_rows(); n, width = len(rows), 5
    p = torch.arange(n*9, dtype=torch.float32).reshape(n, 9)/100+1
    u = torch.arange(n*9*width, dtype=torch.float32).reshape(n, 9, width)/100
    recipient = {"p": p, "u": u, "head": torch.einsum("bk,bkd->bd", p, u)}
    opposite = {"p": p+1, "u": u+2}
    opposite["head"] = torch.einsum("bk,bkd->bd", opposite["p"], opposite["u"])
    tokens = torch.tensor([row["endpoints"]["recipient"]["ids"] for row in rows])
    patch = run._compile(tokens, recipient, opposite, rows, torch)
    assert len(patch["specs"]) == 64
    assert int(patch["native_reinstall_mask"].sum()) == 16
    for i, (row_index, condition, _) in enumerate(patch["specs"]):
        expected = {"native_neither": patch["sr"]+patch["cr"],
                    "opposite_subject_only": patch["so"]+patch["cr"],
                    "opposite_complement_only": patch["sr"]+patch["co"],
                    "complete_opposite_head": patch["so"]+patch["co"]}[condition]
        assert torch.allclose(patch["replacement_heads"][i], expected[row_index])
        assert bool(patch["native_reinstall_mask"][i]) == (condition == "native_neither")


def test_interaction_repair_and_independent_complement_are_separate_predictions():
    scored = run.score(_evidence(), _exact())
    assert scored["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_interaction_repairs_p2s": True,
        "pred_c_complement_independently_carries_task": True,
        "pred_d_complement_asymmetry_persists": False}
    p2s = scored["cells"]["plural_to_singular__across_beside"]
    assert p2s["interaction"]["mean_margin"] == pytest.approx(.5)


def test_asymmetric_complement_is_opposing_outcome():
    scored = run.score(_evidence(asymmetric=True), _exact())
    assert scored["predictions"]["pred_b_interaction_repairs_p2s"]
    assert not scored["predictions"]["pred_c_complement_independently_carries_task"]
    assert scored["predictions"]["pred_d_complement_asymmetry_persists"]


def test_exactness_and_complete_head_controls_can_fail():
    assert not run.score(_evidence(), _exact(1.0))["predictions"]["pred_a_instrument_live"]
    evidence = _evidence()
    for item in evidence:
        if item["condition"] == "complete_opposite_head":
            item["donor_margin_improvement"] = 0.0
            item["donor_CE_improvement"] = 0.0
    assert not run.score(evidence, _exact())["predictions"]["pred_a_instrument_live"]


def test_missing_arm_fails_closed():
    with pytest.raises(run.ComplementFactorialError, match="exact licensed factorial"):
        run.score(_evidence()[:-1], _exact())
