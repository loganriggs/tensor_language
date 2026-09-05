"""Focused CPU tests for the staged narrative A2/C native capability selector."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import native_capability_license as licensing
import run_narrative_tense_a2_c_native_capability_select_holdout as run


def _evidence(rows, *, metric=None):
    values = []
    for row in rows:
        for side in ("base", "donor"):
            correct, margin, ce = True, 1.0, 1.0
            if metric is not None:
                correct, margin, ce = metric(row, side)
            values.append({
                "example_id": f"{row['row_id']}:{side}",
                "cell_id": run._cell_id(row, side),
                "correct": correct,
                "full_vocab_CE": ce,
                "answer_minus_foil_margin": margin,
            })
    return values


def _fit_rows():
    return [row for row in run.authority.build_rows() if row["phase"] == "FIT"]


def test_plan_and_no_model_dry_run_have_exact_maximum_price():
    plan = run.compile_plan()
    assert plan["maximum_price"] == {
        "model_forwards": 2, "example_evaluations": 192,
        "backwards": 0, "parameter_updates": 0,
    }
    assert plan["fit_paired_rows"] == 64
    assert plan["fit_endpoint_evaluations"] == 128
    assert plan["maximum_holdout_endpoint_evaluations"] == 64
    assert plan["model_loaded"] is plan["gpu_accessed"] is plan["queue_touched"] is False

    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], env=env,
        check=True, capture_output=True, text=True)
    emitted = json.loads(completed.stdout)
    assert emitted == plan
    assert emitted["maximum_price"]["model_forwards"] == 2
    assert emitted["maximum_price"]["example_evaluations"] == 192


def test_gate_is_candidate_specific_exact_48_cells_and_package_bound():
    a2, c = "record_coordination", "explicit_period"
    gate = run.build_selected_gate(a2, c)
    assert gate.capability_id == f"narrative_tense.native_capability.{a2}.{c}.v1"
    assert run.causal_candidate_id(a2, c) == \
        f"narrative_tense.attn11_head3_carrier.{a2}.{c}.v1"
    assert gate.authority_logical_sha256 == run.PACKAGE_SHA256[f"{a2}+{c}"]
    assert len(gate.cells) == 48
    assert {cell.minimum_accuracy for cell in gate.cells} == {.875}
    counts = sorted(cell.expected_count for cell in gate.cells)
    assert counts == [2] * 32 + [4] * 16


def test_package_hash_is_independently_recomputed(monkeypatch):
    monkeypatch.setattr(run.authority, "package_sha256", lambda *_: "0" * 64)
    with pytest.raises(run.SelectorError, match="package hash changed"):
        run.build_selected_gate("record_coordination", "explicit_period")


def test_selection_uses_exact_global_lexicographic_rule_and_fixed_ties():
    def metrics(row, side):
        del side
        if row["family"] == "A2":
            margin = {"record_coordination": 2.0, "while_observers": 3.0,
                      "reported_frame": 3.0}[row["template_id"]]
            ce = {"record_coordination": .5, "while_observers": .7,
                  "reported_frame": .8}[row["template_id"]]
        elif row["family"] == "C":
            margin = {"explicit_period": 3.0, "years_nowadays": 3.0,
                      "back_then_right_now": 2.0}[row["template_id"]]
            ce = {"explicit_period": .9, "years_nowadays": .8,
                  "back_then_right_now": .1}[row["template_id"]]
        else:
            margin, ce = 1.0, 1.0
        return True, margin, ce

    selected = run.select_fit_package(_evidence(_fit_rows(), metric=metrics))
    # A2 tie on accuracy/margin is broken by lower worst CE; C likewise.
    assert selected["selected_A2"] == "while_observers"
    assert selected["selected_C"] == "years_nowadays"
    assert selected["fit_package_selected"]

    tied = run.select_fit_package(_evidence(_fit_rows()))
    assert tied["selected_A2"] == run.authority.A2_TEMPLATE_ORDER[0]
    assert tied["selected_C"] == run.authority.C_TEMPLATE_ORDER[0]

    def ineligible_despite_large_metrics(row, side):
        is_first_bad_endpoint = (row["family"] == "A2"
                                 and row["template_id"] == "record_coordination"
                                 and row["direction_id"] == "past_to_present"
                                 and side == "base" and row["group_number"] == 0)
        large = 100.0 if row["template_id"] == "record_coordination" else 1.0
        return not is_first_bad_endpoint, large, .01

    excluded = run.select_fit_package(
        _evidence(_fit_rows(), metric=ineligible_despite_large_metrics))
    assert not excluded["candidate_reports"]["A2"]["record_coordination"]["eligible"]
    assert excluded["selected_A2"] == "while_observers"


def test_selection_fails_closed_on_missing_or_duplicated_fit_endpoint():
    evidence = _evidence(_fit_rows())
    with pytest.raises(run.SelectorError, match="exactly 128"):
        run.select_fit_package(evidence[:-1])
    evidence[-1] = dict(evidence[0])
    with pytest.raises(run.SelectorError, match="duplicated"):
        run.select_fit_package(evidence)


def test_failed_fit_never_builds_or_evaluates_holdout(monkeypatch):
    calls = []

    def fake_evaluate(model, rows, torch, F):
        del model, torch, F
        calls.append({row["phase"] for row in rows})
        evidence = _evidence(rows)
        # One error in a four-example A1 direction/side cell gives .75 < .875.
        first_a1 = next(index for index, item in enumerate(evidence)
                        if item["cell_id"].startswith("FIT/A1/"))
        evidence[first_a1]["correct"] = False
        return evidence

    monkeypatch.setattr(run, "evaluate_native", fake_evaluate)
    monkeypatch.setattr(
        run, "build_selected_gate",
        lambda *_: (_ for _ in ()).throw(AssertionError("holdout gate opened")))
    selection, fit, holdout, capability, result_sha, license_record = \
        run.run_two_stage(None, None, None)
    assert calls == [{"FIT"}]
    assert not selection["fit_package_selected"]
    assert len(fit) == 128 and holdout == []
    assert capability is result_sha is license_record is None


def test_full_pass_finalizes_and_issues_preflight_valid_license(tmp_path, monkeypatch):
    capability_path = tmp_path / "capability.json"
    license_path = tmp_path / "license.json"
    monkeypatch.setattr(run, "CAPABILITY_RESULT", capability_path)
    monkeypatch.setattr(run, "LICENSE", license_path)
    calls = []

    def fake_evaluate(model, rows, torch, F):
        del model, torch, F
        calls.append({row["phase"] for row in rows})
        return _evidence(rows)

    monkeypatch.setattr(run, "evaluate_native", fake_evaluate)
    selection, _, holdout, result, result_sha, license_record = \
        run.run_two_stage(None, None, None)
    assert calls == [{"FIT"}, {"HOLDOUT"}]
    assert len(holdout) == 64 and result["terminal"] == "pass"
    assert result_sha and license_record["sha256"]
    a2, c = selection["selected_A2"], selection["selected_C"]
    gate = run.build_selected_gate(a2, c)
    value = licensing.validate_causal_preflight(
        gate, capability_path, license_path,
        expected_license_sha256=license_record["sha256"],
        causal_candidate_id=run.causal_candidate_id(a2, c))
    assert value["capability_id"].endswith(f"{a2}.{c}.v1")
    assert value["causal_candidate_id"].endswith(f"{a2}.{c}.v1")


def test_holdout_failure_finalizes_fail_and_emits_no_license(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CAPABILITY_RESULT", tmp_path / "capability.json")
    monkeypatch.setattr(run, "LICENSE", tmp_path / "license.json")
    calls = 0

    def fake_evaluate(model, rows, torch, F):
        nonlocal calls
        del model, torch, F
        calls += 1
        evidence = _evidence(rows)
        if calls == 2:
            evidence[0]["correct"] = False
        return evidence

    monkeypatch.setattr(run, "evaluate_native", fake_evaluate)
    _, _, _, result, _, license_record = run.run_two_stage(None, None, None)
    assert result["terminal"] == "fail"
    assert license_record == {"value": None, "sha256": None}
    assert not run.LICENSE.exists()
