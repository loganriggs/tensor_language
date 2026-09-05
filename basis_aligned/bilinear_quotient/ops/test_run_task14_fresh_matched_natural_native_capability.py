#!/usr/bin/env python3
"""Focused CPU tests for the fresh matched-natural Task14 capability stage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as run


def _evidence(rows, *, failed_example: str | None = None):
    evidence = []
    for row in rows:
        for role in run.authority.ROLES:
            example_id = f"{row['row_id']}:{role}"
            correct = example_id != failed_example
            evidence.append({
                "example_id": example_id,
                "cell_id": run._cell_id(row, role),
                "correct": correct,
                "full_vocab_CE": 0.25 if correct else 2.0,
                "answer_minus_foil_margin": 2.0 if correct else -1.0,
            })
    return evidence


def test_authority_is_exact_fresh_matched_and_phase_disjoint():
    rows = run.authority.build_rows()
    assert len(rows) == 32
    assert run.authority.validate_rows(rows) == run.AUTHORITY_LOGICAL_SHA256
    assert {row["phase"] for row in rows} == {"FIT", "HOLDOUT"}
    phase_subjects = {phase: set() for phase in ("FIT", "HOLDOUT")}
    for row in rows:
        endpoints = row["endpoints"]
        recipient = endpoints["recipient"]
        opposite = endpoints["opposite_same_lemma"]
        lexical = endpoints["same_number_different_lemma"]
        assert recipient["ids"][:8] == opposite["ids"][:8] == lexical["ids"][:8]
        assert recipient["ids"][8] != opposite["ids"][8]
        assert recipient["ids"][8] != lexical["ids"][8]
        assert recipient["subject_number"] != opposite["subject_number"]
        assert recipient["subject_number"] == lexical["subject_number"]
        phase_subjects[row["phase"]].update(x["subject"] for x in endpoints.values())
    assert phase_subjects["FIT"].isdisjoint(phase_subjects["HOLDOUT"])


def test_plan_and_no_model_dry_run_have_exact_price():
    plan = run.compile_plan()
    assert plan["maximum_price"] == {
        "model_forwards": 2, "example_evaluations": 96,
        "backwards": 0, "parameter_updates": 0,
    }
    assert plan["fit_rows"] == plan["holdout_rows"] == 16
    assert plan["fit_endpoint_evaluations"] == 48
    assert plan["maximum_holdout_endpoint_evaluations"] == 48
    assert plan["model_loaded"] is plan["gpu_accessed"] is plan["queue_touched"] is False
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], env=env,
        check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == plan


def test_gate_binds_exact_authority_and_twenty_four_cells():
    gate = run.build_gate()
    assert gate.capability_id == run.authority.CAPABILITY_ID
    assert gate.authority_logical_sha256 == run.AUTHORITY_LOGICAL_SHA256
    assert len(gate.cells) == 24
    assert {cell.expected_count for cell in gate.cells} == {4}
    assert {cell.minimum_accuracy for cell in gate.cells} == {.875}
    assert licensing.cells_sha256(gate) == run.compile_plan()["registered_cells_sha256"]


def test_phase_summary_fails_closed_on_missing_duplicate_and_nonfinite():
    rows = run._phase_rows("FIT")
    evidence = _evidence(rows)
    with pytest.raises(run.CapabilityRunnerError, match="count changed"):
        run.summarize_phase(evidence[:-1], "FIT")
    duplicate = list(evidence)
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(run.CapabilityRunnerError, match="duplicated"):
        run.summarize_phase(duplicate, "FIT")
    nonfinite = [dict(item) for item in evidence]
    nonfinite[0]["full_vocab_CE"] = float("nan")
    with pytest.raises(run.CapabilityRunnerError, match="not finite"):
        run.summarize_phase(nonfinite, "FIT")


def test_fit_failure_never_opens_holdout_or_emits_capability(monkeypatch):
    calls = []

    def fake_evaluate(model, rows, torch, F):
        del model, torch, F
        phases = {row["phase"] for row in rows}
        calls.append(phases)
        evidence = _evidence(rows)
        # One miss makes its four-example cell 3/4, below the frozen .875 bar.
        evidence[0]["correct"] = False
        return evidence

    monkeypatch.setattr(run, "evaluate_native", fake_evaluate)
    monkeypatch.setattr(
        run.licensing, "finalize_native_capability",
        lambda *_: (_ for _ in ()).throw(AssertionError("capability finalized")))
    fit, fit_evidence, holdout, holdout_evidence, cap_sha, license_record = \
        run.run_two_stage(None, None, None)
    assert calls == [{"FIT"}]
    assert not fit["passed"] and len(fit_evidence) == 48
    assert holdout is None and holdout_evidence == []
    assert cap_sha is license_record is None


def test_full_pass_finalizes_generic_result_and_license(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CAPABILITY_RESULT", tmp_path / "capability.json")
    monkeypatch.setattr(run, "LICENSE", tmp_path / "license.json")
    calls = []

    def fake_evaluate(model, rows, torch, F):
        del model, torch, F
        calls.append({row["phase"] for row in rows})
        return _evidence(rows)

    monkeypatch.setattr(run, "evaluate_native", fake_evaluate)
    fit, _, holdout, holdout_evidence, cap_sha, license_record = \
        run.run_two_stage(None, None, None)
    assert calls == [{"FIT"}, {"HOLDOUT"}]
    assert fit["passed"] and holdout["passed"] and len(holdout_evidence) == 48
    assert cap_sha and license_record["sha256"]
    value = licensing.validate_causal_preflight(
        run.build_gate(), run.CAPABILITY_RESULT, run.LICENSE,
        expected_license_sha256=license_record["sha256"],
        causal_candidate_id=run.authority.CAUSAL_CANDIDATE_ID)
    assert value["capability_id"] == run.authority.CAPABILITY_ID


def test_holdout_failure_finalizes_fail_but_emits_no_license(tmp_path, monkeypatch):
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
    fit, _, holdout, _, cap_sha, license_record = run.run_two_stage(None, None, None)
    assert fit["passed"] and not holdout["passed"] and cap_sha
    assert license_record == {"value": None, "sha256": None}
    assert not run.LICENSE.exists()
