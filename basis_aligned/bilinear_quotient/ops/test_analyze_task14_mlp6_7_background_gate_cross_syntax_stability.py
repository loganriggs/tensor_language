#!/usr/bin/env python3
"""Focused tests for the frozen-receipt gate stability analysis."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
import analyze_task14_mlp6_7_background_gate_cross_syntax_stability as run


def test_plan_is_zero_gpu_and_hash_bound():
    plan = run.compile_plan()
    assert plan["data_status"] == "RETROSPECTIVE_FROZEN_RECEIPT_REANALYSIS"
    assert set(plan["price"].values()) == {0}


def test_profile_cosine_and_difference():
    left = {"E": 2., "A": 2., "U": 1., "W": 1.}
    right = {"E": 4., "A": 4., "U": 2., "W": 2.}
    assert run._profile(left) == pytest.approx({"E": 1/3, "A": 1/3,
                                                "U": 1/6, "W": 1/6})
    assert run._cosine(run._profile(left), run._profile(right)) == pytest.approx(1.)
    assert run._maximum_difference(run._profile(left), run._profile(right)) == 0


def test_matched_lattice_is_complete_and_closes():
    document = json.loads(run.MATCHED_RESULT.read_text())
    cells = run._matched_cells(document)
    assert len(cells) == 4
    assert max(x["algebra_error"] for x in cells.values()) < 5e-5


def test_no_model_dry_run_matches_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                          check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == run.compile_plan()


def test_preflight_fails_closed(monkeypatch):
    monkeypatch.setattr(run, "OOD_RESULT_SHA256", "0"*64)
    with pytest.raises(run.GateStabilityError, match="OOD result"):
        run.validate_preflight()
