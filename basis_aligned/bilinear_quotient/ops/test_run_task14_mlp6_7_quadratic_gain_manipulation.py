#!/usr/bin/env python3
"""Focused CPU tests for the Task14 quadratic gain manipulation."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
import run_task14_mlp6_7_quadratic_gain_manipulation as run


def test_plan_is_hash_bound_and_fixed_price():
    plan = run.compile_plan()
    assert plan["gains"] == [-0.5, 0.5, 1.5]
    assert plan["price"]["causal_interventions"] == 384
    assert plan["price"]["physical_model_forwards"] == 4


def test_gain_ids_are_unambiguous():
    assert [run._gain_id(x) for x in run.GAINS] == ["m0p5", "p0p5", "p1p5"]


def test_quadratic_formula_predicts_unseen_gains():
    torch = pytest.importorskip("torch")
    x0 = torch.tensor([[1., -2.]])
    x1 = torch.tensor([[3., 1.]])
    fn = lambda x: x.square() + 2*x + 1
    base, _, endpoint, midpoint = run.tangent._directional_jvps(fn, x0, x1, torch)
    curvature = midpoint - endpoint
    for gain in run.GAINS:
        predicted = base + gain*endpoint + gain*gain*curvature
        assert torch.allclose(predicted, fn(x0 + gain*(x1-x0)), atol=1e-6)


def test_no_model_dry_run_matches_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                          check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == run.compile_plan()


def test_preflight_fails_closed(monkeypatch):
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0"*64)
    with pytest.raises(run.QuadraticGainError, match="parent result"):
        run.validate_preflight()


def test_score_rejects_incomplete_lattice():
    torch = pytest.importorskip("torch")
    with pytest.raises(run.QuadraticGainError, match="evidence lattice"):
        run.score([], {}, {}, torch)
