#!/usr/bin/env python3
"""Focused CPU tests for the OOD grouped-source tangent screen."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
import run_task14_ood_fronted_mlp6_7_contextual_midpoint_tangent as run


def test_plan_is_hash_bound_and_honest_about_reuse():
    plan = run.compile_plan()
    assert plan["row_count"] == 16
    assert plan["split"] == "OOD_TEXT_REUSE_NEW_MLP6_7_TANGENT_INTERVENTION"
    assert "already-open" in plan["data_status"]
    assert plan["price"]["example_evaluations"] == 608


def test_ood_rows_match_subject_position_and_endpoint_contract():
    rows = run.build_rows()
    assert len(rows) == 16
    assert {row["subject_position"] for row in rows} == {8}
    assert {row["direction_id"] for row in rows} == {
        "singular_to_plural", "plural_to_singular"}
    assert all(set(row["endpoints"]) == {
        "recipient", "opposite_same_lemma", "same_number_different_lemma"}
               for row in rows)


def test_no_model_dry_run_matches_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                          check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == run.compile_plan()


def test_preflight_fails_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0"*64)
    with pytest.raises(run.OODTangentError, match="prior-art"):
        run.validate_preflight()


def test_score_rejects_incomplete_lattice():
    torch = pytest.importorskip("torch")
    with pytest.raises(run.OODTangentError, match="evidence lattice"):
        run.score([], {}, {}, torch)
