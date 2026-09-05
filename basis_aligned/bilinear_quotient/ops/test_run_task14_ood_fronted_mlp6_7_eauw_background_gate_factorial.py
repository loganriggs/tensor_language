#!/usr/bin/env python3
"""Focused CPU tests for the OOD E/A/U/W by grouped-X factorial."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as run


def test_plan_has_complete_bounded_lattice():
    plan = run.compile_plan()
    assert len(plan["background_subsets"]) == 16
    assert plan["condition_count"] == 64
    assert plan["price"] == {"physical_model_forwards": 10,
        "example_evaluations": 2144, "causal_installations": 1024,
        "backwards": 0, "parameter_updates": 0, "maximum_patch_chunk_rows": 256}


def test_canonical_subset_preserves_parent_family_order():
    assert run._canonical_subset("WAE") == "EAW"
    assert run._canonical_subset("WUYZ") == "UWYZ"


def test_mobius_and_shapley_accounting_on_known_polynomial():
    values = {}
    for size in range(6):
        for parts in __import__("itertools").combinations(run.FACTORS, size):
            key = "".join(parts); chosen = set(parts)
            values[key] = (2*("X" in chosen) + 3*({"E","X"} <= chosen)
                           - 4*({"A","U","X"} <= chosen))
    terms = run._mobius(values)
    assert terms["X"] == pytest.approx(2)
    assert terms["EX"] == pytest.approx(3)
    assert terms["AUX"] == pytest.approx(-4)
    contextual = {"E": terms["EX"], "AU": terms["AUX"]}
    attribution = {factor: sum(v/len(s) for s,v in contextual.items() if factor in s)
                   for factor in run.BACKGROUND_FACTORS}
    assert sum(attribution.values()) == pytest.approx(-1)
    assert values["EAUWX"]-values["EAUW"]-values["X"]+values[""] == pytest.approx(-1)


def test_no_model_dry_run_matches_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                          check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == run.compile_plan()


def test_preflight_fails_closed(monkeypatch):
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0"*64)
    with pytest.raises(run.OODBackgroundGateError, match="parent"):
        run.validate_preflight()


def test_score_rejects_incomplete_lattice():
    with pytest.raises(run.OODBackgroundGateError, match="background lattice"):
        run.score([], {})
