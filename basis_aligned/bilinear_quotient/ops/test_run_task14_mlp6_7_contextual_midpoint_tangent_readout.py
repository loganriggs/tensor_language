#!/usr/bin/env python3
"""Focused CPU tests for the contextual Task14 tangent readout."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as run


def test_plan_is_complete_and_hash_bound():
    plan=run.compile_plan()
    assert len(plan["conditions"])==16
    assert plan["price"]["example_evaluations"]==608
    assert plan["background_subsets"]=={"recipient":["","YZ"],"donor_context":["EAUW","EAUWYZ"]}


def test_midpoint_jvp_is_exact_for_quadratic_map():
    torch=pytest.importorskip("torch")
    base=torch.tensor([[1.,2.]],requires_grad=False); source=torch.tensor([[3.,-1.]])
    function=lambda x: x.square()+2*x
    primal,exact,endpoint,midpoint=run._directional_jvps(function,base,source,torch)
    assert torch.allclose(primal,function(base))
    assert not torch.allclose(endpoint,exact-primal)
    assert torch.allclose(midpoint,exact-primal,atol=1e-6)


def test_no_model_dry_run_matches_plan():
    env=dict(os.environ,BQLIB_NO_MODEL="1",PYTHONDONTWRITEBYTECODE="1")
    done=subprocess.run([sys.executable,str(Path(run.__file__))],env=env,check=True,capture_output=True,text=True)
    assert json.loads(done.stdout)==run.compile_plan()


def test_preflight_fails_closed(monkeypatch):
    monkeypatch.setattr(run,"PARENT_RESULT_SHA256","0"*64)
    with pytest.raises(run.ContextualTangentError,match="parent result"):
        run.validate_preflight()


def test_compile_patch_reinstalls_both_recipient_base_arms():
    torch=pytest.importorskip("torch")
    rows=run.build_rows()
    tokens=torch.zeros((len(rows),3),dtype=torch.long)
    heads={condition: torch.zeros((len(rows),2)) for condition in run.CONDITIONS}
    patch=run._compile_patch(tokens,heads,rows,torch)
    masked=[condition for (_,condition,_), keep in zip(
        patch["specs"],patch["native_reinstall_mask"].tolist()) if keep]
    assert masked==["opposite_recipient_base","lexical_recipient_base"]*len(rows)


def test_score_fails_closed_on_incomplete_evidence():
    torch=pytest.importorskip("torch")
    with pytest.raises(run.ContextualTangentError,match="evidence lattice"):
        run.score([],{}, {},torch)
