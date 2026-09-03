from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("equality_shared_private_transition_consensus_rung529_terminal_audit.py")
SPEC = importlib.util.spec_from_file_location("r529_audit", PATH)
R = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R)


def test_metrics_exact_identity_and_sign():
    value = torch.tensor([1.0, -2.0, 3.0])
    assert R.metrics(value, value) == pytest.approx({"cosine": 1.0, "relative_residual": 0.0})
    assert R.metrics(value, -value) == pytest.approx({"cosine": -1.0, "relative_residual": 2.0})


def test_terminal_audit_recomputes_registered_result(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path / "audit.json")
    report = R.audit()
    assert report["audit_passes"]
    assert report["recomputed_candidates"] == [
        {"target": "Z7", "single_donor": "P", "wrong_control": "W8"}]
    assert report["recomputed_confirmation_passers"] == []
    assert report["validation_and_selectivity_sealed"]
    half0 = report["recomputed_confirmation"]["Z7"]["windows"]["half0"]
    assert half0["clauses"]["beats_frozen_single_by_003"]
    assert not half0["clauses"]["beats_frozen_wrong_by_010_cosine"]
