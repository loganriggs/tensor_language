from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name(
    "equality_distributed_finite_transition_quotient_rung528_terminal_audit.py")
SPEC = importlib.util.spec_from_file_location("r528_audit", PATH)
A = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(A)


def test_terminal_audit_recomputes_the_registered_discovery_stop():
    report = A.audit()
    assert report["status"] == "audit_passed"
    assert report["material_pair_count"] == 3
    assert report["precontrol_passer_count"] == 0
    assert report["physical_confirmation_validation_sealed"]
    assert report["calls_reconciled"]
