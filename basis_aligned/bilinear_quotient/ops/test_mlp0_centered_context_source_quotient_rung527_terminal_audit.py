from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("mlp0_centered_context_source_quotient_rung527_terminal_audit.py")
SPEC = importlib.util.spec_from_file_location("r527_audit", PATH)
A = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(A)


def test_terminal_result_recomputes_exactly():
    result = json.loads(A.RESULT.read_text())
    audit = A.audit_data(result)
    assert audit["status"] == "audit_passed"
    assert audit["candidate_count_recomputed"] == 0
    assert audit["material_term_count"] == 20
    assert audit["incremental_pair_gates"]["passes_both"] == 0


def test_audit_rejects_a_changed_effect_vector():
    result = json.loads(A.RESULT.read_text())
    changed = copy.deepcopy(result)
    changed["discovery"]["effects_by_half"][0][0][0] += 1e-6
    with pytest.raises(AssertionError, match="effect vectors"):
        A.audit_data(changed)


def test_audit_rejects_opened_confirmation_after_failed_gate():
    result = json.loads(A.RESULT.read_text())
    changed = copy.deepcopy(result)
    changed["confirmation_opened"] = True
    with pytest.raises(AssertionError, match="confirmation"):
        A.audit_data(changed)
