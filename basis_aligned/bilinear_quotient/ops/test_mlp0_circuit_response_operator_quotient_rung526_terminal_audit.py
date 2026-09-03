"""Mutation tests for rung 526's terminal auditor."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
ROOT = OPS.parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load("mlp0_circuit_response_operator_quotient_rung526_terminal_audit")
RESULT = json.load(open(ROOT / "mlp0_circuit_response_operator_quotient_rung526_results.json"))
ARTIFACT = torch.load(ROOT / "mlp0_circuit_response_operator_quotient_rung526_pairs.pt", map_location="cpu", weights_only=True)
OLD = torch.load(ROOT / "mlp0_token_context_operator_quotient_rung525_pairs.pt", map_location="cpu", weights_only=True)
ARTIFACT_SHA = RESULT["pair_artifact"]["sha256"]


def test_landed_result_recomputes_exactly():
    audit = AUDIT.audit_terminal_result(RESULT, ARTIFACT, OLD, artifact_sha256=ARTIFACT_SHA)
    assert audit["passes"] and audit["strong_null"]
    assert not audit["validation_circuits_opened"]


def test_tampered_distance_is_rejected():
    artifact = copy.deepcopy(ARTIFACT)
    artifact["d1_candidate_distance"] *= 0.01
    with pytest.raises(ValueError, match="recomputed score differs"):
        AUDIT.audit_terminal_result(RESULT, artifact, OLD, artifact_sha256=ARTIFACT_SHA)


def test_illegal_validation_open_is_rejected():
    result = copy.deepcopy(RESULT)
    result["validation_circuits_opened"] = True
    with pytest.raises(ValueError, match="validation circuits were opened"):
        AUDIT.audit_terminal_result(result, ARTIFACT, OLD, artifact_sha256=ARTIFACT_SHA)


def test_wrong_execution_count_is_rejected():
    result = copy.deepcopy(RESULT)
    result["execution_price"]["batched_backwards"] += 1
    with pytest.raises(ValueError, match="execution count differs"):
        AUDIT.audit_terminal_result(result, ARTIFACT, OLD, artifact_sha256=ARTIFACT_SHA)
