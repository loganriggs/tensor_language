"""Mutation checks for rung 525's terminal auditor."""

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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load("mlp0_token_context_operator_quotient_rung525_terminal_audit")
RESULT = json.load(open(ROOT / "mlp0_token_context_operator_quotient_rung525_results.json"))
ARTIFACT = torch.load(
    ROOT / "mlp0_token_context_operator_quotient_rung525_pairs.pt",
    map_location="cpu", weights_only=True,
)
ARTIFACT_SHA = RESULT["pair_artifact"]["sha256"]


def test_landed_result_recomputes_exactly():
    result = AUDIT.audit_terminal_result(RESULT, ARTIFACT, artifact_file_sha256=ARTIFACT_SHA)
    assert result["passes"]
    assert result["strong_null"]
    assert not result["physical_successor_licensed"]


def test_tampered_distance_is_rejected():
    artifact = copy.deepcopy(ARTIFACT)
    artifact["bank_b_candidate_distance"][0] += 100
    with pytest.raises(ValueError, match="recomputed score differs"):
        AUDIT.audit_terminal_result(RESULT, artifact, artifact_file_sha256=ARTIFACT_SHA)


def test_illegal_downstream_call_is_rejected():
    result = copy.deepcopy(RESULT)
    result["execution_price"]["downstream_model_forwards"] = 1
    with pytest.raises(ValueError, match="unregistered downstream"):
        AUDIT.audit_terminal_result(result, ARTIFACT, artifact_file_sha256=ARTIFACT_SHA)


def test_illegal_physical_license_is_rejected():
    result = copy.deepcopy(RESULT)
    result["physical_downstream_successor_licensed"] = True
    with pytest.raises(ValueError, match="incorrectly licensed"):
        AUDIT.audit_terminal_result(result, ARTIFACT, artifact_file_sha256=ARTIFACT_SHA)
