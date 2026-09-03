"""Mutation checks for rung 524's terminal auditor using the landed receipt."""

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


AUDIT = _load("attention8_direct_grassmann_optimizer_falsifier_rung524_terminal_audit")
RESULT = json.load(open(ROOT / "attention8_direct_grassmann_optimizer_falsifier_rung524_results.json"))
ARTIFACT = torch.load(
    ROOT / "attention8_direct_grassmann_optimizer_falsifier_rung524_frames.pt",
    map_location="cpu", weights_only=False,
)
ARTIFACT_SHA = RESULT["frame_archive"]["file_sha256"]


def test_landed_result_recomputes_exactly():
    audit = AUDIT.audit_terminal_result(RESULT, ARTIFACT, artifact_file_sha256=ARTIFACT_SHA)
    assert audit["passes"]
    assert audit["passing_fit_count"] == 0
    assert not audit["ood_opened"]


def test_tampered_frame_is_rejected():
    artifact = copy.deepcopy(ARTIFACT)
    key = next(iter(artifact["frames"]))
    artifact["frames"][key][0, 0] += 1
    with pytest.raises(ValueError, match="frame hash differs"):
        AUDIT.audit_terminal_result(RESULT, artifact, artifact_file_sha256=ARTIFACT_SHA)


def test_illegal_ood_open_is_rejected():
    result = copy.deepcopy(RESULT)
    result["ood_losses"] = [0.0] * 15
    with pytest.raises(ValueError, match="OOD was evaluated"):
        AUDIT.audit_terminal_result(result, ARTIFACT, artifact_file_sha256=ARTIFACT_SHA)
