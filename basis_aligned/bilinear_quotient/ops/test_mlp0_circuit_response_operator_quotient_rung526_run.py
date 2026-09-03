"""CPU checks for rung 526's runner pre-model gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = _load("mlp0_circuit_response_operator_quotient_rung526_run")


def test_planted_cross_circuit_quotient_passes():
    result = RUN._planted_toy()
    assert result["passes"]
    assert result["correct_class_fraction"] >= 0.95
    assert result["scrambled_correct_class_fraction"] <= 0.25


def test_differentiable_contraction_toy_passes():
    result = RUN._gradient_toy()
    assert result["passes"]
    assert result["relative_squared_error"] <= 1e-10


def test_phase_instrument_requires_every_clause():
    valid = {
        "identity_leaf_logit_max_abs": 0.0,
        "all_circuit_gradients_nonzero": True,
        "member_weight_sum_max_abs_error": 1e-7,
        "control_weight_sum_max_abs_error": 1e-7,
        "aggregate_contraction_relative_squared_error": 1e-7,
        "signature_finite": True, "signature_nonconstant": True,
    }
    assert RUN._phase_instrument_passes(valid)
    for key in tuple(valid):
        changed = dict(valid)
        changed[key] = False if isinstance(changed[key], bool) else 1.0
        assert not RUN._phase_instrument_passes(changed)


def test_circuit_mean_difference_weights_sum_to_zero():
    masks = {
        "c": {
            "member": torch.tensor([1, 0, 0, 1], dtype=torch.bool).repeat(64_000),
            "slice_control": torch.tensor([0, 1, 1, 0], dtype=torch.bool).repeat(64_000),
        }
    }
    counts = RUN._phase_counts(masks, ["c"], (0, 1))
    assert counts.tolist() == [[128.0, 128.0]]
