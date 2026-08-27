from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import early_mlp_state_complete_compiler_v2_preflight as preflight


def test_preflight_binds_committed_prereg_rows_and_model() -> None:
    assert preflight.file_sha256(preflight.PREREG) == preflight.PINS[preflight.PREREG]
    assert preflight.file_sha256(preflight.ROWS_RECEIPT) == preflight.PINS[
        preflight.ROWS_RECEIPT
    ]
    prereg = json.loads(preflight.PREREG.read_text())
    assert prereg["native_algebra"]["type_gate"].startswith("Fail closed")
    assert preflight.MODEL_SNAPSHOT / "config.json" in preflight.PINS


def test_source_closure_contains_runner_and_every_new_focused_test() -> None:
    names = {path.name for path in preflight.SOURCE_CLOSURE}
    assert {
        "early_mlp_state_complete_compiler_v2_preflight.py",
        "test_early_mlp_state_complete_compiler_v2_preflight.py",
        "test_early_mlp_state_complete_compiler_v2.py",
        "test_state_complete_compiler_runtime_v2.py",
        "test_prepare_state_complete_compiler_rows_v2.py",
    }.issubset(names)


def test_scale_aware_bounds_are_explicit_and_finite() -> None:
    value = torch.tensor([-3.0, 2.0])
    bound = preflight.scaled_tolerance(value, 2e-6)
    assert bound == {"relative_multiplier": 2e-6, "scale_max_1": 3.0,
                     "tolerance": 6e-6}
    with pytest.raises(ValueError, match="positive"):
        preflight.scaled_tolerance(value, 0.0)


def test_native_type_gate_rejects_missing_or_biased_factors() -> None:
    with pytest.raises(RuntimeError, match="lacks"):
        preflight._native_tensors(SimpleNamespace(mlp=SimpleNamespace()))

    linear = lambda out_dim, in_dim, bias: SimpleNamespace(  # noqa: E731
        weight=torch.zeros(out_dim, in_dim), bias=bias
    )
    mlp = SimpleNamespace(
        Left=linear(4608, 1152, torch.zeros(4608)),
        Right=linear(4608, 1152, None),
        Down=linear(1152, 4608, None),
        Down_bias=torch.zeros(1152),
    )
    with pytest.raises(RuntimeError, match="bias-free"):
        preflight._native_tensors(SimpleNamespace(mlp=mlp))


def test_preflight_outputs_and_lock_are_isolated_from_v1() -> None:
    assert "state_complete_compiler_v2" in preflight.RESULT.name
    assert "state_complete_compiler_v2" in preflight.MANIFEST.name
    assert "state_complete_compiler_v2" in preflight.LOCK.name
    assert all("affine_compiler_v1" not in str(path) for path in preflight.OUTPUTS)
