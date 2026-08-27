from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import affine_compiler_runtime_v1 as runtime


class DummyMLP:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * value

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return self.forward(value)


def blocks():
    return [SimpleNamespace(mlp=DummyMLP(site + 2.0)) for site in range(3)]


def bases(d: int = runtime.D_MODEL, k: int = 64):
    eye = torch.eye(d)
    return {0: eye[:, :k].contiguous(), 1: eye[:, :k].contiguous()}


def zero_program():
    state = {
        "mean": torch.zeros(runtime.D_MODEL),
        "scale": torch.ones(runtime.D_MODEL),
        "bias": torch.zeros(64),
        "left": torch.zeros(runtime.D_MODEL, 8),
        "right": torch.zeros(8, 64),
    }
    return {"main": {0: state, 1: state}}


def test_original_guard_poison_and_restore() -> None:
    model_blocks = blocks()
    original = model_blocks[0].mlp.forward
    with pytest.raises(RuntimeError, match="poisoned original MLP0"):
        with runtime.OriginalMLPCallGuard(model_blocks, {1}):
            model_blocks[0].mlp(torch.ones(2, 3))
    assert model_blocks[0].mlp.forward == original
    assert torch.equal(model_blocks[0].mlp(torch.ones(1)), torch.full((1,), 2.0))


def test_original_guard_counts_only_allowlisted_calls() -> None:
    model_blocks = blocks()
    with runtime.OriginalMLPCallGuard(model_blocks, {1}) as guard:
        model_blocks[1].mlp(torch.ones(2, 3))
    guard.assert_contract(require_allowed_calls=True)
    assert guard.counts == {0: 0, 1: 1, 2: 0}


def test_capture_uses_identical_live_input_and_registered_positions() -> None:
    model_blocks = blocks()
    hook = runtime.CompilerCorrectionHook(bases(), zero_program())
    hook.configure({}, capture_site=0)
    z = torch.randn(2, 256, runtime.D_MODEL)
    mo = 0.5 * z
    with runtime.OriginalMLPCallGuard(model_blocks, {0}) as guard:
        returned = hook(0, model_blocks[0], z, mo)
    guard.assert_contract(require_allowed_calls=True)
    inputs, coefficients = hook.captured()
    assert torch.equal(returned, mo)
    assert inputs.shape == (2 * 64, runtime.D_MODEL)
    expected = (1.5 * z[:, 64::3])[:, :, :64].reshape(-1, 64)
    assert torch.allclose(coefficients, expected)


def test_predicted_arm_runs_with_original_poisoned() -> None:
    model_blocks = blocks()
    hook = runtime.CompilerCorrectionHook(bases(), zero_program())
    hook.configure({0: "Q", 1: "Q"})
    z = torch.randn(1, 256, runtime.D_MODEL)
    mo = torch.randn_like(z)
    with runtime.OriginalMLPCallGuard(model_blocks, set()) as guard:
        out0 = hook(0, model_blocks[0], z, mo)
        out1 = hook(1, model_blocks[1], z, mo)
    guard.assert_contract(require_allowed_calls=False)
    assert torch.equal(out0, mo)
    assert torch.equal(out1, mo)
    assert guard.counts == {0: 0, 1: 0, 2: 0}


def test_oracle_projection_requires_allowlisted_original() -> None:
    model_blocks = blocks()
    hook = runtime.CompilerCorrectionHook(bases(), zero_program())
    hook.configure({0: "O"})
    z = torch.randn(1, 4, runtime.D_MODEL)
    mo = 0.5 * z
    with runtime.OriginalMLPCallGuard(model_blocks, {0}) as guard:
        out = hook(0, model_blocks[0], z, mo)
    guard.assert_contract(require_allowed_calls=True)
    expected = mo.clone()
    expected[:, :, :64] = 2.0 * z[:, :, :64]
    assert torch.allclose(out, expected)
