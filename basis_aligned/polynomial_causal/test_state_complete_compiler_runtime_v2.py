from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import state_complete_compiler_runtime_v2 as runtime


class DummyMLP:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * value

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return self.forward(value)


def blocks():
    return [SimpleNamespace(mlp=DummyMLP(site + 2.0)) for site in range(3)]


def bases():
    eye = torch.eye(runtime.D_MODEL)
    return {0: eye[:, :runtime.COEFFICIENT_DIM].contiguous(),
            1: eye[:, :runtime.COEFFICIENT_DIM].contiguous()}


def constant_program(interface: str, value: float = 0.0):
    state = {
        "grammar": "constant",
        "interface": interface,
        "bias": torch.full((runtime.COEFFICIENT_DIM,), value),
    }
    return {"candidate": {0: state, 1: state}}


def test_state_complete_runtime_subtracts_live_mo_projection() -> None:
    z = torch.randn(2, 5, runtime.D_MODEL)
    mo = torch.randn_like(z)
    basis = bases()[0]
    state = constant_program("state_complete_p", 0.25)["candidate"][0]
    coefficients = runtime.runtime_coefficients(z, mo, basis, state)
    expected = 0.25 - mo.float().reshape(-1, runtime.D_MODEL)[:, :64]
    assert torch.equal(coefficients, expected)


def test_z_only_anchor_deliberately_omits_live_mo() -> None:
    z = torch.randn(2, 5, runtime.D_MODEL)
    mo = torch.randn_like(z)
    state = constant_program("z_only_c", 0.25)["candidate"][0]
    coefficients = runtime.runtime_coefficients(z, mo, bases()[0], state)
    assert torch.equal(coefficients, torch.full_like(coefficients, 0.25))


def test_native_runtime_evaluates_serialized_terms() -> None:
    generator = torch.Generator().manual_seed(3)
    z = torch.randn(2, 4, runtime.D_MODEL, generator=generator)
    left = torch.randn(7, runtime.D_MODEL, generator=generator)
    right = torch.randn(7, runtime.D_MODEL, generator=generator)
    decoder = torch.randn(7, runtime.COEFFICIENT_DIM, generator=generator)
    beta = torch.randn(runtime.COEFFICIENT_DIM, generator=generator)
    state = {"grammar": "native", "interface": "state_complete_p",
             "left": left, "right": right,
             "projected_decoder": decoder, "beta": beta}
    predicted = runtime.runtime_projected_output(z, state)
    flat = z.reshape(-1, runtime.D_MODEL)
    expected = ((flat @ left.T) * (flat @ right.T)) @ decoder + beta
    assert torch.allclose(predicted, expected)


def test_capture_records_p_mo_and_c_on_identical_positions() -> None:
    model_blocks = blocks()
    hook = runtime.StateCompleteCorrectionHook(bases(), {})
    hook.configure({}, capture_site=0)
    z = torch.randn(2, 256, runtime.D_MODEL)
    mo = 0.5 * z
    with runtime.OriginalMLPCallGuard(model_blocks, {0}) as guard:
        returned = hook(0, model_blocks[0], z, mo)
    guard.assert_contract(require_allowed_calls=True)
    captured = hook.captured()
    assert torch.equal(returned, mo)
    assert captured["z"].shape == (128, runtime.D_MODEL)
    sampled = z[:, 64::3, :64].reshape(-1, 64)
    assert torch.allclose(captured["p"], 2.0 * sampled)
    assert torch.allclose(captured["mo"], 0.5 * sampled)
    assert torch.allclose(captured["c"], 1.5 * sampled)


def test_teacher_adjoint_capture_preserves_projected_state_and_severs_upstream() -> None:
    model_blocks = blocks()
    hook = runtime.StateCompleteCorrectionHook(bases(), {})
    hook.configure({}, capture_site=0, capture_adjoint=True)
    z = torch.randn(2, 256, runtime.D_MODEL, requires_grad=True)
    mo = (0.5 * z).clone()
    returned = hook(0, model_blocks[0], z, mo)
    expected = mo.detach().clone()
    expected[:, :, :64] = 2.0 * z.detach()[:, :, :64]
    assert torch.allclose(returned, expected)
    weights = torch.linspace(0.1, 1.0, runtime.D_MODEL).view(1, 1, -1)
    loss = (returned.float() * weights).sum()
    loss.backward()
    hook.collect_pending_adjoint()
    captured = hook.captured()
    expected_adjoint = weights[0, 0, :64].expand(2 * 64, -1)
    assert torch.allclose(captured["adjoint"], expected_adjoint)
    assert z.grad is None


def test_compiled_state_complete_arm_uses_no_original_call() -> None:
    model_blocks = blocks()
    programs = constant_program("state_complete_p", 0.0)
    hook = runtime.StateCompleteCorrectionHook(bases(), programs)
    hook.configure({0: "Q", 1: "Q"}, program_name="candidate")
    z = torch.randn(1, 4, runtime.D_MODEL)
    mo = torch.randn_like(z)
    with runtime.OriginalMLPCallGuard(model_blocks, set()) as guard:
        out0 = hook(0, model_blocks[0], z, mo)
        out1 = hook(1, model_blocks[1], z, mo)
    guard.assert_contract(require_allowed_calls=False)
    expected = mo.clone()
    expected[:, :, :64] = 0.0
    assert torch.equal(out0, expected)
    assert torch.equal(out1, expected)


def test_hook_rejects_incomplete_program_and_guard_restores() -> None:
    hook = runtime.StateCompleteCorrectionHook(bases(), {"bad": {0: {}}})
    with pytest.raises(ValueError, match="lacks sites"):
        hook.configure({1: "Q"}, program_name="bad")
    model_blocks = blocks()
    original = model_blocks[0].mlp.forward
    with pytest.raises(RuntimeError, match="poisoned original MLP0"):
        with runtime.OriginalMLPCallGuard(model_blocks, {1}):
            model_blocks[0].mlp(torch.ones(1))
    assert model_blocks[0].mlp.forward == original
