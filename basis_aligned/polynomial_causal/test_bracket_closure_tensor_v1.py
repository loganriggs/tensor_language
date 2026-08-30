from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from bracket_closure_tensor_v1 import (
    BracketTensorArm,
    PRODUCTION_STORED_VALUES,
    build_bracket_tensor_program,
    cyclic_derangement,
    exact_stored_attention_price,
    program_state_sha256,
    spectral_derange_output_head,
)


class FakeRotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.25]))


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(20260830)
        self.n_embd = 8
        self.n_head = 2
        self.c_q = nn.Linear(8, 8, bias=False)
        self.c_k = nn.Linear(8, 8, bias=False)
        self.c_q2 = nn.Linear(8, 8, bias=False)
        self.c_k2 = nn.Linear(8, 8, bias=False)
        self.c_v = nn.Linear(8, 8, bias=False)
        self.c_proj = nn.Linear(8, 8, bias=False)
        self.lamb = nn.Parameter(torch.tensor(0.3))
        self.rotary = FakeRotary()


def _program(attention: FakeAttention, arm: BracketTensorArm):
    # The production target is H8; this tiny known-answer maps that target to H1.
    import bracket_closure_tensor_v1 as module
    old = module.TARGET_HEAD
    module.TARGET_HEAD = 1
    try:
        return build_bracket_tensor_program(
            attention,
            arm,
            permutation=(cyclic_derangement(4) if arm is BracketTensorArm.DERANGED_H8 else None),
        )
    finally:
        module.TARGET_HEAD = old


def test_stored_all_head_replay_is_owned_and_deletion_is_constant_global_projector() -> None:
    native = FakeAttention()
    full = _program(native, BracketTensorArm.STORED_ALL_HEADS)
    deleted = _program(native, BracketTensorArm.DELETE_H8)
    state = torch.randn(2, 5, 8)
    first_value = torch.randn(2, 5, 2, 4)
    full_write, full_bus = full(state, first_value)
    deleted_write, deleted_bus = deleted(state, first_value)
    assert full_write.shape == deleted_write.shape == state.shape
    assert full_bus is deleted_bus is first_value
    assert torch.equal(full.head_weights, torch.ones(2))
    assert torch.equal(deleted.head_weights, torch.tensor([1.0, 0.0]))
    assert not torch.equal(full_write, deleted_write)
    native_copy = copy.deepcopy(native)
    for module in (native, native_copy):
        for parameter in module.parameters():
            parameter.data.fill_(float("nan"))
    # Stored programs retain no native reference and remain finite after poisoning.
    assert bool(torch.isfinite(full(state, first_value)[0]).all())
    assert bool(torch.isfinite(deleted(state, first_value)[0]).all())


def test_spectral_derangement_preserves_target_slice_singular_values_and_price() -> None:
    weight = torch.diag(torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]))
    null = spectral_derange_output_head(
        weight, head=1, head_dim=4, permutation=cyclic_derangement(4),
    )
    torch.testing.assert_close(null[:, :4], weight[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.svdvals(null[:, 4:]),
        torch.linalg.svdvals(weight[:, 4:]),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(null[:, 4:], weight[:, 4:])
    native = FakeAttention()
    programs = tuple(_program(native, arm) for arm in BracketTensorArm)
    prices = tuple(program.cost_receipt().total_stored_values for program in programs)
    assert prices == (6 * 8 * 8 + 1 + 2 + 2,) * 3
    assert len({program_state_sha256(program) for program in programs}) == 3


def test_price_formula_and_derangement_fail_closed() -> None:
    assert PRODUCTION_STORED_VALUES == 7_962_698
    assert exact_stored_attention_price(width=1152, heads=9, rotary_values=64) == 7_962_698
    with pytest.raises(ValueError, match="fixed-point-free"):
        spectral_derange_output_head(
            torch.eye(8), head=1, head_dim=4,
            permutation=torch.tensor([0, 2, 3, 1]),
        )
    with pytest.raises(ValueError, match="requires"):
        _program(FakeAttention(), BracketTensorArm.STORED_ALL_HEADS) if False else (
            build_bracket_tensor_program(FakeAttention(), BracketTensorArm.DERANGED_H8)
        )
