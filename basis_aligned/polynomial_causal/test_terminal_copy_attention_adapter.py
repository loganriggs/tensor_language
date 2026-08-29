from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention


class FakeNative(nn.Module):
    def __init__(self, width: int = 8, heads: int = 2) -> None:
        super().__init__()
        torch.manual_seed(20260829)
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.lamb = nn.Parameter(torch.tensor(0.37))
        self.n_head = heads
        self.n_embd = width
        head_dim = width // heads
        self.rotary = SimpleNamespace(inv_freq=(
            1.0 / (10_000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        ))

    def forward(self, state: torch.Tensor, first_value: torch.Tensor | None = None):
        batch, sequence, width = state.shape
        head_dim = width // self.n_head

        def project(name: str) -> torch.Tensor:
            return getattr(self, f"c_{name}")(state).view(
                batch, sequence, self.n_head, head_dim,
            )

        positions = torch.arange(sequence, dtype=self.rotary.inv_freq.dtype)
        angles = torch.outer(positions, self.rotary.inv_freq)
        cosine = angles.cos().bfloat16()[None, :, None, :]
        sine = angles.sin().bfloat16()[None, :, None, :]

        def rotate(value: torch.Tensor) -> torch.Tensor:
            value = F.rms_norm(value, (head_dim,))
            first, second = value[..., : head_dim // 2], value[..., head_dim // 2 :]
            return torch.cat((
                first * cosine + second * sine,
                first * (-sine) + second * cosine,
            ), dim=-1).to(value.dtype)

        query, key = rotate(project("q")), rotate(project("k"))
        query2, key2 = rotate(project("q2")), rotate(project("k2"))
        pattern = (
            torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
        ) * (
            torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
        )
        causal = torch.tril(torch.ones(sequence, sequence, dtype=torch.bool))
        pattern = pattern.masked_fill(~causal, 0.0)
        value = project("v")
        bus = value if first_value is None else first_value
        mixed = (1 - self.lamb) * value + self.lamb * bus
        output = torch.einsum("bhqk,bkhd->bqhd", pattern, mixed)
        return self.c_proj(output.reshape(batch, sequence, width)), bus


def test_all_heads_recompose_exact_native_formula_and_shared_value_bus():
    native = FakeNative()
    program = OwnedPerHeadTensorAttention.from_native(native)
    first_state, second_state = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    expected_first, expected_bus = native(first_state)
    with program.begin(first_state) as transaction:
        observed_first = transaction.all_heads()
        observed_bus = transaction.first_value_bus()
    assert torch.equal(observed_first, expected_first)
    assert torch.equal(observed_bus, expected_bus)
    assert transaction.closure.all_head_recomposition_max_abs_error == 0.0

    expected_second, _ = native(second_state, expected_bus)
    with program.begin(second_state, observed_bus) as transaction2:
        observed_second = transaction2.all_heads()
    assert torch.equal(observed_second, expected_second)
    assert transaction2.closure.all_head_recomposition_relative_error == 0.0


def test_disjoint_head_sums_add_and_returned_tensors_do_not_alias_storage():
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    with program.begin(torch.randn(1, 4, 8)) as transaction:
        head0 = transaction.select((0,))
        head1 = transaction.select((1,))
        full = transaction.native_full_write()
        head0.zero_()
        replay = transaction.select((0,)) + head1
        assert torch.equal(replay, full)
    assert transaction.closure.selected_head_sets == ((0,), (1,), (0,))
    with pytest.raises(RuntimeError, match="closed"):
        transaction.all_heads()


def test_program_owns_weights_and_never_calls_native_projection_after_clone():
    native = FakeNative()
    program = OwnedPerHeadTensorAttention.from_native(native)
    for layer in native.modules():
        if isinstance(layer, nn.Linear):
            layer.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("native projection called")
            )
    with program.begin(torch.randn(1, 3, 8)) as transaction:
        assert torch.isfinite(transaction.all_heads()).all()
    assert program.price() == {
        "projection_values": 6 * 8 * 8,
        "scalar_values": 1,
        "rotary_values": 2,
        "total_stored_values": 6 * 8 * 8 + 3,
        "native_calls_per_forward": 0,
        "token_table_values": 0,
        "total_input_support": True,
    }


def test_adapter_refuses_projection_bias_and_invalid_head_selection():
    native = FakeNative()
    native.c_q = nn.Linear(8, 8, bias=True)
    with pytest.raises(ValueError, match="bias-free"):
        OwnedPerHeadTensorAttention.from_native(native)
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    with program.begin(torch.randn(1, 3, 8)) as transaction:
        with pytest.raises(ValueError, match="malformed"):
            transaction.select(())
        with pytest.raises(ValueError, match="malformed"):
            transaction.select((0, 0))
        with pytest.raises(ValueError, match="malformed"):
            transaction.select((2,))

