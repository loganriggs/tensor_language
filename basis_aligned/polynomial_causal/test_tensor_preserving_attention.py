from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_preserving_attention import (
    PROJECTION_NAMES, StoredLinear, TensorPreservingSquaredAttention,
)


class FakeNative(nn.Module):
    def __init__(self, width: int = 8, heads: int = 2) -> None:
        super().__init__()
        torch.manual_seed(9)
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.lamb = nn.Parameter(torch.tensor(0.37))
        self.n_head = heads
        self.rotary = SimpleNamespace(
            inv_freq=1.0 / (10_000 ** (torch.arange(0, width // heads, 2).float() / (width // heads)))
        )

    def forward(self, state: torch.Tensor, first_value: torch.Tensor | None = None):
        batch, sequence, width = state.shape
        head_dim = width // self.n_head

        def head(name: str) -> torch.Tensor:
            return getattr(self, f"c_{name}")(state).view(
                batch, sequence, self.n_head, head_dim,
            )

        positions = torch.arange(sequence, dtype=self.rotary.inv_freq.dtype)
        angles = torch.outer(positions, self.rotary.inv_freq)
        cosine = angles.cos().bfloat16()[None, :, None, :]
        sine = angles.sin().bfloat16()[None, :, None, :]

        def rotate(value: torch.Tensor) -> torch.Tensor:
            value = F.rms_norm(value, (head_dim,))
            half = head_dim // 2
            first, second = value[..., :half], value[..., half:]
            return torch.cat([
                first * cosine + second * sine,
                first * (-sine) + second * cosine,
            ], dim=-1).to(value.dtype)

        query, key = rotate(head("q")), rotate(head("k"))
        query2, key2 = rotate(head("q2")), rotate(head("k2"))
        pattern = (
            torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
        ) * (
            torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
        )
        causal = torch.tril(torch.ones(sequence, sequence, dtype=torch.bool))
        pattern = pattern.masked_fill(~causal, 0.0)
        value = head("v")
        bus = value if first_value is None else first_value.view_as(value)
        mixed = (1 - self.lamb) * value + self.lamb * bus
        output = torch.einsum("bhqk,bkhd->bqhd", pattern, mixed)
        return self.c_proj(output.reshape(batch, sequence, width)), bus


def exact_program(native: FakeNative) -> TensorPreservingSquaredAttention:
    return TensorPreservingSquaredAttention.from_native(
        native, ranks={name: None for name in PROJECTION_NAMES},
    )


def test_dense_clone_has_exact_native_tensor_formula_and_value_bus() -> None:
    native = FakeNative()
    program = exact_program(native)
    state = torch.randn(2, 5, 8)
    expected0, expected_bus = native(state)
    write0, bus = program(state)
    expected1, _ = native(state * 0.7, expected_bus)
    write1, returned = program(state * 0.7, bus)
    assert torch.equal(write0, expected0)
    assert torch.equal(bus, expected_bus)
    assert torch.equal(write1, expected1)
    assert returned.data_ptr() == bus.data_ptr()
    assert torch.isfinite(write0).all() and torch.isfinite(write1).all()


def test_program_retains_no_native_module_and_reports_zero_calls() -> None:
    native = FakeNative()
    program = exact_program(native)
    for layer in native.modules():
        if isinstance(layer, nn.Linear):
            layer.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("native projection called")
            )
    program(torch.randn(1, 3, 8))
    receipt = program.cost_receipt()
    assert receipt.native_calls_per_forward == 0
    assert receipt.token_table_values == 0
    assert receipt.total_input_support
    assert receipt.total_stored_values == 6 * 8 * 8 + 1 + 2


def test_low_rank_projection_price_and_formula() -> None:
    weight = torch.randn(7, 5)
    layer = StoredLinear.from_weight(weight, rank=3)
    assert not layer.is_dense and layer.rank == 3
    assert layer.stored_values == 3 * (7 + 5)
    value = torch.randn(4, 5)
    expected = (value @ layer.input_factor.T) @ layer.output_factor.T
    assert torch.equal(layer(value), expected)


def test_identity_rank_is_stored_dense_not_as_two_full_factors() -> None:
    weight = torch.randn(6, 4)
    layer = StoredLinear.from_weight(weight, rank=4)
    assert layer.is_dense
    assert layer.stored_values == 24
    value = torch.randn(2, 4)
    assert torch.equal(layer(value), torch.nn.functional.linear(value, weight))


def test_bad_projection_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        StoredLinear()
    native = FakeNative()
    with pytest.raises(ValueError, match="incomplete"):
        TensorPreservingSquaredAttention.from_native(native, ranks={"q": 2})
    program = exact_program(native)
    with pytest.raises(ValueError, match="shape"):
        program(torch.randn(2, 3, 7))
    with pytest.raises(ValueError, match="positive"):
        program.multiply_adds(batch=0, sequence=3)
