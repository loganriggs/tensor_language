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
        native_full_first = transaction.native_full_write()
        observed_bus = transaction.first_value_bus()
    assert torch.equal(native_full_first, expected_first)
    assert torch.allclose(observed_first, expected_first, rtol=1e-6, atol=1e-7)
    assert torch.equal(observed_bus, expected_bus)
    assert transaction.closure.all_head_recomposition_relative_error < 1e-6

    expected_second, _ = native(second_state, expected_bus)
    with program.begin(second_state, observed_bus) as transaction2:
        observed_second = transaction2.all_heads()
        native_full_second = transaction2.native_full_write()
    assert torch.equal(native_full_second, expected_second)
    assert torch.allclose(observed_second, expected_second, rtol=1e-6, atol=1e-7)
    assert transaction2.closure.all_head_recomposition_relative_error < 1e-6


def test_disjoint_head_sums_add_and_returned_tensors_do_not_alias_storage():
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    with program.begin(torch.randn(1, 4, 8)) as transaction:
        head0 = transaction.select((0,))
        head1 = transaction.select((1,))
        full = transaction.native_full_write()
        head0.zero_()
        replay = transaction.select((0,)) + head1
        assert torch.allclose(replay, full, rtol=1e-6, atol=1e-7)
    assert transaction.closure.selected_head_sets == ((0,), (1,), (0,))
    with pytest.raises(RuntimeError, match="closed"):
        transaction.all_heads()


def test_source_writes_sum_to_head_writes_without_materializing_source_tensor():
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    state = torch.randn(2, 5, 8)
    with program.begin(state) as transaction:
        expected = transaction.select((0, 1))
        observed = torch.zeros_like(expected)
        for source in range(state.shape[1]):
            indices = torch.full(
                state.shape[:2], source, dtype=torch.long, device=state.device,
            )
            observed += transaction.source_write((0, 1), indices)
        assert torch.allclose(observed, expected, rtol=1e-6, atol=1e-7)
    assert transaction.closure.selected_head_sets == (
        (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
    )


def test_source_write_is_additive_selective_masked_and_revoked_on_close():
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    state = torch.randn(2, 4, 8)
    sources = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 3]], dtype=torch.long)
    mask = torch.tensor(
        [[False, True, False, True], [True, False, True, False]],
        dtype=torch.bool,
    )
    with program.begin(state) as transaction:
        head0 = transaction.source_write((0,), sources, mask)
        head1 = transaction.source_write((1,), sources, mask)
        both = transaction.source_write((0, 1), sources, mask)
        unmasked = transaction.source_write((0, 1), sources)
        assert torch.allclose(head0 + head1, both, rtol=1e-6, atol=1e-7)
        assert torch.equal(both[~mask], torch.zeros_like(both[~mask]))
        assert torch.allclose(both[mask], unmasked[mask], rtol=1e-6, atol=1e-7)
        both.zero_()
        replay = transaction.source_write((0, 1), sources, mask)
        assert torch.count_nonzero(replay[mask]) > 0
    with pytest.raises(RuntimeError, match="closed"):
        transaction.source_write((0,), sources)


def test_source_write_splits_mixed_route_into_fresh_and_broadcast_terms():
    native = FakeNative()
    program = OwnedPerHeadTensorAttention.from_native(native)
    first_state, state = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    _, bus = native(first_state)
    bus = bus.detach()
    sources = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 2]], dtype=torch.long)
    with program.begin(state, bus) as transaction:
        mixed = transaction.source_write((0, 1), sources, route="mixed")
        fresh = transaction.source_write((0, 1), sources, route="fresh")
        broadcast = transaction.source_write((0, 1), sources, route="broadcast")
        assert torch.allclose(mixed, fresh + broadcast, rtol=1e-6, atol=1e-7)
        with pytest.raises(ValueError, match="route"):
            transaction.source_write((0,), sources, route="unknown")


def test_source_pattern_masks_and_native_override_reconstructs_source_write():
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    state = torch.randn(2, 4, 8)
    sources = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 2]], dtype=torch.long)
    mask = torch.tensor(
        [[False, True, True, False], [True, False, True, True]], dtype=torch.bool,
    )
    with program.begin(state) as transaction:
        pattern = transaction.source_pattern((0, 1), sources, mask)
        native = transaction.source_write((0, 1), sources, mask)
        overridden = transaction.source_write(
            (0, 1), sources, mask, pattern_override=pattern,
        )
        assert pattern.shape == (2, 4, 2)
        assert torch.equal(pattern[~mask], torch.zeros_like(pattern[~mask]))
        assert torch.allclose(overridden, native, rtol=1e-6, atol=1e-7)


def test_source_write_accepts_per_head_constant_pattern_and_rejects_bad_shapes():
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    state = torch.randn(2, 4, 8)
    sources = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 2]], dtype=torch.long)
    constants = torch.tensor([-0.12, 0.19])
    expanded = constants.view(1, 1, 2).expand(2, 4, 2)
    with program.begin(state) as transaction:
        compact = transaction.source_write(
            (0, 1), sources, route="broadcast", pattern_override=constants,
        )
        explicit = transaction.source_write(
            (0, 1), sources, route="broadcast", pattern_override=expanded,
        )
        assert torch.equal(compact, explicit)
        with pytest.raises(ValueError, match="override"):
            transaction.source_write(
                (0, 1), sources, pattern_override=torch.ones(2, 4),
            )
        with pytest.raises(ValueError, match="override"):
            transaction.source_write(
                (0, 1), sources, pattern_override=torch.tensor([float("nan"), 0.0]),
            )


def test_source_write_rejects_malformed_indices_and_masks():
    program = OwnedPerHeadTensorAttention.from_native(FakeNative())
    state = torch.randn(1, 3, 8)
    valid = torch.zeros(1, 3, dtype=torch.long)
    with program.begin(state) as transaction:
        with pytest.raises(ValueError, match="heads"):
            transaction.source_write((), valid)
        with pytest.raises(ValueError, match="indices"):
            transaction.source_write((0,), valid.float())
        with pytest.raises(ValueError, match="indices"):
            transaction.source_write((0,), torch.full_like(valid, 3))
        with pytest.raises(ValueError, match="mask"):
            transaction.source_write((0,), valid, torch.ones(1, 3))


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


def test_clone_preserves_float_rotary_when_projection_weights_are_bfloat16():
    native = FakeNative().to(dtype=torch.bfloat16)
    # FakeNative.rotary is intentionally not an nn.Module, matching the production
    # checkpoint's unregistered float32 inv_freq attribute.
    assert native.c_q.weight.dtype == torch.bfloat16
    assert native.rotary.inv_freq.dtype == torch.float32
    program = OwnedPerHeadTensorAttention.from_native(native)
    assert program.q.dtype == torch.bfloat16
    assert program.inv_freq.dtype == torch.float32


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
