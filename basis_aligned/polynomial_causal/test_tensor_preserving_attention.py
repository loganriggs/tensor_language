from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_preserving_attention import (
    PROJECTION_NAMES, QK_NAMES, SharedInputLinearBank, StoredLinear, TensorAttentionBank,
    TensorPreservingSquaredAttention,
)


class FakeNative(nn.Module):
    def __init__(self, width: int = 8, heads: int = 2) -> None:
        super().__init__()
        torch.manual_seed(9)
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.lamb = nn.Parameter(torch.tensor(0.37))
        self.n_head = heads
        self.n_embd = width
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


def test_fixed_head_slice_projector_is_exact_global_tensor_edit() -> None:
    native = FakeNative()
    state = torch.randn(2, 5, 8)
    first_value = torch.randn(2, 5, 2, 4)
    weights = torch.tensor([1.0, 0.0])
    program = TensorPreservingSquaredAttention.from_native(
        native, ranks={name: None for name in PROJECTION_NAMES},
        head_weights=weights,
    )

    batch, sequence, _ = state.shape
    unprojected = exact_program(native)
    captured = {}
    handle = unprojected.projections["proj"].register_forward_pre_hook(
        lambda _module, args: captured.setdefault("value", args[0].detach().clone())
    )
    unprojected(state, first_value)
    handle.remove()
    expected_heads = captured["value"].view(batch, sequence, 2, 4)
    expected_heads[:, :, 1] = 0
    expected = native.c_proj(expected_heads.reshape(batch, sequence, 8))

    actual, returned_bus = program(state, first_value)
    assert torch.equal(actual, expected)
    assert returned_bus.data_ptr() == first_value.data_ptr()
    receipt = program.cost_receipt()
    assert receipt.projection_values["head_weights"] == 2
    assert receipt.total_stored_values == 6 * 8 * 8 + 1 + 2 + 2


def test_head_slice_projector_contract_fails_closed() -> None:
    native = FakeNative()
    for bad in (torch.ones(3), torch.ones(2, dtype=torch.int64),
                torch.tensor([1.0, float("nan")])):
        with pytest.raises(ValueError, match="head weights"):
            TensorPreservingSquaredAttention.from_native(
                native, ranks={name: None for name in PROJECTION_NAMES},
                head_weights=bad,
            )


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


def test_bank_enforces_order_block_identity_and_one_root_bus() -> None:
    natives = [FakeNative() for _ in range(18)]
    programs = [exact_program(native) for native in natives]
    bank = TensorAttentionBank(programs)
    blocks = [object() for _ in range(18)]
    transaction = bank.begin(blocks)
    state = torch.randn(1, 3, 8)
    bus = None
    for site, block in enumerate(blocks):
        write, next_bus = transaction(SimpleNamespace(
            site=site, block=block, state=state, first_value=bus,
        ))
        assert write.shape == state.shape
        if site == 0:
            bus = next_bus
        else:
            assert next_bus is bus
    closure = transaction.close()
    assert closure.ordered and closure.block_identity and closure.first_value_identity
    assert closure.sites == tuple((site, 1) for site in range(18))
    with pytest.raises(RuntimeError, match="closed"):
        transaction(SimpleNamespace(site=0))
    receipt = bank.cost_receipt()
    assert receipt["native_calls_per_forward"] == 0
    assert receipt["token_table_values"] == 0


def test_bank_rejects_reorder_wrong_block_and_wrong_bus_identity() -> None:
    bank = TensorAttentionBank([exact_program(FakeNative()) for _ in range(2)])
    blocks = [object(), object()]
    with pytest.raises(RuntimeError, match="reordered"):
        with bank.begin(blocks) as transaction:
            transaction(SimpleNamespace(
                site=1, block=blocks[1], state=torch.randn(1, 2, 8), first_value=None,
            ))
    with pytest.raises(RuntimeError, match="wrong block"):
        with bank.begin(blocks) as transaction:
            transaction(SimpleNamespace(
                site=0, block=object(), state=torch.randn(1, 2, 8), first_value=None,
            ))
    with pytest.raises(RuntimeError, match="bus identity"):
        with bank.begin(blocks) as transaction:
            state = torch.randn(1, 2, 8)
            transaction(SimpleNamespace(
                site=0, block=blocks[0], state=state, first_value=None,
            ))
            transaction(SimpleNamespace(
                site=1, block=blocks[1], state=state,
                first_value=torch.randn(1, 2, 2, 4),
            ))


def test_projection_schema_and_native_topology_validation() -> None:
    native = FakeNative()
    native.n_embd = 7
    with pytest.raises(ValueError, match="topology"):
        exact_program(native)
    native = FakeNative()
    native.c_k = nn.Linear(7, 8, bias=False)
    with pytest.raises(ValueError, match="common width"):
        exact_program(native)
    bad = torch.randn(8, 8)
    bad[0, 0] = float("nan")
    projections = {
        name: StoredLinear(weight=bad if name == "q" else torch.randn(8, 8))
        for name in PROJECTION_NAMES
    }
    with pytest.raises(ValueError, match="nonfinite"):
        TensorPreservingSquaredAttention(
            projections, lamb=torch.tensor(0.5), inv_freq=torch.ones(2), n_head=2,
        )


def test_shared_qk_bank_reads_one_basis_and_prices_one_encoder() -> None:
    torch.manual_seed(41)
    width, rank = 8, 3
    basis, _ = torch.linalg.qr(torch.randn(width, rank))
    weights = {name: torch.randn(width, width) for name in QK_NAMES}
    shared = SharedInputLinearBank.from_basis(weights, basis)
    value = torch.randn(2, 4, width)
    for name in QK_NAMES:
        expected = (value @ basis) @ (weights[name] @ basis).T
        assert torch.allclose(shared(name, value), expected, atol=1e-6, rtol=1e-6)
    assert shared.stored_values == 5 * width * rank

    native = FakeNative(width=width, heads=2)
    projections = {
        "v": StoredLinear.from_weight(native.c_v.weight),
        "proj": StoredLinear.from_weight(native.c_proj.weight),
    }
    program = TensorPreservingSquaredAttention(
        projections, lamb=native.lamb, inv_freq=native.rotary.inv_freq,
        n_head=2, shared_qk=shared,
    )
    receipt = program.cost_receipt()
    assert receipt.projection_values["qk_shared"] == 5 * width * rank
    assert receipt.total_stored_values == 5 * width * rank + 2 * width * width + 3
