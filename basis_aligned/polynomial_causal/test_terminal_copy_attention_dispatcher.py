from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention
from terminal_copy_attention_dispatcher import (
    FROZEN_HEAD_PLANS,
    NAMED_LAYERS,
    PhysicalCandidateDispatcher,
    registered_candidate_arithmetic,
)


class TinyAttention(nn.Module):
    def __init__(self, width: int = 18, heads: int = 9) -> None:
        super().__init__()
        torch.manual_seed(90)
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.lamb = nn.Parameter(torch.tensor(0.25))
        self.n_head = heads
        self.rotary = SimpleNamespace(inv_freq=torch.ones(width // heads // 2))


def make_dispatcher(sequence: int = 7) -> PhysicalCandidateDispatcher:
    adapters = {
        layer: OwnedPerHeadTensorAttention.from_native(TinyAttention())
        for layer in NAMED_LAYERS
    }
    means = {
        layer: torch.zeros(sequence, 2 if layer == 8 else 1, 18)
        for layer in NAMED_LAYERS
    }
    means[5][:, 0] = 0.125
    means[8][:, 0] = 0.25
    means[8][:, 1] = -0.5
    return PhysicalCandidateDispatcher(
        adapters=adapters, per_head_position_means=means,
    )


def test_frozen_candidate_plans_are_exact_and_same_layer_pair_is_atomic():
    assert FROZEN_HEAD_PLANS == {
        "L5H5": ((5, (5,)),),
        "L7H3": ((7, (3,)),),
        "L8H3": ((8, (3,)),),
        "L8H4": ((8, (4,)),),
        "L13H0": ((13, (0,)),),
        "L14H7": ((14, (7,)),),
        "registered_four_head_set": ((5, (5,)), (7, (3,)), (8, (3, 4))),
        "registered_late_pair": ((13, (0,)), (14, (7,))),
    }


def test_dispatch_is_exact_native_minus_selected_plus_fit_mean():
    dispatcher = make_dispatcher()
    state = torch.randn(2, 5, 18)
    adapter = dispatcher.adapters["5"]
    with adapter.begin(state) as transaction:
        native = transaction.native_full_write()
        selected = transaction.select((5,))
        bus = transaction.first_value_bus()
    result = dispatcher.dispatch(
        candidate="L5H5", layer=5, state=state, first_value=None,
    )
    expected = native - selected + 0.125
    assert torch.equal(result.write, expected)
    assert torch.equal(result.first_value_bus, bus)
    assert result.heads == (5,)
    assert result.closure.closed


def test_two_heads_in_one_layer_use_one_transaction_and_sum_their_means():
    dispatcher = make_dispatcher()
    state = torch.randn(1, 6, 18)
    result = dispatcher.dispatch(
        candidate="registered_four_head_set", layer=8,
        state=state, first_value=None,
    )
    adapter = dispatcher.adapters["8"]
    with adapter.begin(state) as transaction:
        expected = transaction.native_full_write() - transaction.select((3, 4)) - 0.25
    assert torch.equal(result.write, expected)
    assert result.closure.selected_head_sets == ((3, 4),)


def test_dispatch_preserves_existing_first_value_bus_and_returns_no_alias():
    dispatcher = make_dispatcher()
    state = torch.randn(2, 4, 18)
    first_value = torch.randn(2, 4, 9, 2)
    result = dispatcher.dispatch(
        candidate="registered_late_pair", layer=13,
        state=state, first_value=first_value,
    )
    assert torch.equal(result.first_value_bus, first_value)
    result.first_value_bus.zero_()
    assert not torch.equal(result.first_value_bus, first_value)


def test_dispatch_fails_closed_outside_candidate_plan_or_sequence_bank():
    dispatcher = make_dispatcher(sequence=4)
    with pytest.raises(ValueError, match="outside"):
        dispatcher.plan("not_registered")
    with pytest.raises(ValueError, match="does not intervene"):
        dispatcher.dispatch(
            candidate="L5H5", layer=7, state=torch.randn(1, 4, 18), first_value=None,
        )
    with pytest.raises(ValueError, match="malformed"):
        dispatcher.dispatch(
            candidate="L5H5", layer=5, state=torch.randn(1, 5, 18), first_value=None,
        )


def test_production_dispatch_rejects_synthetic_shape_and_missing_value_bus():
    dispatcher = make_dispatcher(sequence=4)
    with pytest.raises(ValueError, match="production"):
        dispatcher.dispatch(
            candidate="L5H5", layer=5, state=torch.randn(1, 4, 18),
            first_value=None, require_production=True,
        )


def test_bfloat_candidate_arithmetic_is_not_reassociated():
    native = torch.tensor(52.75, dtype=torch.bfloat16)
    selected = torch.tensor(67.5, dtype=torch.bfloat16)
    mean = torch.tensor(-69.0, dtype=torch.bfloat16)
    registered = registered_candidate_arithmetic(native, selected, mean)
    reassociated = native + (-selected + mean)
    assert registered.item() == -84.0
    assert reassociated.item() == -83.0


def test_constructor_rejects_missing_layers_and_bad_mean_topology():
    adapters = {
        layer: OwnedPerHeadTensorAttention.from_native(TinyAttention())
        for layer in NAMED_LAYERS
    }
    means = {
        layer: torch.zeros(4, 2 if layer == 8 else 1, 18)
        for layer in NAMED_LAYERS
    }
    with pytest.raises(ValueError, match="exact five"):
        PhysicalCandidateDispatcher(
            adapters={5: adapters[5]}, per_head_position_means=means,
        )
    means[14] = torch.zeros(4, 2, 18)
    with pytest.raises(ValueError, match="malformed"):
        PhysicalCandidateDispatcher(
            adapters=adapters, per_head_position_means=means,
        )


def test_dispatcher_prices_owned_adapters_and_full_mean_bank():
    dispatcher = make_dispatcher(sequence=7)
    price = dispatcher.price()
    assert price["fit_mean_values"] == 6 * 7 * 18
    assert price["owned_adapter_values"] == 5 * (6 * 18 * 18 + 2)
    assert price["total_instrument_values"] == (
        price["owned_adapter_values"] + price["fit_mean_values"]
    )
