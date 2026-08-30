from __future__ import annotations

import pytest
import torch

from successor_attention_backend import (
    StoredSuccessorAttention,
    StoredSuccessorFactors,
    SuccessorAttentionArm,
)
from tensor_preserving_attention import PROJECTION_NAMES, TensorPreservingSquaredAttention
from test_tensor_preserving_attention import FakeNative


def exact_background(native: FakeNative) -> TensorPreservingSquaredAttention:
    return TensorPreservingSquaredAttention.from_native(
        native, ranks={name: None for name in PROJECTION_NAMES},
    )


def exact_head_factors(native: FakeNative, head: int) -> StoredSuccessorFactors:
    width = native.n_embd
    head_dim = width // native.n_head
    start, stop = head * head_dim, (head + 1) * head_dim
    current = (1 - native.lamb.detach()) * native.c_v.weight[start:stop]
    saved = native.lamb.detach() * torch.eye(head_dim)
    output = native.c_proj.weight[:, start:stop]
    return StoredSuccessorFactors(current, saved, output)


def test_rank_full_shared_bus_candidate_exactly_replays_native_attention() -> None:
    native = FakeNative(width=8, heads=2)
    background = exact_background(native)
    candidate = exact_head_factors(native, 1)
    program = StoredSuccessorAttention(
        background, target_head=1, arm=SuccessorAttentionArm.CANDIDATE,
        candidate=candidate,
    )
    state = torch.randn(2, 5, 8)
    saved = torch.randn(2, 5, 2, 4)
    expected, expected_bus = native(state, saved)
    actual, bus = program(state, saved)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
    assert bus.data_ptr() == expected_bus.data_ptr() == saved.data_ptr()
    receipt = program.receipt()
    assert receipt.candidate_rank == 4
    assert receipt.candidate_stored_values == 4 * (8 + 4 + 8)
    assert receipt.native_calls_per_forward == 0 and receipt.shared_value_bus


def test_deleted_head_matches_literal_zero_of_unprojected_head() -> None:
    native = FakeNative(width=8, heads=2)
    background = exact_background(native)
    program = StoredSuccessorAttention(
        background, target_head=1, arm=SuccessorAttentionArm.HEAD_DELETED,
    )
    state = torch.randn(1, 4, 8)
    saved = torch.randn(1, 4, 2, 4)
    _, heads, _ = background.contract_heads(state, saved)
    heads[:, :, 1] = 0
    expected = background.project_heads(heads)
    actual, returned = program(state, saved)
    assert torch.equal(actual, expected)
    assert returned.data_ptr() == saved.data_ptr()


def test_full_replay_remains_bit_exact_and_calls_no_native_projection() -> None:
    native = FakeNative(width=8, heads=2)
    background = exact_background(native)
    program = StoredSuccessorAttention(
        background, target_head=1, arm=SuccessorAttentionArm.FULL_REPLAY,
    )
    state = torch.randn(1, 4, 8)
    expected, _ = native(state)
    for layer in native.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("native projection called")
            )
    actual, _ = program(state, None)
    assert torch.equal(actual, expected)


def test_backend_rejects_wrong_saved_dimension_and_inconsistent_arm() -> None:
    native = FakeNative(width=8, heads=2)
    background = exact_background(native)
    bad = StoredSuccessorFactors(torch.randn(3, 8), torch.randn(3, 5), torch.randn(8, 3))
    with pytest.raises(ValueError, match="interfaces"):
        StoredSuccessorAttention(
            background, target_head=1, arm=SuccessorAttentionArm.CANDIDATE,
            candidate=bad,
        )
    with pytest.raises(ValueError, match="required exactly"):
        StoredSuccessorAttention(
            background, target_head=1, arm=SuccessorAttentionArm.CANDIDATE,
        )


def test_contract_heads_rejects_malformed_projection_input() -> None:
    background = exact_background(FakeNative(width=8, heads=2))
    with pytest.raises(ValueError, match="wrong shape"):
        background.project_heads(torch.randn(1, 3, 8))
