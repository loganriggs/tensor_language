from __future__ import annotations

import pytest
import torch

from successor_attention_backend import (
    StoredHeadBlockAttentionBackground,
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


def test_compact_rank_full_candidate_exactly_replays_native_attention() -> None:
    native = FakeNative(width=8, heads=2)
    source = exact_background(native)
    candidate = exact_head_factors(native, 1)
    program = StoredSuccessorAttention(
        source, target_head=1, arm=SuccessorAttentionArm.CANDIDATE,
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
    assert receipt.target_qk_values_used_from_background == 4 * 4 * 8
    assert receipt.target_native_v_stored_values == 0
    assert receipt.target_native_output_stored_values == 0
    assert receipt.unused_target_vo_values_still_stored == 0
    assert receipt.shared_bus_producer_stored_values == 4 * 8
    assert receipt.candidate_circuit_stored_values == 4 * 4 * 8 + 4 * (8 + 4 + 8)
    assert receipt.candidate_circuit_with_shared_bus_producer_stored_values == (
        receipt.candidate_circuit_stored_values
        + receipt.shared_bus_producer_stored_values
    )
    assert receipt.storage_closed


def test_candidate_and_deletion_physically_omit_target_native_vo_blocks() -> None:
    native = FakeNative(width=12, heads=3)
    source = exact_background(native)
    for arm, candidate in (
        (SuccessorAttentionArm.HEAD_DELETED, None),
        (SuccessorAttentionArm.CANDIDATE, exact_head_factors(native, 1)),
    ):
        program = StoredSuccessorAttention(
            source, target_head=1, arm=arm, candidate=candidate,
        )
        assert type(program.background) is StoredHeadBlockAttentionBackground
        background = program.background
        assert tuple(background.q_weight.shape) == (3, 4, 12)
        assert tuple(background.k_weight.shape) == (3, 4, 12)
        assert tuple(background.q2_weight.shape) == (3, 4, 12)
        assert tuple(background.k2_weight.shape) == (3, 4, 12)
        assert tuple(background.other_v_weight.shape) == (2, 4, 12)
        assert tuple(background.other_output_weight.shape) == (12, 2, 4)
        assert background.other_heads.tolist() == [0, 2]
        floating_shapes = [
            tuple(buffer.shape) for buffer in background.buffers()
            if buffer.is_floating_point()
        ]
        assert (3, 4, 12) in floating_shapes
        assert (2, 4, 12) in floating_shapes
        assert (12, 2, 4) in floating_shapes
        assert (4, 12) not in floating_shapes
        assert (12, 4) not in floating_shapes
        receipt = program.receipt()
        actual_float_values = sum(
            buffer.numel() for buffer in program.buffers() if buffer.is_floating_point()
        )
        assert receipt.serialized_stored_values == actual_float_values
        assert receipt.storage_closed


def test_nine_head_target_h7_retains_all_qk_and_exactly_eight_other_vo_blocks() -> None:
    native = FakeNative(width=18, heads=9)
    program = StoredSuccessorAttention(
        exact_background(native), target_head=7,
        arm=SuccessorAttentionArm.HEAD_DELETED,
    )
    assert type(program.background) is StoredHeadBlockAttentionBackground
    background = program.background
    assert tuple(background.q_weight.shape) == (9, 2, 18)
    assert tuple(background.other_v_weight.shape) == (8, 2, 18)
    assert tuple(background.other_output_weight.shape) == (18, 8, 2)
    assert background.other_heads.tolist() == [0, 1, 2, 3, 4, 5, 6, 8]
    receipt = program.receipt()
    assert receipt.background_stored_values == 4 * 9 * 2 * 18 + 2 * 8 * 2 * 18 + 2
    assert receipt.target_qk_values_used_from_background == 4 * 2 * 18
    assert receipt.shared_bus_producer_stored_values == 2 * 18


def test_deleted_head_matches_literal_zero_of_unprojected_head() -> None:
    native = FakeNative(width=8, heads=2)
    source = exact_background(native)
    program = StoredSuccessorAttention(
        source, target_head=1, arm=SuccessorAttentionArm.HEAD_DELETED,
    )
    state = torch.randn(1, 4, 8)
    saved = torch.randn(1, 4, 2, 4)
    _, heads, _ = source.contract_heads(state, saved)
    # The reference may mask densely; the deployed compact module does not.
    reference_heads = heads.clone()
    reference_heads[:, :, 1] = 0
    expected = source.project_heads(reference_heads)
    actual, returned = program(state, saved)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
    assert returned.data_ptr() == saved.data_ptr()


def test_compact_program_calls_no_native_projection_and_owns_source_clones() -> None:
    native = FakeNative(width=8, heads=2)
    source = exact_background(native)
    program = StoredSuccessorAttention(
        source, target_head=1, arm=SuccessorAttentionArm.CANDIDATE,
        candidate=exact_head_factors(native, 1),
    )
    state = torch.randn(1, 4, 8)
    saved = torch.randn(1, 4, 2, 4)
    expected, _ = program(state, saved)
    for layer in native.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("native projection called")
            )
    for layer in source.projections.values():
        assert layer.weight is not None
        layer.weight.fill_(float("nan"))
    actual, _ = program(state, saved)
    assert torch.equal(actual, expected)


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
    receipt = program.receipt()
    assert receipt.storage_closed
    assert receipt.target_native_v_stored_values == 4 * 8
    assert receipt.target_native_output_stored_values == 4 * 8


def test_backend_rejects_wrong_saved_dimension_inconsistent_arm_and_missing_bus() -> None:
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
    program = StoredSuccessorAttention(
        background, target_head=1, arm=SuccessorAttentionArm.HEAD_DELETED,
    )
    with pytest.raises(ValueError, match="requires the shared"):
        program(torch.randn(1, 3, 8), None)


def test_compact_background_rejects_factored_source() -> None:
    native = FakeNative(width=8, heads=2)
    ranks = {name: None for name in PROJECTION_NAMES}
    ranks["v"] = 2
    factored = TensorPreservingSquaredAttention.from_native(native, ranks=ranks)
    with pytest.raises(ValueError, match="exact dense"):
        StoredSuccessorAttention(
            factored, target_head=1, arm=SuccessorAttentionArm.HEAD_DELETED,
        )


def test_contract_heads_rejects_malformed_projection_input() -> None:
    background = exact_background(FakeNative(width=8, heads=2))
    with pytest.raises(ValueError, match="wrong shape"):
        background.project_heads(torch.randn(1, 3, 8))
