from __future__ import annotations

import dataclasses

import pytest

import ordered_successor_tensor_backend_adapter_v1 as adapter
import ordered_successor_tensor_discovery_v1 as discovery
from successor_attention_backend import SuccessorBackendReceipt


def _receipt(
    arm: str, *, rank: int = 0, candidate: int = 0,
    full: bool = False,
) -> SuccessorBackendReceipt:
    background = (
        adapter.PRODUCTION_FULL_BACKGROUND_STORED_VALUES
        if full else adapter.PRODUCTION_COMPACT_BACKGROUND_STORED_VALUES
    )
    candidate_circuit = adapter.PRODUCTION_TARGET_QK_VALUES + candidate
    return SuccessorBackendReceipt(
        arm=arm,
        target_head=7,
        candidate_rank=rank,
        candidate_stored_values=candidate,
        background_stored_values=background,
        target_qk_values_used_from_background=adapter.PRODUCTION_TARGET_QK_VALUES,
        unused_target_vo_values_still_stored=0,
        native_calls_per_forward=0,
        shared_value_bus=True,
        storage_closed=True,
        serialized_stored_values=background + candidate,
        shared_bus_producer_stored_values=(
            adapter.PRODUCTION_SHARED_BUS_PRODUCER_STORED_VALUES
        ),
        candidate_circuit_stored_values=candidate_circuit,
        candidate_circuit_with_shared_bus_producer_stored_values=(
            candidate_circuit + adapter.PRODUCTION_SHARED_BUS_PRODUCER_STORED_VALUES
        ),
        target_native_v_stored_values=147_456 if full else 0,
        target_native_output_stored_values=147_456 if full else 0,
    )


def test_full_replay_and_primary_candidate_bind_two_distinct_prices() -> None:
    full = adapter.binding_from_receipt(
        discovery.FULL_REPLAY, "a" * 64,
        _receipt("full_replay", full=True),
    )
    assert full.stored_parameters == 884_736
    assert full.executable_stored_parameters == 7_962_689
    assert full.storage_closed

    arm = "head8_7_both_r64_true"
    candidate_values = 64 * (1152 + 128 + 1152)
    candidate = adapter.binding_from_receipt(
        arm, "b" * 64,
        _receipt("candidate", rank=64, candidate=candidate_values),
    )
    assert candidate.stored_parameters == 745_472
    assert candidate.candidate_stored_parameters == 155_648
    assert candidate.executable_stored_parameters == 7_823_425
    assert candidate.unused_target_vo_values == 0
    assert candidate.storage_closed


def test_deletion_is_causal_control_not_storage_closed_compression() -> None:
    binding = adapter.binding_from_receipt(
        discovery.HEAD_DELETED, "c" * 64, _receipt("head_deleted"),
    )
    assert binding.stored_parameters == 0
    assert binding.executable_stored_parameters == 7_667_777
    assert binding.storage_closed


def test_source_omission_requires_backend_that_physically_omits_factor() -> None:
    both_source_rank128 = 128 * (1152 + 128 + 1152)
    with pytest.raises(ValueError, match="omission-aware"):
        adapter.binding_from_receipt(
            discovery.CURRENT_ONLY, "d" * 64,
            _receipt("candidate", rank=128, candidate=both_source_rank128),
        )
    with pytest.raises(ValueError, match="omission-aware"):
        adapter.binding_from_receipt(
            discovery.V1_ONLY, "e" * 64,
            _receipt("candidate", rank=128, candidate=both_source_rank128),
        )


def test_receipt_fails_closed_on_rank_qk_and_storage_flags() -> None:
    arm = "head8_7_both_r8_true"
    receipt = _receipt("candidate", rank=8, candidate=8 * 2432)
    with pytest.raises(ValueError, match="rank"):
        adapter.binding_from_receipt(
            arm, "f" * 64, dataclasses.replace(receipt, candidate_rank=16),
        )
    with pytest.raises(ValueError, match="background/QK"):
        adapter.binding_from_receipt(
            arm, "f" * 64,
            dataclasses.replace(receipt, target_qk_values_used_from_background=1),
        )
    with pytest.raises(ValueError, match="storage-closure"):
        adapter.binding_from_receipt(
            arm, "f" * 64, dataclasses.replace(receipt, storage_closed=False),
        )
