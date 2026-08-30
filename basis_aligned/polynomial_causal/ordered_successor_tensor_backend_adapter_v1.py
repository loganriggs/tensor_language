"""Typed bridge from the frozen successor assay to its zero-native backend.

This module owns no model, rows, outcomes, or publication.  It converts an exact
backend receipt into the discovery authority's two-price ledger: target-component
storage and the larger currently materialized executable storage.  Candidate and
deletion arms are explicitly not storage-closed in v1.
"""

from __future__ import annotations

import hashlib
import json

import torch

import circuit_campaign_runtime as campaign
import ordered_successor_tensor_discovery_v1 as discovery
from successor_attention_backend import (
    StoredSuccessorAttention,
    SuccessorAttentionArm,
    SuccessorBackendReceipt,
)


PRODUCTION_FULL_BACKGROUND_STORED_VALUES = discovery.FULL_BACKGROUND_STORED_VALUES
PRODUCTION_TARGET_QK_VALUES = discovery.TARGET_QK_STORED_VALUES
PRODUCTION_TARGET_VO_VALUES = discovery.TARGET_VO_STORED_VALUES
PRODUCTION_COMPACT_BACKGROUND_STORED_VALUES = discovery.COMPACT_BACKGROUND_STORED_VALUES
PRODUCTION_SHARED_BUS_PRODUCER_STORED_VALUES = 1152 * 128


def backend_state_sha256(program: StoredSuccessorAttention) -> str:
    if not isinstance(program, StoredSuccessorAttention):
        raise ValueError("program must be one StoredSuccessorAttention")
    digest = hashlib.sha256()
    for name, value in sorted(program.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _candidate_rank(arm: str) -> int | None:
    matches = tuple(candidate for candidate in discovery.CANDIDATES if candidate.arm == arm)
    if len(matches) == 1:
        return matches[0].rank
    if arm in (discovery.CURRENT_ONLY, discovery.V1_ONLY):
        return discovery.NATIVE_VALUE_RANK
    return None


def binding_from_receipt(
    arm: str, sha256: str, receipt: SuccessorBackendReceipt,
) -> discovery.ProgramBinding:
    """Validate a literal backend receipt and mint no stronger storage claim."""

    if arm not in discovery.ARM_NAMES[1:] or not isinstance(receipt, SuccessorBackendReceipt):
        raise ValueError("backend receipt arm is outside the discovery registry")
    if receipt.target_head != discovery.TARGET_HEAD or receipt.native_calls_per_forward != 0 or (
        not receipt.shared_value_bus
    ):
        raise ValueError("backend physical interface differs from frozen L8H7 shared bus")
    expected_backend_arm = (
        SuccessorAttentionArm.FULL_REPLAY.value if arm == discovery.FULL_REPLAY else
        SuccessorAttentionArm.HEAD_DELETED.value if arm == discovery.HEAD_DELETED else
        SuccessorAttentionArm.CANDIDATE.value
    )
    if receipt.arm != expected_backend_arm:
        raise ValueError("backend arm semantics differ from discovery arm")
    expected_rank = _candidate_rank(arm)
    if receipt.candidate_rank != (0 if expected_rank is None else expected_rank):
        raise ValueError("backend candidate rank differs from the frozen arm")
    expected_background = (
        PRODUCTION_FULL_BACKGROUND_STORED_VALUES
        if arm == discovery.FULL_REPLAY
        else PRODUCTION_COMPACT_BACKGROUND_STORED_VALUES
    )
    if receipt.background_stored_values != expected_background or (
        receipt.target_qk_values_used_from_background != PRODUCTION_TARGET_QK_VALUES
    ):
        raise ValueError("backend dense background/QK storage changed")
    if receipt.unused_target_vo_values_still_stored != 0 or not receipt.storage_closed:
        raise ValueError("backend storage-closure receipt changed")
    expected_candidate = (
        0 if arm in (discovery.FULL_REPLAY, discovery.HEAD_DELETED)
        else discovery.arm_stored_parameters(arm) - PRODUCTION_TARGET_QK_VALUES
    )
    if receipt.candidate_stored_values != expected_candidate:
        if arm in (discovery.CURRENT_ONLY, discovery.V1_ONLY):
            raise ValueError("source-omission arm needs an omission-aware factor backend")
        raise ValueError("backend candidate factor storage differs from frozen target price")
    expected_native_vo_each = 128 * 1152 if arm == discovery.FULL_REPLAY else 0
    if (
        receipt.serialized_stored_values
        != receipt.background_stored_values + receipt.candidate_stored_values
        or receipt.shared_bus_producer_stored_values
        != PRODUCTION_SHARED_BUS_PRODUCER_STORED_VALUES
        or receipt.candidate_circuit_stored_values
        != PRODUCTION_TARGET_QK_VALUES + receipt.candidate_stored_values
        or receipt.candidate_circuit_with_shared_bus_producer_stored_values
        != receipt.candidate_circuit_stored_values
        + PRODUCTION_SHARED_BUS_PRODUCER_STORED_VALUES
        or receipt.target_native_v_stored_values != expected_native_vo_each
        or receipt.target_native_output_stored_values != expected_native_vo_each
    ):
        raise ValueError("backend serialized/producer/target storage receipt changed")
    return discovery.ProgramBinding(
        arm=arm,
        sha256=sha256,
        stored_parameters=discovery.arm_stored_parameters(arm),
        executable_stored_parameters=(
            receipt.serialized_stored_values
        ),
        background_stored_parameters=receipt.background_stored_values,
        candidate_stored_parameters=receipt.candidate_stored_values,
        unused_target_vo_values=receipt.unused_target_vo_values_still_stored,
        storage_closed=receipt.storage_closed,
    )


def binding_from_program(
    arm: str, program: StoredSuccessorAttention,
) -> discovery.ProgramBinding:
    if not isinstance(program, StoredSuccessorAttention):
        raise ValueError("successor adapter requires the exact backend type")
    return binding_from_receipt(arm, backend_state_sha256(program), program.receipt())


def make_attention_replacement(program: StoredSuccessorAttention):
    """One mask-blind attention-8 callback; tokens/lexicons are inaccessible."""

    if not isinstance(program, StoredSuccessorAttention):
        raise ValueError("successor replacement requires the exact backend type")

    def replacement(event: campaign.AttentionReplacementEvent):
        if type(event) is not campaign.AttentionReplacementEvent or event.site != discovery.TARGET_SITE:
            raise ValueError("successor callback received the wrong physical site")
        return program(event.state, event.first_value)

    return replacement


__all__ = (
    "PRODUCTION_COMPACT_BACKGROUND_STORED_VALUES",
    "PRODUCTION_FULL_BACKGROUND_STORED_VALUES",
    "PRODUCTION_SHARED_BUS_PRODUCER_STORED_VALUES", "PRODUCTION_TARGET_QK_VALUES",
    "PRODUCTION_TARGET_VO_VALUES", "backend_state_sha256", "binding_from_program",
    "binding_from_receipt", "make_attention_replacement",
)
