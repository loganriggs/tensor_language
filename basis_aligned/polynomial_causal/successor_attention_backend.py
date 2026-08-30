"""Zero-native-call stored attention with one fixed successor-head replacement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

import circuit_successor_tensor as successor
from tensor_preserving_attention import TensorPreservingSquaredAttention


class SuccessorAttentionArm(str, Enum):
    FULL_REPLAY = "full_replay"
    HEAD_DELETED = "head_deleted"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class SuccessorBackendReceipt:
    arm: str
    target_head: int
    candidate_rank: int
    candidate_stored_values: int
    background_stored_values: int
    native_calls_per_forward: int
    shared_value_bus: bool


class StoredSuccessorFactors(nn.Module):
    """A physical low-rank map from current state and projected saved value."""

    def __init__(
        self,
        current_right: torch.Tensor,
        saved_right: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        super().__init__()
        values = (current_right, saved_right, output)
        if any(
            not torch.is_tensor(value) or value.ndim != 2
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in values
        ):
            raise ValueError("successor factors must be finite floating matrices")
        rank = current_right.shape[0]
        if rank <= 0 or saved_right.shape[0] != rank or output.shape[1] != rank:
            raise ValueError("successor factors must share one positive rank")
        self.register_buffer("current_right", current_right.detach().clone())
        self.register_buffer("saved_right", saved_right.detach().clone())
        self.register_buffer("output", output.detach().clone())

    @property
    def rank(self) -> int:
        return self.current_right.shape[0]

    @property
    def stored_values(self) -> int:
        return (
            self.current_right.numel()
            + self.saved_right.numel()
            + self.output.numel()
        )

    def forward(
        self,
        scores: torch.Tensor,
        current_state: torch.Tensor,
        saved_value: torch.Tensor,
    ) -> torch.Tensor:
        return successor.two_source_preweighted_write(
            scores,
            current_state,
            saved_value,
            self.current_right.to(dtype=scores.dtype, device=scores.device),
            self.saved_right.to(dtype=scores.dtype, device=scores.device),
            self.output.to(dtype=scores.dtype, device=scores.device),
        )


class StoredSuccessorAttention(nn.Module):
    """Replay a stored attention site and optionally replace one head's full write.

    The saved source is the already-projected first-value head slice `[B,T,d_head]`.
    No token, lexicon, target mask, parser, native score, or native module enters the
    replacement.  Other heads are evaluated by the owned exact background program.
    """

    def __init__(
        self,
        background: TensorPreservingSquaredAttention,
        *,
        target_head: int,
        arm: SuccessorAttentionArm,
        candidate: StoredSuccessorFactors | None = None,
    ) -> None:
        super().__init__()
        if type(background) is not TensorPreservingSquaredAttention:
            raise TypeError("background must be one exact stored attention program")
        if type(target_head) is not int or not 0 <= target_head < background.n_head:
            raise ValueError("target head is outside the stored attention topology")
        if type(arm) is not SuccessorAttentionArm:
            raise TypeError("arm must be a SuccessorAttentionArm")
        if (arm is SuccessorAttentionArm.CANDIDATE) != (candidate is not None):
            raise ValueError("candidate factors are required exactly for candidate arm")
        if candidate is not None and (
            candidate.current_right.shape[1] != background.width
            or candidate.saved_right.shape[1] != background.head_dim
            or candidate.output.shape[0] != background.width
        ):
            raise ValueError("successor factors do not match current/bus/output interfaces")
        if background.head_weights is not None:
            raise ValueError("successor background must not already mask heads")
        self.background = background
        self.target_head = target_head
        self.arm = arm
        self.candidate = candidate

    def forward(
        self, state: torch.Tensor, first_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pattern, native_heads, bus = self.background.contract_heads(state, first_value)
        if self.arm is SuccessorAttentionArm.FULL_REPLAY:
            return self.background.project_heads(native_heads), bus
        other_heads = native_heads.clone()
        other_heads[:, :, self.target_head] = 0
        write = self.background.project_heads(other_heads)
        if self.arm is SuccessorAttentionArm.HEAD_DELETED:
            return write, bus
        assert self.candidate is not None
        saved = bus[:, :, self.target_head]
        candidate_write = self.candidate(pattern[:, self.target_head], state, saved)
        return write + candidate_write.to(write.dtype), bus

    def receipt(self) -> SuccessorBackendReceipt:
        background_values = self.background.cost_receipt().total_stored_values
        candidate_values = 0 if self.candidate is None else self.candidate.stored_values
        return SuccessorBackendReceipt(
            arm=self.arm.value,
            target_head=self.target_head,
            candidate_rank=0 if self.candidate is None else self.candidate.rank,
            candidate_stored_values=candidate_values,
            background_stored_values=background_values,
            native_calls_per_forward=0,
            shared_value_bus=True,
        )


__all__ = [
    "StoredSuccessorAttention",
    "StoredSuccessorFactors",
    "SuccessorAttentionArm",
    "SuccessorBackendReceipt",
]
