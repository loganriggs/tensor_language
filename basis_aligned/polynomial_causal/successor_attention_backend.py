"""Zero-native-call stored attention with one fixed successor-head replacement.

Candidate and deletion programs own a physically compact background: all score
projections are retained, while the target head's native V rows and output columns
are absent from the module state. The compact shapes, rather than dense zero masks,
are the storage certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

import circuit_successor_tensor as successor
from tensor_preserving_attention import QK_NAMES, TensorPreservingSquaredAttention


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
    target_qk_values_used_from_background: int
    unused_target_vo_values_still_stored: int
    native_calls_per_forward: int
    shared_value_bus: bool
    storage_closed: bool
    # Defaults preserve construction of historical receipts while new programs always
    # populate these exact storage-accounting fields.
    serialized_stored_values: int = 0
    shared_bus_producer_stored_values: int = 0
    candidate_circuit_stored_values: int = 0
    candidate_circuit_with_shared_bus_producer_stored_values: int = 0
    target_native_v_stored_values: int = 0
    target_native_output_stored_values: int = 0


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


class StoredHeadBlockAttentionBackground(nn.Module):
    """Exact other-head replay with no target-head native V/O storage.

    Q/K/Q2/K2 are stored as explicit ``[head, head_dim, width]`` blocks because the
    candidate still needs the target score pattern. V and c_proj are stored only for
    ``other_heads``. No zero target block is registered or constructed.
    """

    def __init__(
        self,
        qk_weights: dict[str, torch.Tensor],
        other_v_weight: torch.Tensor,
        other_output_weight: torch.Tensor,
        *,
        lamb: torch.Tensor,
        inv_freq: torch.Tensor,
        target_head: int,
    ) -> None:
        super().__init__()
        if set(qk_weights) != set(QK_NAMES):
            raise ValueError("compact background QK projection set is incomplete")
        shapes = {tuple(value.shape) for value in qk_weights.values()}
        if len(shapes) != 1:
            raise ValueError("compact background QK shapes disagree")
        n_head, head_dim, width = next(iter(shapes))
        if min(n_head, head_dim, width) <= 0 or n_head * head_dim != width:
            raise ValueError("compact background topology is malformed")
        if type(target_head) is not int or not 0 <= target_head < n_head:
            raise ValueError("target head is outside the compact topology")
        expected_v = (n_head - 1, head_dim, width)
        expected_output = (width, n_head - 1, head_dim)
        tensors = tuple(qk_weights.values()) + (other_v_weight, other_output_weight)
        if (
            tuple(other_v_weight.shape) != expected_v
            or tuple(other_output_weight.shape) != expected_output
            or any(
                not value.is_floating_point() or not bool(torch.isfinite(value).all())
                for value in tensors
            )
        ):
            raise ValueError("compact background V/O blocks are malformed")
        scalar_lamb = torch.as_tensor(lamb)
        if (
            scalar_lamb.numel() != 1 or not scalar_lamb.is_floating_point()
            or not bool(torch.isfinite(scalar_lamb).all())
        ):
            raise ValueError("compact background lambda is malformed")
        if (
            inv_freq.ndim != 1 or inv_freq.numel() * 2 != head_dim
            or not inv_freq.is_floating_point() or not bool(torch.isfinite(inv_freq).all())
        ):
            raise ValueError("compact background rotary frequencies are malformed")
        for name in QK_NAMES:
            self.register_buffer(f"{name}_weight", qk_weights[name].detach().clone())
        self.register_buffer("other_v_weight", other_v_weight.detach().clone())
        self.register_buffer("other_output_weight", other_output_weight.detach().clone())
        self.register_buffer("lamb", scalar_lamb.detach().clone().reshape(()))
        self.register_buffer("inv_freq", inv_freq.detach().clone())
        other_heads = [head for head in range(n_head) if head != target_head]
        self.register_buffer("other_heads", torch.tensor(other_heads, dtype=torch.int64))
        self.width = width
        self.n_head = n_head
        self.head_dim = head_dim
        self.target_head = target_head

    @classmethod
    def from_full(
        cls, background: TensorPreservingSquaredAttention, *, target_head: int,
    ) -> "StoredHeadBlockAttentionBackground":
        if type(background) is not TensorPreservingSquaredAttention:
            raise TypeError("background must be one exact stored attention program")
        if background.shared_qk is not None or background.head_weights is not None:
            raise ValueError("compact replay requires unshared, unmasked exact projections")
        if any(background.projections[name].weight is None for name in (*QK_NAMES, "v", "proj")):
            raise ValueError("compact replay requires exact dense source projections")
        if type(target_head) is not int or not 0 <= target_head < background.n_head:
            raise ValueError("target head is outside the stored attention topology")
        width, n_head, head_dim = background.width, background.n_head, background.head_dim
        other_heads = [head for head in range(n_head) if head != target_head]

        def head_blocks(name: str) -> torch.Tensor:
            weight = background.projections[name].weight
            assert weight is not None
            return weight.detach().reshape(n_head, head_dim, width)

        proj = background.projections["proj"].weight
        assert proj is not None
        return cls(
            {name: head_blocks(name) for name in QK_NAMES},
            head_blocks("v")[other_heads],
            proj.detach().reshape(width, n_head, head_dim)[:, other_heads],
            lamb=background.lamb.detach(),
            inv_freq=background.inv_freq.detach(),
            target_head=target_head,
        )

    @property
    def qk_stored_values(self) -> int:
        return sum(getattr(self, f"{name}_weight").numel() for name in QK_NAMES)

    @property
    def vo_stored_values(self) -> int:
        return self.other_v_weight.numel() + self.other_output_weight.numel()

    @property
    def stored_values(self) -> int:
        # Integer head indices are topology, not learned floating scalars.
        return (
            self.qk_stored_values + self.vo_stored_values
            + self.lamb.numel() + self.inv_freq.numel()
        )

    def _rotary(self, value: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(value.shape[1], device=value.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq.to(value.device))
        cosine = angles.cos().bfloat16()[None, :, None, :]
        sine = angles.sin().bfloat16()[None, :, None, :]
        half = value.shape[-1] // 2
        first, second = value[..., :half], value[..., half:]
        return torch.cat(
            [first * cosine + second * sine, first * (-sine) + second * cosine],
            dim=-1,
        ).to(value.dtype)

    def forward(
        self, state: torch.Tensor, first_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 3 or state.shape[-1] != self.width:
            raise ValueError("attention state shape changed")
        batch, sequence, _ = state.shape
        expected_bus = (batch, sequence, self.n_head, self.head_dim)
        if (
            not torch.is_tensor(first_value) or tuple(first_value.shape) != expected_bus
            or first_value.dtype != state.dtype or first_value.device != state.device
            or not bool(torch.isfinite(first_value.detach()).all())
        ):
            raise ValueError("first-value bus is malformed")

        def project(name: str) -> torch.Tensor:
            weight = getattr(self, f"{name}_weight").to(
                dtype=state.dtype, device=state.device,
            )
            return torch.einsum("bti,hdi->bthd", state, weight)

        query, key = project("q"), project("k")
        query2, key2 = project("q2"), project("k2")
        query = self._rotary(F.rms_norm(query, (self.head_dim,)))
        key = self._rotary(F.rms_norm(key, (self.head_dim,)))
        query2 = self._rotary(F.rms_norm(query2, (self.head_dim,)))
        key2 = self._rotary(F.rms_norm(key2, (self.head_dim,)))
        pattern = (
            torch.einsum("bqhd,bkhd->bhqk", query, key) / self.head_dim
        ) * (
            torch.einsum("bqhd,bkhd->bhqk", query2, key2) / self.head_dim
        )
        causal = torch.tril(torch.ones(
            sequence, sequence, dtype=torch.bool, device=state.device,
        ))
        pattern = pattern.masked_fill(~causal, 0.0)

        other_heads = self.other_heads.to(device=state.device)
        v_weight = self.other_v_weight.to(dtype=state.dtype, device=state.device)
        current_value = torch.einsum("bti,hdi->bthd", state, v_weight)
        saved_value = first_value.index_select(2, other_heads)
        mixed = (1 - self.lamb) * current_value + self.lamb * saved_value
        other_pattern = pattern.index_select(1, other_heads)
        other_write = torch.einsum(
            "bhqk,bkhd->bqhd", other_pattern.to(mixed.dtype), mixed,
        )
        output_weight = self.other_output_weight.to(
            dtype=state.dtype, device=state.device,
        )
        write = torch.einsum("bthd,ohd->bto", other_write, output_weight)
        return pattern, write, first_value


class StoredSuccessorAttention(nn.Module):
    """Replay a stored attention site and optionally replace one head's full write.

    The saved source is the already-projected first-value head slice ``[B,T,d_head]``.
    Candidate/deletion arms own no target-head native V rows or c_proj columns. Other
    heads and the target score pattern are evaluated by the compact stored background.
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
        self.target_head = target_head
        self.arm = arm
        self.candidate = (
            None if candidate is None else StoredSuccessorFactors(
                candidate.current_right, candidate.saved_right, candidate.output,
            )
        )
        if arm is SuccessorAttentionArm.FULL_REPLAY:
            self.background: nn.Module = background
        else:
            self.background = StoredHeadBlockAttentionBackground.from_full(
                background, target_head=target_head,
            )

    @property
    def width(self) -> int:
        return int(self.background.width)

    @property
    def head_dim(self) -> int:
        return int(self.background.head_dim)

    def forward(
        self, state: torch.Tensor, first_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.arm is SuccessorAttentionArm.FULL_REPLAY:
            assert isinstance(self.background, TensorPreservingSquaredAttention)
            return self.background(state, first_value)
        if first_value is None:
            raise ValueError("successor replacement requires the shared first-value bus")
        assert isinstance(self.background, StoredHeadBlockAttentionBackground)
        pattern, write, bus = self.background(state, first_value)
        if self.arm is SuccessorAttentionArm.HEAD_DELETED:
            return write, bus
        assert self.candidate is not None
        saved = bus[:, :, self.target_head]
        candidate_write = self.candidate(pattern[:, self.target_head], state, saved)
        return write + candidate_write.to(write.dtype), bus

    def receipt(self) -> SuccessorBackendReceipt:
        if self.arm is SuccessorAttentionArm.FULL_REPLAY:
            assert isinstance(self.background, TensorPreservingSquaredAttention)
            background_values = self.background.cost_receipt().total_stored_values
            target_qk = 4 * self.background.head_dim * self.background.width
        else:
            assert isinstance(self.background, StoredHeadBlockAttentionBackground)
            background_values = self.background.stored_values
            target_qk = 4 * self.background.head_dim * self.background.width
        candidate_values = 0 if self.candidate is None else self.candidate.stored_values
        serialized = background_values + candidate_values
        bus_producer = successor.shared_bus_producer_parameter_count(
            self.width, self.head_dim,
        )
        return SuccessorBackendReceipt(
            arm=self.arm.value,
            target_head=self.target_head,
            candidate_rank=0 if self.candidate is None else self.candidate.rank,
            candidate_stored_values=candidate_values,
            background_stored_values=background_values,
            serialized_stored_values=serialized,
            shared_bus_producer_stored_values=bus_producer,
            candidate_circuit_stored_values=target_qk + candidate_values,
            candidate_circuit_with_shared_bus_producer_stored_values=(
                target_qk + candidate_values + bus_producer
            ),
            target_qk_values_used_from_background=target_qk,
            target_native_v_stored_values=(
                self.head_dim * self.width
                if self.arm is SuccessorAttentionArm.FULL_REPLAY else 0
            ),
            target_native_output_stored_values=(
                self.head_dim * self.width
                if self.arm is SuccessorAttentionArm.FULL_REPLAY else 0
            ),
            unused_target_vo_values_still_stored=0,
            native_calls_per_forward=0,
            shared_value_bus=True,
            storage_closed=True,
        )


__all__ = [
    "StoredHeadBlockAttentionBackground",
    "StoredSuccessorAttention",
    "StoredSuccessorFactors",
    "SuccessorAttentionArm",
    "SuccessorBackendReceipt",
]
