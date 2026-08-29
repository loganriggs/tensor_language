"""Physical candidate dispatcher for the frozen E4 attention-only copy screen.

This module maps each registered candidate to exact (layer, head) interventions and
constructs the mean-ablated physical attention write

    native_full_write - selected_head_write + fit_role_mean_write.

It owns the attention adapters and mean bank.  It does not load a model, rows, a
checkpoint, or outcomes, and it does not select candidates or thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn as nn

from terminal_copy_attention_adapter import (
    HeadWriteClosure,
    OwnedPerHeadTensorAttention,
)
from terminal_copy_induction_v1 import (
    NAMED_SIX_HEAD_FAMILY,
    REGISTERED_FOUR_HEAD_SET,
    REGISTERED_LATE_PAIR,
)
from terminal_copy_streaming_statistics import FROZEN_CANDIDATES


NAMED_LAYERS = (5, 7, 8, 13, 14)


def _parse_head(name: str) -> tuple[int, int]:
    if not isinstance(name, str) or not name.startswith("L") or "H" not in name:
        raise ValueError("registered head name is malformed")
    layer_text, head_text = name[1:].split("H", 1)
    if not layer_text.isdigit() or not head_text.isdigit():
        raise ValueError("registered head name is malformed")
    return int(layer_text), int(head_text)


def _group(names: tuple[str, ...]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    grouped: dict[int, list[int]] = {}
    for name in names:
        layer, head = _parse_head(name)
        grouped.setdefault(layer, []).append(head)
    return tuple(
        (layer, tuple(sorted(heads))) for layer, heads in sorted(grouped.items())
    )


_plans = {name: _group((name,)) for name in NAMED_SIX_HEAD_FAMILY}
_plans.update({
    "registered_four_head_set": _group(REGISTERED_FOUR_HEAD_SET),
    "registered_late_pair": _group(REGISTERED_LATE_PAIR),
})
if set(_plans) != set(FROZEN_CANDIDATES):
    raise RuntimeError("candidate dispatcher and statistical bank disagree")
FROZEN_HEAD_PLANS: Mapping[str, tuple[tuple[int, tuple[int, ...]], ...]] = (
    MappingProxyType(_plans)
)


@dataclass(frozen=True)
class DispatchedAttentionWrite:
    candidate: str
    layer: int
    heads: tuple[int, ...]
    write: torch.Tensor
    first_value_bus: torch.Tensor
    closure: HeadWriteClosure


class PhysicalCandidateDispatcher(nn.Module):
    """Own the five per-head adapters and position-wise fit-role mean bank."""

    def __init__(
        self,
        *,
        adapters: Mapping[int, OwnedPerHeadTensorAttention],
        per_head_position_means: Mapping[int, torch.Tensor],
    ) -> None:
        super().__init__()
        if set(adapters) != set(NAMED_LAYERS) or set(per_head_position_means) != set(
            NAMED_LAYERS
        ):
            raise ValueError("dispatcher requires the exact five registered layers")
        if any(
            type(layer) is not int
            or not isinstance(adapter, OwnedPerHeadTensorAttention)
            for layer, adapter in adapters.items()
        ):
            raise ValueError("dispatcher adapters are malformed")
        widths = {adapter.width for adapter in adapters.values()}
        head_counts = {adapter.n_head for adapter in adapters.values()}
        if len(widths) != 1 or len(head_counts) != 1:
            raise ValueError("registered attention adapters disagree in topology")
        self.width = widths.pop()
        self.n_head = head_counts.pop()
        self.adapters = nn.ModuleDict({str(layer): adapters[layer] for layer in NAMED_LAYERS})
        lengths: set[int] = set()
        for layer in NAMED_LAYERS:
            mean = per_head_position_means[layer]
            if (
                not torch.is_tensor(mean)
                or not mean.is_floating_point()
                or mean.ndim != 3
                or mean.shape[1:] != (self.n_head, self.width)
                or mean.shape[0] <= 0
                or not bool(torch.isfinite(mean).all())
            ):
                raise ValueError("per-head position mean bank is malformed")
            lengths.add(int(mean.shape[0]))
            self.register_buffer(f"position_mean_{layer}", mean.detach().clone())
        if len(lengths) != 1:
            raise ValueError("position mean banks have inconsistent sequence lengths")
        self.maximum_sequence = lengths.pop()
        for plan in FROZEN_HEAD_PLANS.values():
            if any(
                layer not in NAMED_LAYERS
                or any(not 0 <= head < self.n_head for head in heads)
                for layer, heads in plan
            ):
                raise ValueError("frozen candidate exceeds adapter topology")

    @staticmethod
    def plan(candidate: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
        try:
            return FROZEN_HEAD_PLANS[candidate]
        except (KeyError, TypeError) as error:
            raise ValueError("candidate is outside the frozen bank") from error

    @classmethod
    def from_native(
        cls,
        *,
        attentions: Mapping[int, nn.Module],
        per_head_position_means: Mapping[int, torch.Tensor],
    ) -> "PhysicalCandidateDispatcher":
        if set(attentions) != set(NAMED_LAYERS):
            raise ValueError("native attention bank differs from registered layers")
        return cls(
            adapters={
                layer: OwnedPerHeadTensorAttention.from_native(attentions[layer])
                for layer in NAMED_LAYERS
            },
            per_head_position_means=per_head_position_means,
        )

    def _heads_at(self, candidate: str, layer: int) -> tuple[int, ...]:
        if type(layer) is not int:
            raise ValueError("layer must be an integer")
        by_layer = dict(self.plan(candidate))
        if layer not in by_layer:
            raise ValueError("candidate does not intervene at this layer")
        return by_layer[layer]

    def dispatch(
        self,
        *,
        candidate: str,
        layer: int,
        state: torch.Tensor,
        first_value: torch.Tensor | None,
    ) -> DispatchedAttentionWrite:
        """Return one physical mean-ablation write on the candidate's live state."""

        heads = self._heads_at(candidate, layer)
        if state.ndim != 3 or state.shape[-1] != self.width or (
            state.shape[1] > self.maximum_sequence
        ):
            raise ValueError("candidate attention state is malformed")
        adapter = self.adapters[str(layer)]
        with adapter.begin(state, first_value) as transaction:
            selected = transaction.select(heads)
            native = transaction.native_full_write()
            bus = transaction.first_value_bus()
        mean_bank = getattr(self, f"position_mean_{layer}")
        mean = mean_bank[: state.shape[1], heads, :].sum(1).to(
            device=native.device, dtype=native.dtype,
        )
        write = native - selected + mean.unsqueeze(0)
        if not bool(torch.isfinite(write).all()):
            raise RuntimeError("candidate attention write is nonfinite")
        return DispatchedAttentionWrite(
            candidate=candidate,
            layer=layer,
            heads=heads,
            write=write.clone(),
            first_value_bus=bus.clone(),
            closure=transaction.closure,
        )

    def price(self) -> dict[str, int]:
        adapter_values = sum(
            int(self.adapters[str(layer)].price()["total_stored_values"])
            for layer in NAMED_LAYERS
        )
        mean_values = sum(
            int(getattr(self, f"position_mean_{layer}").numel())
            for layer in NAMED_LAYERS
        )
        return {
            "owned_adapter_values": adapter_values,
            "fit_mean_values": mean_values,
            "total_instrument_values": adapter_values + mean_values,
        }
