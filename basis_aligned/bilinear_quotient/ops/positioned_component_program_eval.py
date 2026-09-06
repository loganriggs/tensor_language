"""Exact full-sequence capture and paired-position component patching.

The fast-screen backend caches only the declared semantic position.  Upstream
writer experiments instead need to intervene where the contextual carrier is
formed.  This module keeps that distinction explicit: every row supplies a
recipient bank and a donor bank, and only paired positions in those banks are
replaced.

Attention components are captured and patched before ``c_proj`` so individual
head slices remain exact.  MLP components are captured and patched at the MLP
output.  Multiple components are installed in one forward and therefore execute
in the model's causal layer order.
"""

# BQGATE: LIBRARY
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import circuit_fast_screen_producer as producer


Kind = Literal["attention_heads", "mlp"]


class PositionedComponentError(ValueError):
    """A positioned component program is malformed or cannot be executed."""


@dataclass(frozen=True, order=True)
class Component:
    kind: Kind
    layer: int
    heads: tuple[int, ...] = ()

    @property
    def site_id(self) -> str:
        if self.kind == "mlp":
            return f"mlp:{self.layer:02d}"
        # The full pre-projection tensor is shared by every subset of heads at
        # this layer; the intervention mask, not the cache key, selects slices.
        return f"attn:{self.layer:02d}:preprojection"


PositionBanks = tuple[tuple[int, ...], ...]


def validate_components(components: Sequence[Component], *, layers: int, heads: int) -> tuple[Component, ...]:
    selected = tuple(components)
    if not selected or len(selected) != len(set(selected)):
        raise PositionedComponentError("component program must be nonempty and unique")
    occupied: set[tuple[str, int]] = set()
    for component in selected:
        if component.kind not in {"attention_heads", "mlp"} or not 0 <= component.layer < layers:
            raise PositionedComponentError("component kind or layer is invalid")
        if component.kind == "mlp":
            if component.heads:
                raise PositionedComponentError("MLP components cannot declare heads")
        elif (not component.heads or len(component.heads) != len(set(component.heads))
              or any(type(head) is not int or not 0 <= head < heads for head in component.heads)):
            raise PositionedComponentError("attention heads must be nonempty, unique, and in range")
        key = (component.kind, component.layer)
        if key in occupied:
            raise PositionedComponentError("a module can appear only once in a component program")
        occupied.add(key)
    return selected


def validate_position_banks(
    recipient_lengths: Sequence[int],
    donor_lengths: Sequence[int],
    recipient_banks: Sequence[Sequence[int]],
    donor_banks: Sequence[Sequence[int]],
) -> tuple[PositionBanks, PositionBanks]:
    if not (len(recipient_lengths) == len(donor_lengths) == len(recipient_banks) == len(donor_banks)):
        raise PositionedComponentError("row and position-bank counts differ")
    recipients, donors = [], []
    for recipient_length, donor_length, recipient, donor in zip(
        recipient_lengths, donor_lengths, recipient_banks, donor_banks
    ):
        rbank, dbank = tuple(recipient), tuple(donor)
        if (not rbank or len(rbank) != len(dbank)
                or len(rbank) != len(set(rbank)) or len(dbank) != len(set(dbank))
                or any(type(position) is not int or not 0 <= position < recipient_length for position in rbank)
                or any(type(position) is not int or not 0 <= position < donor_length for position in dbank)):
            raise PositionedComponentError("position banks must be nonempty, aligned, unique, and in range")
        recipients.append(rbank)
        donors.append(dbank)
    return tuple(recipients), tuple(donors)


def _replace_tensor_positions(value, donor, recipient_banks: PositionBanks, donor_banks: PositionBanks):
    if value.ndim != 3 or donor.ndim != 3 or value.shape[0] != donor.shape[0] \
            or value.shape[0] != len(recipient_banks) or value.shape[2] != donor.shape[2]:
        raise PositionedComponentError("component tensors have incompatible batch or width")
    changed = value.clone()
    for row, (recipient, source) in enumerate(zip(recipient_banks, donor_banks)):
        for recipient_position, donor_position in zip(recipient, source):
            changed[row, recipient_position] = donor[row, donor_position].to(
                device=value.device, dtype=value.dtype
            )
    return changed


def _replace_head_positions(
    value, donor, recipient_banks: PositionBanks, donor_banks: PositionBanks,
    selected_heads: Sequence[int], n_head: int,
):
    if value.ndim != 3 or donor.ndim != 3 or value.shape[2] % n_head:
        raise PositionedComponentError("attention pre-projection tensors have invalid shape")
    if value.shape[0] != donor.shape[0] or value.shape[2] != donor.shape[2] \
            or value.shape[0] != len(recipient_banks):
        raise PositionedComponentError("attention tensors have incompatible batch or width")
    head_width = value.shape[2] // n_head
    changed = value.clone()
    for row, (recipient, source) in enumerate(zip(recipient_banks, donor_banks)):
        for recipient_position, donor_position in zip(recipient, source):
            for head in selected_heads:
                start, stop = head * head_width, (head + 1) * head_width
                changed[row, recipient_position, start:stop] = donor[
                    row, donor_position, start:stop
                ].to(device=value.device, dtype=value.dtype)
    return changed


def capture_full_components(
    backend: producer.Bilin18TorchBackend,
    batch: producer.ModelBatch,
    components: Sequence[Component],
) -> tuple[producer.BatchOutput, Mapping[str, object]]:
    """Capture full sequence tensors for every declared component in one pass."""
    selected = validate_components(
        components, layers=len(backend.model.transformer.h), heads=backend.model.config.n_head
    )
    captured: dict[str, object] = {}
    handles = []

    def save(site_id):
        def hook(_module, _arguments, output):
            if site_id in captured:
                raise PositionedComponentError(f"component executed twice: {site_id}")
            captured[site_id] = output.detach().clone()
        return hook

    def save_input(site_id):
        def hook(_module, arguments):
            if site_id in captured:
                raise PositionedComponentError(f"component executed twice: {site_id}")
            captured[site_id] = arguments[0].detach().clone()
        return hook

    for component in selected:
        block = backend.model.transformer.h[component.layer]
        if component.kind == "attention_heads":
            handles.append(block.attn.c_proj.register_forward_pre_hook(save_input(component.site_id)))
        else:
            handles.append(block.mlp.register_forward_hook(save(component.site_id)))
    try:
        output = backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {component.site_id for component in selected}:
        raise PositionedComponentError("full component capture was incomplete")
    return output, captured


def patch_positioned_components(
    backend: producer.Bilin18TorchBackend,
    batch: producer.ModelBatch,
    donor_batch: producer.ModelBatch,
    components: Sequence[Component],
    donor_cache: Mapping[str, object],
    recipient_banks: Sequence[Sequence[int]],
    donor_banks: Sequence[Sequence[int]],
) -> producer.BatchOutput:
    """Replace declared components only at aligned contextual positions."""
    selected = validate_components(
        components, layers=len(backend.model.transformer.h), heads=backend.model.config.n_head
    )
    if batch.row_ids != donor_batch.row_ids:
        raise PositionedComponentError("recipient and donor row identities differ")
    if any(component.site_id not in donor_cache for component in selected):
        raise PositionedComponentError("donor cache is missing a declared component")
    recipients, donors = validate_position_banks(
        tuple(len(row) for row in batch.token_rows),
        tuple(len(row) for row in donor_batch.token_rows),
        recipient_banks, donor_banks,
    )
    # Every cached component must agree on batch, sequence length, and hidden width.
    reference_shape = tuple(donor_cache[selected[0].site_id].shape)
    if len(reference_shape) != 3 or reference_shape[0] != len(batch.row_ids) or any(
        tuple(donor_cache[component.site_id].shape) != reference_shape for component in selected
    ):
        raise PositionedComponentError("donor component cache shapes disagree")
    handles = []
    n_head = backend.model.config.n_head

    def patch_heads(component):
        def hook(_module, arguments):
            return (_replace_head_positions(
                arguments[0], donor_cache[component.site_id], recipients, donors,
                component.heads, n_head,
            ),) + tuple(arguments[1:])
        return hook

    def patch_mlp(component):
        def hook(_module, _arguments, output):
            return _replace_tensor_positions(
                output, donor_cache[component.site_id], recipients, donors
            )
        return hook

    for component in selected:
        block = backend.model.transformer.h[component.layer]
        if component.kind == "attention_heads":
            handles.append(block.attn.c_proj.register_forward_pre_hook(patch_heads(component)))
        else:
            handles.append(block.mlp.register_forward_hook(patch_mlp(component)))
    try:
        return backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()
