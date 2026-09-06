"""Exact source-term interventions for attention heads at declared destinations."""

# BQGATE: LIBRARY
from __future__ import annotations

import torch
import torch.nn.functional as F

from jacclust.tt_model import apply_rotary_emb, einsum


GROUPS = ("prefix", "cue", "local")
RESPONSE_FACTORS = (
    "pattern_on_base_value",
    "base_pattern_on_value_change",
    "pattern_value_interaction",
)


class AttentionSourceDestinationError(RuntimeError):
    pass


def attention_response_factor_deltas(
    base_capture,
    changed_capture,
    destinations,
    positions_by_row,
    *,
    selected_heads,
):
    """Exact ``P'V' - PV`` decomposition for declared source-to-destination terms.

    Returns three full head-output-shaped delta tensors whose sum is
    ``(P'-P)V + P(V'-V) + (P'-P)(V'-V)`` on the selected terms and zero elsewhere.
    They can be installed singly or jointly by adding them to a copied base capture and using
    ``intervene_ordered_head_output_deltas``.  This is an operation split, not a low-rank fit.
    """
    required = {"pattern", "value", "head_output"}
    if any(not required.issubset(capture) for capture in (base_capture, changed_capture)):
        raise AttentionSourceDestinationError("attention response capture is incomplete")
    if (base_capture["pattern"].shape != changed_capture["pattern"].shape
            or base_capture["value"].shape != changed_capture["value"].shape
            or base_capture["head_output"].shape != changed_capture["head_output"].shape):
        raise AttentionSourceDestinationError("base and changed capture shapes differ")
    batch, length, head_count, _head_dim = base_capture["head_output"].shape
    destinations = tuple(int(value) for value in destinations)
    positions = tuple(tuple(int(value) for value in row) for row in positions_by_row)
    heads = tuple(int(value) for value in selected_heads)
    if (len(destinations) != batch or len(positions) != batch or not heads
            or len(heads) != len(set(heads)) or any(not 0 <= head < head_count for head in heads)
            or any(not 0 <= destination < length for destination in destinations)
            or any(not row or len(row) != len(set(row)) for row in positions)
            or any(any(not 0 <= source <= destination for source in row)
                   for row, destination in zip(positions, destinations))):
        raise AttentionSourceDestinationError("response factor coverage is invalid")
    output = {
        factor: torch.zeros_like(base_capture["head_output"], dtype=torch.float32)
        for factor in RESPONSE_FACTORS
    }
    for index, (destination, sources) in enumerate(zip(destinations, positions)):
        for head in heads:
            p0 = base_capture["pattern"][index, head, destination, list(sources)].float()
            p1 = changed_capture["pattern"][index, head, destination, list(sources)].float()
            v0 = base_capture["value"][index, list(sources), head].float()
            v1 = changed_capture["value"][index, list(sources), head].float()
            dp, dv = p1 - p0, v1 - v0
            output["pattern_on_base_value"][index, destination, head] = (dp[:, None] * v0).sum(0)
            output["base_pattern_on_value_change"][index, destination, head] = (p0[:, None] * dv).sum(0)
            output["pattern_value_interaction"][index, destination, head] = (dp[:, None] * dv).sum(0)
    return output


def cue_partition(base_ids, donor_ids, destination):
    base, donor = tuple(base_ids), tuple(donor_ids)
    destination = int(destination)
    if len(base) != len(donor) or not 0 <= destination < len(base):
        raise AttentionSourceDestinationError("paired token length or destination is invalid")
    differences = [index for index, pair in enumerate(zip(base, donor)) if pair[0] != pair[1]]
    if len(differences) != 1:
        raise AttentionSourceDestinationError("partition requires exactly one cue-token change")
    cue = differences[0]
    if destination <= cue:
        raise AttentionSourceDestinationError("destination must follow the temporal cue")
    groups = {
        "prefix": tuple(range(cue)),
        "cue": (cue,),
        "local": tuple(range(cue + 1, destination + 1)),
    }
    flattened = tuple(position for name in GROUPS for position in groups[name])
    if sorted(flattened) != list(range(destination + 1)) or len(flattened) != len(set(flattened)):
        raise AttentionSourceDestinationError("source groups are not an exact causal partition")
    return groups


def batch_destination_partitions(base_batch, donor_batch, destinations):
    if base_batch.row_ids != donor_batch.row_ids or len(destinations) != len(base_batch.row_ids):
        raise AttentionSourceDestinationError("paired batch or destination coverage changed")
    output = []
    for base_ids, donor_ids, row_destinations in zip(
        base_batch.token_rows, donor_batch.token_rows, destinations
    ):
        output.append(tuple(
            cue_partition(base_ids, donor_ids, destination)
            for destination in row_destinations
        ))
    return tuple(output)


def _attention_terms(backend, attention, current, v1):
    heads = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // heads
    batch, length, _width = current.shape

    def qk(linear):
        return linear(current).view(batch, length, heads, head_dim)

    query, key = qk(attention.c_q), qk(attention.c_k)
    query2, key2 = qk(attention.c_q2), qk(attention.c_k2)
    value = attention.c_v(current).view(batch, length, heads, head_dim)
    initial_value = value if v1 is None else v1.view_as(value)
    value = (1.0 - attention.lamb) * value + attention.lamb * initial_value
    cosine, sine = attention.rotary(query)
    query = apply_rotary_emb(F.rms_norm(query, (head_dim,)), cosine, sine)
    key = apply_rotary_emb(F.rms_norm(key, (head_dim,)), cosine, sine)
    query2 = apply_rotary_emb(F.rms_norm(query2, (head_dim,)), cosine, sine)
    key2 = apply_rotary_emb(F.rms_norm(key2, (head_dim,)), cosine, sine)
    score1 = einsum(
        query, key,
        "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k",
    )
    score2 = einsum(
        query2, key2,
        "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k",
    )
    mask = torch.tril(torch.ones(length, length, device=current.device, dtype=torch.bool))
    if attention.squared_attn:
        pattern = ((score1 / head_dim) * (score2 / head_dim)).masked_fill(~mask, 0.0)
    else:
        score1 = score1.masked_fill(~mask, float("-inf"))
        score2 = score2.masked_fill(~mask, float("-inf"))
        pattern = F.softmax(score1 / head_dim, dim=-1) - attention.bilinear_lamb * F.softmax(
            score2 / head_dim, dim=-1
        )
    head_output = einsum(
        pattern, value,
        "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... n_head seq_q d_head",
    ).transpose(1, 2)
    return pattern, value, head_output


def capture_layer_attention(backend, batch, layer, *, call=None):
    layer = int(layer)
    if not 0 <= layer < len(backend.model.transformer.h):
        raise AttentionSourceDestinationError("attention layer is invalid")
    attention = backend.model.transformer.h[layer].attn
    captured = {}

    def capture_inputs(_module, arguments):
        current = arguments[0]
        v1 = arguments[1] if len(arguments) > 1 else None
        pattern, value, reconstructed = _attention_terms(backend, attention, current, v1)
        captured["pattern"] = pattern.detach().clone()
        captured["value"] = value.detach().clone()
        captured["reconstructed"] = reconstructed.detach().clone()

    def capture_native(_module, arguments):
        flattened = arguments[0]
        heads = backend.model.config.n_head
        head_dim = backend.model.config.n_embd // heads
        captured["head_output"] = flattened.detach().clone().view(
            len(batch.row_ids), flattened.shape[1], heads, head_dim
        )

    handles = [
        attention.register_forward_pre_hook(capture_inputs),
        attention.c_proj.register_forward_pre_hook(capture_native),
    ]
    try:
        output = backend.native(batch, capture=False) if call is None else call()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"pattern", "value", "reconstructed", "head_output"}:
        raise AttentionSourceDestinationError("attention source capture is incomplete")
    captured["reconstruction_max_abs"] = float(
        (captured["reconstructed"].float() - captured["head_output"].float()).abs().max()
    )
    return output, captured


def intervene_head_output_delta(
    backend,
    batch,
    base_capture,
    changed_capture,
    *,
    layer,
    selected_heads,
    positions_by_row=None,
):
    """Add an exact captured head-response delta at declared destinations."""
    heads = tuple(int(head) for head in selected_heads)
    head_count = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // head_count
    if not heads or len(heads) != len(set(heads)) or any(not 0 <= head < head_count for head in heads):
        raise AttentionSourceDestinationError("head selection is invalid")
    if positions_by_row is None:
        positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    else:
        positions = tuple(tuple(int(position) for position in row) for row in positions_by_row)
    if len(positions) != len(batch.row_ids) or any(
        len(row) != len(set(row)) or any(not 0 <= position <= int(query) for position in row)
        for row, query in zip(positions, batch.semantic_positions)
    ):
        raise AttentionSourceDestinationError("head-response destination coverage is invalid")

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(
            len(batch.row_ids), flattened.shape[1], head_count, head_dim
        )
        for index, row_positions in enumerate(positions):
            for position in row_positions:
                for head in heads:
                    changed[index, position, head] += (
                        changed_capture["head_output"][index, position, head]
                        - base_capture["head_output"][index, position, head]
                    ).to(device=changed.device, dtype=changed.dtype)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[int(layer)].attn.c_proj.register_forward_pre_hook(patch)
    try:
        return backend.native(batch, capture=False)
    finally:
        handle.remove()


def intervene_ordered_head_output_deltas(backend, batch, specs):
    """Install captured response deltas at multiple attention layers in causal order."""
    specs = tuple(specs)
    layers = [int(spec["layer"]) for spec in specs]
    if not specs or layers != sorted(layers) or len(layers) != len(set(layers)):
        raise AttentionSourceDestinationError(
            "response specs must name unique increasing attention layers")
    handles = []
    head_count = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // head_count
    for spec in specs:
        layer = int(spec["layer"])
        heads = tuple(int(head) for head in spec["selected_heads"])
        if not heads or len(heads) != len(set(heads)) or any(
            not 0 <= head < head_count for head in heads
        ):
            raise AttentionSourceDestinationError("response head selection is invalid")
        supplied = spec.get("positions_by_row")
        positions = (tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
                     if supplied is None else tuple(tuple(int(position) for position in row)
                                                    for row in supplied))
        if len(positions) != len(batch.row_ids) or any(
            len(row) != len(set(row)) or any(not 0 <= position <= int(query) for position in row)
            for row, query in zip(positions, batch.semantic_positions)
        ):
            raise AttentionSourceDestinationError("response destination coverage is invalid")
        base_capture = spec["base_capture"]
        changed_capture = spec["changed_capture"]

        def patch(
            _module,
            arguments,
            heads=heads,
            positions=positions,
            base_capture=base_capture,
            changed_capture=changed_capture,
        ):
            flattened = arguments[0]
            changed = flattened.clone().view(
                len(batch.row_ids), flattened.shape[1], head_count, head_dim
            )
            for index, row_positions in enumerate(positions):
                for position in row_positions:
                    for head in heads:
                        changed[index, position, head] += (
                            changed_capture["head_output"][index, position, head]
                            - base_capture["head_output"][index, position, head]
                        ).to(device=changed.device, dtype=changed.dtype)
            return (changed.reshape_as(flattened),) + tuple(arguments[1:])

        handles.append(
            backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch)
        )
    try:
        return backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()


def intervene_source_groups(
    backend,
    base_batch,
    donor_batch,
    base_capture,
    donor_capture,
    destinations,
    group_names,
    *,
    layer,
    selected_heads,
):
    names = tuple(group_names)
    if not names or len(names) != len(set(names)) or any(name not in GROUPS for name in names):
        raise AttentionSourceDestinationError("source group selection is invalid")
    partitions = batch_destination_partitions(base_batch, donor_batch, destinations)
    heads = tuple(int(head) for head in selected_heads)
    head_count = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // head_count
    if not heads or len(heads) != len(set(heads)) or any(not 0 <= head < head_count for head in heads):
        raise AttentionSourceDestinationError("head selection is invalid")

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(
            len(base_batch.row_ids), flattened.shape[1], head_count, head_dim
        )
        for index, row_destinations in enumerate(destinations):
            for destination_index, destination in enumerate(row_destinations):
                positions = tuple(
                    position for name in names for position in partitions[index][destination_index][name]
                )
                for head in heads:
                    for position in positions:
                        base_term = (
                            base_capture["pattern"][index, head, destination, position]
                            * base_capture["value"][index, position, head]
                        )
                        donor_term = (
                            donor_capture["pattern"][index, head, destination, position]
                            * donor_capture["value"][index, position, head]
                        )
                        changed[index, destination, head] += donor_term - base_term
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[int(layer)].attn.c_proj.register_forward_pre_hook(patch)
    try:
        return backend.native(base_batch, capture=False)
    finally:
        handle.remove()


def intervene_complete_heads(
    backend, base_batch, donor_batch, donor_capture, destinations, *, layer, selected_heads
):
    batch_destination_partitions(base_batch, donor_batch, destinations)
    heads = tuple(int(head) for head in selected_heads)
    head_count = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // head_count

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(
            len(base_batch.row_ids), flattened.shape[1], head_count, head_dim
        )
        for index, row_destinations in enumerate(destinations):
            for destination in row_destinations:
                for head in heads:
                    changed[index, destination, head] = donor_capture["head_output"][
                        index, destination, head
                    ].to(device=changed.device, dtype=changed.dtype)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[int(layer)].attn.c_proj.register_forward_pre_hook(patch)
    try:
        return backend.native(base_batch, capture=False)
    finally:
        handle.remove()
