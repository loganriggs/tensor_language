"""Compose an upstream attention source write with downstream source-term clamps."""

# BQGATE: LIBRARY
from __future__ import annotations

import attention_source_destination_eval as destination_source


class AttentionPathMediationError(RuntimeError):
    pass


def fixed_source_delta_hook(
    backend,
    base_batch,
    donor_batch,
    base_capture,
    donor_capture,
    destinations,
    group_names,
    *,
    selected_heads,
):
    names = tuple(group_names)
    partitions = destination_source.batch_destination_partitions(
        base_batch, donor_batch, destinations
    )
    heads = tuple(int(head) for head in selected_heads)
    head_count = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // head_count

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(
            len(base_batch.row_ids), flattened.shape[1], head_count, head_dim
        )
        for index, row_destinations in enumerate(destinations):
            for destination_index, destination in enumerate(row_destinations):
                positions = tuple(
                    position
                    for name in names
                    for position in partitions[index][destination_index][name]
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

    return patch


def validate_reader_positions(batch, positions_by_row):
    if len(positions_by_row) != len(batch.row_ids):
        raise AttentionPathMediationError("reader source coverage changed")
    checked = []
    for query, positions in zip(batch.semantic_positions, positions_by_row):
        selected = tuple(int(position) for position in positions)
        if len(selected) != len(set(selected)) or any(not 0 <= position <= query for position in selected):
            raise AttentionPathMediationError("reader source position is invalid")
        checked.append(selected)
    return tuple(checked)


def run_composed(
    backend,
    base_batch,
    donor_batch,
    writer_base_capture,
    writer_donor_capture,
    writer_destinations,
    reader_base_capture,
    *,
    writer_layer=8,
    writer_heads=(1,),
    writer_groups=("cue",),
    reader_layer=9,
    reader_heads=(1, 4),
    reader_positions_by_row=None,
    clamp_complete_reader=False,
    enable_writer=True,
):
    if clamp_complete_reader and reader_positions_by_row is not None:
        raise AttentionPathMediationError("complete and source reader clamps are mutually exclusive")
    positions = None
    if reader_positions_by_row is not None:
        positions = validate_reader_positions(base_batch, reader_positions_by_row)
    handles = []
    if enable_writer:
        writer_hook = fixed_source_delta_hook(
            backend,
            base_batch,
            donor_batch,
            writer_base_capture,
            writer_donor_capture,
            writer_destinations,
            writer_groups,
            selected_heads=writer_heads,
        )
        handles.append(
            backend.model.transformer.h[int(writer_layer)].attn.c_proj.register_forward_pre_hook(
                writer_hook
            )
        )

    dynamic = {}
    if clamp_complete_reader or positions is not None:
        attention = backend.model.transformer.h[int(reader_layer)].attn
        head_count = backend.model.config.n_head
        head_dim = backend.model.config.n_embd // head_count
        heads = tuple(int(head) for head in reader_heads)

        def capture_reader(_module, arguments):
            current = arguments[0]
            v1 = arguments[1] if len(arguments) > 1 else None
            pattern, value, reconstructed = destination_source._attention_terms(
                backend, attention, current, v1
            )
            dynamic["pattern"] = pattern
            dynamic["value"] = value
            dynamic["reconstructed"] = reconstructed

        def clamp_reader(_module, arguments):
            required = {"pattern", "value", "reconstructed"}
            if not required.issubset(dynamic):
                raise AttentionPathMediationError("dynamic reader capture missing")
            flattened = arguments[0]
            native = flattened.view(
                len(base_batch.row_ids), flattened.shape[1], head_count, head_dim
            )
            dynamic["reconstruction_max_abs"] = float(
                (dynamic["reconstructed"].float() - native.float()).abs().max()
            )
            changed = native.clone()
            for index, query in enumerate(base_batch.semantic_positions):
                for head in heads:
                    if clamp_complete_reader:
                        changed[index, query, head] = reader_base_capture["head_output"][
                            index, query, head
                        ].to(device=changed.device, dtype=changed.dtype)
                        continue
                    for position in positions[index]:
                        current_term = (
                            dynamic["pattern"][index, head, query, position]
                            * dynamic["value"][index, position, head]
                        )
                        base_term = (
                            reader_base_capture["pattern"][index, head, query, position]
                            * reader_base_capture["value"][index, position, head]
                        )
                        changed[index, query, head] += base_term - current_term
            return (changed.reshape_as(flattened),) + tuple(arguments[1:])

        handles.extend([
            attention.register_forward_pre_hook(capture_reader),
            attention.c_proj.register_forward_pre_hook(clamp_reader),
        ])
    try:
        output = backend.native(base_batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()
    return output, float(dynamic.get("reconstruction_max_abs", 0.0))
