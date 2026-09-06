"""Reusable exact source-group interventions for selected attention heads.

This module deliberately contains no task thresholds or outcome decisions. A task
runner supplies aligned batches, frozen semantic group names, and its preregistered
bars; this library validates a complete source partition and performs the repeated
pre-c_proj source-term intervention without fitting.
"""

# BQGATE: LIBRARY
from __future__ import annotations

import math
import statistics

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer


DEFAULT_HEADS = (1, 4)
GROUP_ORDER = ("prefix", "cue", "subject_onset", "intervening_suffix", "self")


class SourceGroupError(RuntimeError):
    pass


def aligned_source_partition(base_ids, donor_ids, query):
    """Partition every causal source through query around one aligned cue change."""
    base = tuple(int(item) for item in base_ids)
    donor = tuple(int(item) for item in donor_ids)
    query = int(query)
    if len(base) != len(donor) or not 0 <= query < len(base):
        raise SourceGroupError("paired token length or query is invalid")
    differences = [index for index, pair in enumerate(zip(base, donor)) if pair[0] != pair[1]]
    if len(differences) != 1:
        raise SourceGroupError("source partition requires exactly one aligned cue change")
    cue = differences[0]
    if cue + 2 >= query:
        raise SourceGroupError("cue lacks a two-token subject onset before the query")
    groups = {
        "prefix": tuple(range(cue)),
        "cue": (cue,),
        "subject_onset": (cue + 1, cue + 2),
        "intervening_suffix": tuple(range(cue + 3, query)),
        "self": (query,),
    }
    flattened = tuple(position for name in GROUP_ORDER for position in groups[name])
    if sorted(flattened) != list(range(query + 1)) or len(flattened) != len(set(flattened)):
        raise SourceGroupError("semantic source groups do not form a disjoint complete partition")
    return groups


def batch_partitions(base_batch, donor_batch):
    if base_batch.row_ids != donor_batch.row_ids:
        raise SourceGroupError("recipient and donor row order differs")
    return tuple(
        aligned_source_partition(base_ids, donor_ids, query)
        for base_ids, donor_ids, query in zip(
            base_batch.token_rows,
            donor_batch.token_rows,
            base_batch.semantic_positions,
        )
    )


def validate_group_names(group_names):
    names = tuple(group_names)
    if len(names) != len(set(names)) or any(name not in GROUP_ORDER for name in names):
        raise SourceGroupError("source group selection is unknown or duplicated")
    return names


def _attention_shapes(backend, batch, capture):
    required = {"pattern", "value", "head_output", "reconstruction_max_abs"}
    if not required.issubset(capture):
        raise SourceGroupError("attention capture is incomplete")
    heads = int(backend.model.config.n_head)
    head_dim = int(backend.model.config.n_embd) // heads
    if capture["head_output"].shape[0] != len(batch.row_ids):
        raise SourceGroupError("attention capture batch differs from request")
    return heads, head_dim


def intervene_source_groups(
    backend,
    base_batch,
    donor_batch,
    base_capture,
    donor_capture,
    group_names,
    *,
    selected_heads=DEFAULT_HEADS,
):
    """Install exact donor-minus-base P[k]V[k] terms for selected groups/heads."""
    names = validate_group_names(group_names)
    partitions = batch_partitions(base_batch, donor_batch)
    head_count, head_dim = _attention_shapes(backend, base_batch, base_capture)
    _attention_shapes(backend, donor_batch, donor_capture)
    heads = tuple(int(head) for head in selected_heads)
    if not heads or len(heads) != len(set(heads)) or any(not 0 <= head < head_count for head in heads):
        raise SourceGroupError("selected head set is invalid")

    def patch_heads(_module, arguments):
        flattened = arguments[0]
        head_output = flattened.view(
            len(base_batch.row_ids), flattened.shape[1], head_count, head_dim
        ).clone()
        for index, (query, donor_query, groups) in enumerate(
            zip(base_batch.semantic_positions, donor_batch.semantic_positions, partitions)
        ):
            if query != donor_query:
                raise SourceGroupError("semantic query alignment changed")
            positions = tuple(position for name in names for position in groups[name])
            for position in positions:
                for head in heads:
                    base_term = (
                        base_capture["pattern"][index, head, query, position]
                        * base_capture["value"][index, position, head]
                    )
                    donor_term = (
                        donor_capture["pattern"][index, head, donor_query, position]
                        * donor_capture["value"][index, position, head]
                    )
                    head_output[index, query, head] += donor_term - base_term
        return (head_output.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
    try:
        output, _capture = backend.manual_forward(base_batch)
    finally:
        handle.remove()
    return output


def intervene_complete_heads(
    backend,
    base_batch,
    donor_batch,
    donor_capture,
    *,
    selected_heads=DEFAULT_HEADS,
):
    """Install complete donor head outputs for a frozen selected-head ceiling."""
    batch_partitions(base_batch, donor_batch)
    head_count, head_dim = _attention_shapes(backend, donor_batch, donor_capture)
    heads = tuple(int(head) for head in selected_heads)
    if not heads or len(heads) != len(set(heads)) or any(not 0 <= head < head_count for head in heads):
        raise SourceGroupError("selected head set is invalid")

    def patch_heads(_module, arguments):
        flattened = arguments[0]
        head_output = flattened.view(
            len(base_batch.row_ids), flattened.shape[1], head_count, head_dim
        ).clone()
        for index, (query, donor_query) in enumerate(
            zip(base_batch.semantic_positions, donor_batch.semantic_positions)
        ):
            if query != donor_query:
                raise SourceGroupError("semantic query alignment changed")
            for head in heads:
                head_output[index, query, head] = donor_capture["head_output"][
                    index, donor_query, head
                ]
        return (head_output.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
    try:
        output, _capture = backend.manual_forward(base_batch)
    finally:
        handle.remove()
    return output


def recovery_records(rows, base_output, donor_output, patched_output, *, arm):
    records = []
    for row, base_pair, donor_pair, patched_pair in zip(
        rows,
        base_output.answer_foil,
        donor_output.answer_foil,
        patched_output.answer_foil,
    ):
        base_margin = float(base_pair[0]) - float(base_pair[1])
        donor_margin = float(donor_pair[0]) - float(donor_pair[1])
        patched_margin = float(patched_pair[0]) - float(patched_pair[1])
        values = (base_margin, donor_margin, patched_margin)
        if any(not math.isfinite(value) for value in values):
            raise SourceGroupError("nonfinite recovery input")
        recovery = kernel.signed_pairwise_donor_recovery(
            -base_margin, donor_margin, -patched_margin
        )
        records.append(
            {
                "arm": str(arm),
                "family": str(row["transform_id"]),
                "direction": str(row["direction_id"]),
                "row_id": str(row["row_id"]),
                "recovery": recovery,
            }
        )
    return records


def summarize(records):
    values = [float(record["recovery"]) for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise SourceGroupError("missing or nonfinite source-group recovery")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def summarize_by_family(records, families=("A1", "A2")):
    return {
        family: summarize([record for record in records if record["family"] == family])
        for family in families
    }


def verify_contract():
    first = aligned_source_partition((1, 2, 3, 4, 5, 6), (9, 2, 3, 4, 5, 6), 5)
    second = aligned_source_partition(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (1, 2, 3, 4, 0, 6, 7, 8, 9, 10),
        9,
    )
    if first != {
        "prefix": (),
        "cue": (0,),
        "subject_onset": (1, 2),
        "intervening_suffix": (3, 4),
        "self": (5,),
    }:
        raise SourceGroupError("direct-frame partition contract failed")
    if second["prefix"] != (0, 1, 2, 3) or second["cue"] != (4,) or second["self"] != (9,):
        raise SourceGroupError("report-frame partition contract failed")
    return True


verify_contract()
