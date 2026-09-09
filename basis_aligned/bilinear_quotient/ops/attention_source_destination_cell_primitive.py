#!/usr/bin/env python3
"""Exact source-role by destination-role cells for Bilin18 attention writes."""

# BQGATE: LIBRARY
from __future__ import annotations

import attention_source_factor_primitive as source


def source_destination_cell_deltas(
    native, donor, source_groups, destination_groups, torch,
):
    """Partition the full donor-minus-native head write into exact cells.

    ``source_groups`` is boolean ``[batch, source_group, source]`` and
    ``destination_groups`` is boolean ``[batch, destination_group, query]``.
    The returned cells have shape
    ``[batch, query, destination_group, source_group, output]``.

    Each cell is the complete five-factor donor term minus the complete native
    term for one source role, retained only at one destination role. Therefore
    summing all cells exactly reconstructs the donor-minus-native write wherever
    the destination partition is covered. This is termwise algebra, not a claim
    that a singleton cell corresponds to a coherent residual-state intervention.
    """
    base_grouped = source.mixed_grouped_query_source_writes(
        native, donor, (), source_groups, torch)
    donor_grouped = source.mixed_grouped_query_source_writes(
        native, donor, source.SOURCE_FACTORS, source_groups, torch)
    batch, queries, _source_count, output = base_grouped.shape
    if (destination_groups.dtype != torch.bool
            or destination_groups.ndim != 3
            or destination_groups.shape[0] != batch
            or destination_groups.shape[2] != queries
            or destination_groups.device != base_grouped.device):
        raise ValueError(
            "destination groups must be boolean [batch,group,query] on the factor device"
        )
    if (destination_groups.sum(1) > 1).any():
        raise ValueError("destination groups overlap")
    destination = destination_groups.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    cells = destination.to(base_grouped.dtype) * (
        donor_grouped - base_grouped
    ).unsqueeze(2)
    covered = destination_groups.any(1)
    return {
        "base": base_grouped.sum(2),
        "donor": donor_grouped.sum(2),
        "cells": cells,
        "covered_destinations": covered,
        "shape": {
            "batch": batch,
            "queries": queries,
            "destination_groups": int(destination_groups.shape[1]),
            "source_groups": int(source_groups.shape[1]),
            "output": output,
        },
    }


def compose_cell_intervention(decomposition, selected_cells, mode, torch):
    """Compose selected exact cells as base sufficiency or donor reset."""
    if mode not in {"sufficiency", "reset"}:
        raise ValueError("cell intervention mode must be sufficiency or reset")
    cells = decomposition["cells"]
    if (selected_cells.dtype != torch.bool or selected_cells.ndim != 2
            or tuple(selected_cells.shape) != tuple(cells.shape[2:4])
            or selected_cells.device != cells.device):
        raise ValueError(
            "selected cells must be boolean [destination_group,source_group] on the cell device"
        )
    delta = (cells * selected_cells[None, None, :, :, None].to(cells.dtype)).sum((2, 3))
    return decomposition["base"] + delta if mode == "sufficiency" \
        else decomposition["donor"] - delta
