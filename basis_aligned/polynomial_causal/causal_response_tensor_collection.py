"""Model-free utilities for lawful per-document causal-response tensors.

The GPU collector stores four additive sufficient statistics for every
phase/source/target/source-document cell.  This module owns their aggregation,
validation, and create-only serialization so those invariants can be tested without
loading bilin18 or opening an outcome artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import torch


STATISTIC_NAMES = (
    "member_signed_sum",
    "member_abs_sum",
    "off_signed_sum",
    "off_abs_sum",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_position_index(
    row_document_ids: torch.Tensor,
    selected_rows: torch.Tensor,
    *,
    positions_per_row: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted document IDs and one dense document index per selected position."""
    row_document_ids = torch.as_tensor(row_document_ids, dtype=torch.int64)
    selected_rows = torch.as_tensor(selected_rows, dtype=torch.int64)
    if row_document_ids.ndim != 1 or selected_rows.ndim != 1:
        raise ValueError("row document IDs and selected rows must be vectors")
    if selected_rows.numel() == 0 or positions_per_row <= 0:
        raise ValueError("at least one row must be selected")
    if selected_rows.min() < 0 or selected_rows.max() >= row_document_ids.numel():
        raise ValueError("selected row is out of range")
    selected_document_ids = row_document_ids[selected_rows]
    document_ids, row_to_document = torch.unique(
        selected_document_ids, sorted=True, return_inverse=True
    )
    position_to_document = row_to_document[:, None].expand(
        -1, positions_per_row
    ).reshape(-1)
    return document_ids, position_to_document


def local_mask_from_global(
    global_mask: torch.Tensor,
    selected_rows: torch.Tensor,
    *,
    positions_per_row: int = 256,
) -> torch.Tensor:
    """Restrict a flat census-grid mask to selected rows, preserving row order."""
    global_mask = torch.as_tensor(global_mask, dtype=torch.bool)
    selected_rows = torch.as_tensor(selected_rows, dtype=torch.int64)
    if positions_per_row <= 0 or global_mask.ndim != 1 or (
        global_mask.numel() % positions_per_row != 0
    ):
        raise ValueError("global mask must be a flat whole-row census mask")
    return global_mask.reshape(-1, positions_per_row)[selected_rows].reshape(-1)


def aggregate_document_responses(
    dce: torch.Tensor,
    position_to_document: torch.Tensor,
    member_masks: Mapping[str, torch.Tensor],
    off_masks: Mapping[str, torch.Tensor],
    *,
    document_count: int,
) -> dict[str, torch.Tensor]:
    """Aggregate a per-position signed CE change into additive document statistics.

    Returns each statistic as ``[target, document]`` in mapping insertion order, plus
    static member/off counts.  Documents with zero member support remain explicitly
    masked by ``member_count == 0``; they are not lawful response cells.
    """
    dce = torch.as_tensor(dce, dtype=torch.float64).reshape(-1)
    position_to_document = torch.as_tensor(position_to_document, dtype=torch.int64)
    if dce.shape != position_to_document.shape:
        raise ValueError("dCE and position-to-document index must align")
    if document_count <= 0:
        raise ValueError("document_count must be positive")
    if position_to_document.min() < 0 or position_to_document.max() >= document_count:
        raise ValueError("position document index is out of range")
    if list(member_masks) != list(off_masks):
        raise ValueError("member and off-mask target order must match")

    result = {
        name: torch.zeros((len(member_masks), document_count), dtype=torch.float64)
        for name in STATISTIC_NAMES
    }
    result["member_count"] = torch.zeros(
        (len(member_masks), document_count), dtype=torch.int64
    )
    result["off_count"] = torch.zeros(
        (len(member_masks), document_count), dtype=torch.int64
    )

    for target_index, target in enumerate(member_masks):
        member = torch.as_tensor(member_masks[target], dtype=torch.bool).reshape(-1)
        off = torch.as_tensor(off_masks[target], dtype=torch.bool).reshape(-1)
        if member.shape != dce.shape or off.shape != dce.shape:
            raise ValueError("every target mask must align with dCE")
        if torch.any(member & off):
            raise ValueError("member and off masks must be disjoint")
        if not off.any():
            raise ValueError("every target must have off-slice positions")
        for label, mask in (("member", member), ("off", off)):
            docs = position_to_document[mask]
            values = dce[mask]
            result[f"{label}_signed_sum"][target_index].scatter_add_(0, docs, values)
            result[f"{label}_abs_sum"][target_index].scatter_add_(
                0, docs, values.abs()
            )
            result[f"{label}_count"][target_index].scatter_add_(
                0, docs, torch.ones_like(docs, dtype=torch.int64)
            )
    return result


def validate_response_tensors(
    statistics: Mapping[str, torch.Tensor],
    member_count: torch.Tensor,
    off_count: torch.Tensor,
    *,
    expected_prefix: tuple[int, int, int],
    tolerance: float = 1e-5,
) -> dict[str, int | float]:
    """Validate dense ``[phase, source, target, document]`` response arrays."""
    member_count = torch.as_tensor(member_count, dtype=torch.int64)
    off_count = torch.as_tensor(off_count, dtype=torch.int64)
    if member_count.ndim != 2 or off_count.shape != member_count.shape:
        raise ValueError("member/off counts must be aligned [target, document] arrays")
    expected_shape = expected_prefix + (member_count.shape[1],)
    if expected_prefix[2] != member_count.shape[0]:
        raise ValueError("target count does not align with statistics")
    for name in STATISTIC_NAMES:
        value = torch.as_tensor(statistics[name])
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected_shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains a nonfinite value")
    if torch.any(member_count < 0) or torch.any(off_count <= 0):
        raise ValueError("member counts must be nonnegative and off counts positive")
    valid = member_count > 0
    expanded_valid = valid[None, None, :, :].expand(expected_shape)
    if not expanded_valid.any():
        raise ValueError("no valid member-supported response cells")
    member_abs = torch.as_tensor(statistics["member_abs_sum"], dtype=torch.float64)
    member_signed = torch.as_tensor(statistics["member_signed_sum"], dtype=torch.float64)
    off_abs = torch.as_tensor(statistics["off_abs_sum"], dtype=torch.float64)
    off_signed = torch.as_tensor(statistics["off_signed_sum"], dtype=torch.float64)
    if torch.any(member_abs[expanded_valid] + tolerance < member_signed[expanded_valid].abs()):
        raise ValueError("member absolute sums violate the triangle inequality")
    if torch.any(off_abs + tolerance < off_signed.abs()):
        raise ValueError("off absolute sums violate the triangle inequality")
    return {
        "valid_cells": int(expanded_valid.sum()),
        "unsupported_dense_slots": int((~expanded_valid).sum()),
        "minimum_member_count_valid": int(member_count[valid].min()),
        "minimum_off_count": int(off_count.min()),
        "maximum_triangle_slack_violation": float(
            max(
                (member_signed.abs() - member_abs)[expanded_valid].max().item(),
                (off_signed.abs() - off_abs).max().item(),
            )
        ),
    }


def atomic_create_json(path: Path, value: object) -> None:
    """Create a JSON file without overwriting an existing terminal artifact."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    _atomic_create_bytes(path, payload)


def atomic_create_torch(path: Path, value: object) -> None:
    """Create a torch artifact without overwriting an existing artifact."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
