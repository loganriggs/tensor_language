# BQGATE: LIBRARY
"""CPU helpers for held-out factorial states in exact read/write tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FactorialStates:
    grand: np.ndarray
    panel: Mapping[int, np.ndarray]
    direction: Mapping[int, np.ndarray]
    cell: Mapping[tuple[int, int], np.ndarray]
    additive: Mapping[tuple[int, int], np.ndarray]


def balanced_mod4_folds(row_count: int = 16) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return two reporter-disjoint folds balanced over row-index parity."""
    if row_count <= 0 or row_count % 4:
        raise ValueError("row_count must be a positive multiple of four")
    first = np.asarray([index for index in range(row_count) if index % 4 < 2], dtype=np.int64)
    second = np.asarray([index for index in range(row_count) if index % 4 >= 2], dtype=np.int64)
    return {"first_to_second": (first, second), "second_to_first": (second, first)}


def suffix_align(tensor: np.ndarray, mask: np.ndarray, panel_count: int = 3) -> tuple[np.ndarray, int]:
    """Align each panel/row tensor on the final common valid suffix."""
    tensor, mask = np.asarray(tensor), np.asarray(mask, dtype=bool)
    if tensor.ndim != 4 or mask.shape != tensor.shape[:3]:
        raise ValueError("expected tensor[P,R,L,N] and mask[P,R,L]")
    if not 0 < panel_count <= tensor.shape[0]:
        raise ValueError("invalid panel_count")
    positions = [[np.flatnonzero(mask[p, row]) for row in range(tensor.shape[1])]
                 for p in range(panel_count)]
    if any(len(value) == 0 for panel in positions for value in panel):
        raise ValueError("every selected row needs at least one valid position")
    common = min(len(value) for panel in positions for value in panel)
    aligned = np.empty((panel_count, tensor.shape[1], common, tensor.shape[3]), dtype=tensor.dtype)
    for panel in range(panel_count):
        for row in range(tensor.shape[1]):
            aligned[panel, row] = tensor[panel, row, positions[panel][row][-common:]]
    return aligned, common


def fit_factorial_states(values: np.ndarray, fit_rows: np.ndarray) -> FactorialStates:
    """Fit raw panel, direction, cell, and additive tensor states."""
    values, fit_rows = np.asarray(values), np.asarray(fit_rows, dtype=np.int64)
    if values.ndim < 3 or values.shape[0] != 3 or fit_rows.ndim != 1 or not len(fit_rows):
        raise ValueError("expected values[3,rows,...] and nonempty one-dimensional fit rows")
    if min(fit_rows) < 0 or max(fit_rows) >= values.shape[1]:
        raise ValueError("fit row outside tensor")
    if any(np.sum(fit_rows % 2 == direction) == 0 for direction in (0, 1)):
        raise ValueError("fit rows must include both directions")
    grand = values[:, fit_rows].mean(axis=(0, 1), dtype=np.float64)
    panel = {p: values[p, fit_rows].mean(axis=0, dtype=np.float64) for p in range(3)}
    direction = {d: values[:, fit_rows[fit_rows % 2 == d]].mean(axis=(0, 1), dtype=np.float64)
                 for d in (0, 1)}
    cell = {(p, d): values[p, fit_rows[fit_rows % 2 == d]].mean(axis=0, dtype=np.float64)
            for p in range(3) for d in (0, 1)}
    additive = {(p, d): panel[p] + direction[d] - grand for p in range(3) for d in (0, 1)}
    return FactorialStates(grand=grand, panel=panel, direction=direction,
                           cell=cell, additive=additive)


def contract_state_program(
    writer: FactorialStates,
    reader: FactorialStates,
    model: str,
) -> dict[tuple[int, int], np.ndarray]:
    """Contract independently fitted writer/reader states into factor-response states."""
    if model not in {"cell", "additive"}:
        raise ValueError("model must be cell or additive")
    hstates, rstates = getattr(writer, model), getattr(reader, model)
    return {key: np.sum(hstates[key] * rstates[key], axis=0, dtype=np.float64)
            for key in hstates}


def cosine_and_recovery(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes differ")
    rr = float(reference @ reference)
    pp = float(prediction @ prediction)
    if rr <= 0 or pp <= 0:
        raise ValueError("nonzero prediction and reference required")
    dot = float(prediction @ reference)
    return dot / np.sqrt(pp * rr), dot / rr
