#!/usr/bin/env python3
"""Exact Möbius/zeta transforms for response programs indexed by subset masks."""

# BQGATE: LIBRARY
from __future__ import annotations

import numpy as np


def _bits_from_length(length: int) -> int:
    if length <= 0 or length & (length - 1):
        raise ValueError("first dimension must have positive power-of-two length")
    return length.bit_length() - 1


def mobius_transform(values):
    """Return coefficients c with values[S] = sum_{T subset S} c[T]."""
    out = np.asarray(values, dtype=np.float64).copy()
    bits = _bits_from_length(out.shape[0])
    for bit in range(bits):
        flag = 1 << bit
        for mask in range(out.shape[0]):
            if mask & flag:
                out[mask] -= out[mask ^ flag]
    return out


def zeta_transform(coefficients):
    """Invert :func:`mobius_transform` in subset-mask order."""
    out = np.asarray(coefficients, dtype=np.float64).copy()
    bits = _bits_from_length(out.shape[0])
    for bit in range(bits):
        flag = 1 << bit
        for mask in range(out.shape[0]):
            if mask & flag:
                out[mask] += out[mask ^ flag]
    return out


def degree_energy(coefficients):
    """Squared Frobenius mass grouped by interaction degree."""
    values = np.asarray(coefficients, dtype=np.float64)
    bits = _bits_from_length(values.shape[0])
    return {degree: float(sum(np.square(values[mask]).sum()
            for mask in range(values.shape[0]) if mask.bit_count() == degree))
            for degree in range(bits + 1)}


def top_terms(coefficients, labels, limit=20):
    """Rank nonempty interaction tensors by Frobenius norm."""
    values = np.asarray(coefficients, dtype=np.float64)
    bits = _bits_from_length(values.shape[0])
    if len(labels) != bits:
        raise ValueError("label count must equal log2(first dimension)")
    rows = []
    for mask in range(1, values.shape[0]):
        rows.append({"mask": mask, "degree": mask.bit_count(),
            "sites": [labels[bit] for bit in range(bits) if mask & (1 << bit)],
            "norm": float(np.linalg.norm(values[mask])),
            "coefficient": values[mask].tolist()})
    return sorted(rows, key=lambda row: (-row["norm"], row["mask"]))[:limit]
