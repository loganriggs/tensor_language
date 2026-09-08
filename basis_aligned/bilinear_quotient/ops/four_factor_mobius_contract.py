"""Pure exact accounting for a four-factor intervention cube."""

# BQGATE: LIBRARY
from __future__ import annotations

from itertools import combinations
import math

import numpy as np


FACTORS = ("q", "k", "q2", "k2")
MASKS = tuple(range(16))


def subsets(mask: int):
    if mask not in MASKS:
        raise ValueError("mask must be a four-bit integer")
    return tuple(part for part in MASKS if part & mask == part)


def validate_corners(corners):
    if set(corners) != set(MASKS):
        raise ValueError("all and only 16 masks are required")
    arrays = {mask: np.asarray(value, dtype=np.float64) for mask, value in corners.items()}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("corner vectors must have one finite common shape")
    return arrays


def mobius(corners):
    arrays = validate_corners(corners)
    coefficients = {}
    for mask in MASKS:
        total = np.zeros_like(arrays[0])
        for part in subsets(mask):
            total += (-1.0 if (mask.bit_count() - part.bit_count()) % 2 else 1.0) * arrays[part]
        coefficients[mask] = total
    return coefficients


def reconstruct(coefficients):
    arrays = validate_corners(coefficients)
    return {mask: sum((arrays[part] for part in subsets(mask)), np.zeros_like(arrays[0]))
            for mask in MASKS}


def shapley(corners):
    arrays = validate_corners(corners)
    n = len(FACTORS)
    values = {}
    for index, factor in enumerate(FACTORS):
        bit = 1 << index
        total = np.zeros_like(arrays[0])
        for mask in MASKS:
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            total += weight * (arrays[mask | bit] - arrays[mask])
        values[factor] = total
    return values


def shortest_norm_mass_prefix(shapley_vectors, fraction=0.8):
    if not 0.0 < fraction <= 1.0 or set(shapley_vectors) != set(FACTORS):
        raise ValueError("invalid factor vectors or mass fraction")
    norms = {factor: float(np.linalg.norm(np.asarray(shapley_vectors[factor], dtype=np.float64)))
             for factor in FACTORS}
    if not all(math.isfinite(value) for value in norms.values()):
        raise ValueError("factor norms must be finite")
    ordered = sorted(FACTORS, key=lambda factor: (-norms[factor], FACTORS.index(factor)))
    total = sum(norms.values())
    if total == 0.0:
        return (), norms
    chosen, mass = [], 0.0
    for factor in ordered:
        chosen.append(factor)
        mass += norms[factor]
        if mass >= fraction * total:
            break
    return tuple(chosen), norms
