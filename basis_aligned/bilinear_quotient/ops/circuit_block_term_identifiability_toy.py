#!/usr/bin/env python3
"""CPU toy for a tensor route to circuit-factor identifiability.

This is not a model experiment and makes no scientific claim about bilin18. It
checks two exact algebraic facts used by the 21:30 mathematical review:

1. flattening donor-by-recipient-by-output data to a matrix leaves a full GL(R)
   change-of-basis gauge; and
2. keeping all three modes can satisfy published generic uniqueness conditions
   for rank-(1,L,L) block-term decompositions.

It also implements Boolean-lattice Möbius inversion, the exact finite-difference
decomposition needed to keep mediator interaction terms separate.
"""

# BQLANE: cpu

from __future__ import annotations

import json

import numpy as np


def generic_unique_equal_blocks(i: int, j: int, k: int, r: int, ell: int) -> bool:
    """Domanov--De Lathauwer generic sufficient condition (equal block sizes)."""
    if min(i, j, k, r, ell) < 1 or ell > min(j, k):
        return False
    return r <= min((j - ell) * (k - ell), i)


def generic_unique_variable_blocks(
    i: int, j: int, k: int, block_ranks: tuple[int, ...]
) -> bool:
    """Second published generic sufficient condition for variable block sizes."""
    if min(i, j, k, *block_ranks) < 1 or any(rank > min(j, k) for rank in block_ranks):
        return False
    largest_pair = sum(sorted(block_ranks, reverse=True)[:2])
    return sum(block_ranks) <= min((i - 1) * (j - 1), k) and j >= largest_pair


def make_block_term_tensor(
    rng: np.random.Generator, i: int, j: int, k: int, block_ranks: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return T=sum_r a_r outer (U_r V_r^T), plus its matrix factors."""
    a = rng.normal(size=(i, len(block_ranks)))
    blocks = []
    for rank in block_ranks:
        left = rng.normal(size=(j, rank))
        right = rng.normal(size=(k, rank))
        blocks.append(left @ right.T)
    tensor = sum(np.einsum("i,jk->ijk", a[:, term], block)
                 for term, block in enumerate(blocks))
    flattened_blocks = np.stack([block.reshape(-1) for block in blocks], axis=1)
    return tensor, a, flattened_blocks


def matrix_gauge_residual(
    tensor: np.ndarray, a: np.ndarray, flattened_blocks: np.ndarray, gauge: np.ndarray
) -> float:
    """Show that A G and B G^{-T} reproduce the same matrix flattening."""
    matrix = tensor.reshape(tensor.shape[0], -1)
    transformed_a = a @ gauge
    transformed_b = flattened_blocks @ np.linalg.inv(gauge).T
    return float(np.max(np.abs(matrix - transformed_a @ transformed_b.T)))


def mobius_coefficients(values: dict[int, np.ndarray], variables: int) -> dict[int, np.ndarray]:
    """Unique finite-set interaction coefficients relative to the zero state."""
    expected = set(range(1 << variables))
    if set(values) != expected:
        raise ValueError("values must contain the complete Boolean intervention lattice")
    coefficients: dict[int, np.ndarray] = {}
    for subset in range(1 << variables):
        terms = []
        for contained in range(1 << variables):
            if contained & ~subset:
                continue
            sign = -1 if (subset.bit_count() - contained.bit_count()) % 2 else 1
            terms.append(sign * np.asarray(values[contained], dtype=np.float64))
        coefficients[subset] = sum(terms, np.zeros_like(terms[0]))
    return coefficients


def reconstruct_from_mobius(coefficients: dict[int, np.ndarray], variables: int) -> dict[int, np.ndarray]:
    reconstructed = {}
    for subset in range(1 << variables):
        included = [value for key, value in coefficients.items() if not key & ~subset]
        reconstructed[subset] = sum(included, np.zeros_like(included[0]))
    return reconstructed


def run_toy() -> dict[str, object]:
    rng = np.random.default_rng(585)
    dimensions = {"donors": 7, "recipients": 8, "outputs": 9}
    block_ranks = (2, 2)
    tensor, a, blocks = make_block_term_tensor(
        rng, dimensions["donors"], dimensions["recipients"],
        dimensions["outputs"], block_ranks,
    )
    gauge = np.array([[1.0, 0.7], [-0.4, 1.2]], dtype=np.float64)
    gauge_error = matrix_gauge_residual(tensor, a, blocks, gauge)

    # Two mediators with a genuine interaction in two output coordinates.
    values = {
        0b00: np.array([1.0, -2.0]),
        0b01: np.array([4.0, -1.0]),
        0b10: np.array([2.0, 3.0]),
        0b11: np.array([9.0, 8.0]),
    }
    coefficients = mobius_coefficients(values, variables=2)
    reconstructed = reconstruct_from_mobius(coefficients, variables=2)
    mobius_error = max(float(np.max(np.abs(values[key] - reconstructed[key]))) for key in values)

    return {
        "schema": "circuit_block_term_identifiability_toy_v1",
        "dimensions": dimensions,
        "term_count": len(block_ranks),
        "block_ranks": list(block_ranks),
        "matrix_flattening_gl_gauge_max_abs_error": gauge_error,
        "equal_block_generic_uniqueness_condition": generic_unique_equal_blocks(
            dimensions["donors"], dimensions["recipients"], dimensions["outputs"],
            len(block_ranks), block_ranks[0],
        ),
        "variable_block_generic_uniqueness_condition": generic_unique_variable_blocks(
            dimensions["donors"], dimensions["recipients"], dimensions["outputs"], block_ranks,
        ),
        "mobius_reconstruction_max_abs_error": mobius_error,
        "two_mediator_interaction": coefficients[0b11].tolist(),
        "limitations": [
            "generic uniqueness is not a deterministic certificate for structured model factors",
            "the theorem requires a complete three-way tensor rather than arbitrary paired observations",
            "immediate-output identification does not establish downstream causal use",
        ],
    }


def main() -> None:
    print(json.dumps(run_toy(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
