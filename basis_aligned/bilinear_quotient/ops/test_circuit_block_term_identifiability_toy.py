"""Model-free tests for the circuit block-term identifiability toy."""

# BQLANE: cpu

from __future__ import annotations

import numpy as np
import pytest

import circuit_block_term_identifiability_toy as toy


def test_toy_exposes_matrix_gauge_but_meets_two_generic_btd_conditions():
    result = toy.run_toy()
    assert result["matrix_flattening_gl_gauge_max_abs_error"] < 1e-12
    assert result["equal_block_generic_uniqueness_condition"] is True
    assert result["variable_block_generic_uniqueness_condition"] is True
    assert result["mobius_reconstruction_max_abs_error"] == 0.0
    assert result["two_mediator_interaction"] == [4.0, 4.0]


@pytest.mark.parametrize(
    "dimensions,expected",
    [
        ((7, 8, 9, 2, 2), True),
        ((1, 8, 9, 2, 2), False),
        ((7, 2, 2, 2, 2), False),
        ((7, 8, 9, 8, 7), False),
    ],
)
def test_equal_block_condition_boundaries(dimensions, expected):
    assert toy.generic_unique_equal_blocks(*dimensions) is expected


def test_variable_block_condition_boundaries():
    assert toy.generic_unique_variable_blocks(7, 8, 9, (2, 2))
    assert not toy.generic_unique_variable_blocks(2, 2, 2, (2, 2))
    assert not toy.generic_unique_variable_blocks(7, 3, 20, (2, 2))


def test_mobius_requires_complete_lattice_and_reconstructs_three_variables():
    with pytest.raises(ValueError, match="complete Boolean"):
        toy.mobius_coefficients({0: np.array([0.0])}, variables=2)
    values = {mask: np.array([float(mask * mask)]) for mask in range(8)}
    coefficients = toy.mobius_coefficients(values, variables=3)
    reconstructed = toy.reconstruct_from_mobius(coefficients, variables=3)
    assert all(np.array_equal(reconstructed[mask], values[mask]) for mask in values)
