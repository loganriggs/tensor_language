"""Model-free tensor checks for the centered induction-factor derivation."""

from __future__ import annotations

import numpy as np


def bilinear(selector: np.ndarray, content: np.ndarray) -> np.ndarray:
    """B(E,U) = sum_r E_r U_r for role-aligned rows."""
    return np.einsum("r,rd->d", selector, content)


def centered_deltas(ex, ux, ey, uy):
    base = bilinear(ex, ux)
    score = bilinear(ey, ux) - base
    content = bilinear(ex, uy) - base
    joint = bilinear(ey, uy) - base
    mixed = bilinear(ey - ex, uy - ux)
    return np.zeros_like(base), score, content, joint, mixed


def test_bilinear_finite_difference_identity_and_literal_zero_replay():
    ex = np.array([0.25, -0.10], dtype=np.float64)
    ey = np.array([-0.30, 0.45], dtype=np.float64)
    ux = np.array([[2.0, -1.0, 3.0], [0.5, 4.0, -2.0]])
    uy = np.array([[-1.0, 2.0, 1.5], [3.0, -0.5, 2.0]])
    replay, score, content, joint, mixed = centered_deltas(ex, ux, ey, uy)
    assert np.array_equal(replay, np.zeros(3))
    np.testing.assert_allclose(joint, score + content + mixed, rtol=0, atol=1e-15)


def test_centered_and_literal_replacement_differ_by_native_contraction_error():
    ex = np.array([0.2, 0.3])
    ey = np.array([0.4, 0.1])
    ux = np.array([[1.0, -2.0], [3.0, 0.5]])
    uy = np.array([[2.0, 1.0], [-1.0, 4.0]])
    base = np.array([10.0, -7.0])
    bx = bilinear(ex, ux)
    bnew = bilinear(ey, uy)
    contraction_error = np.array([2e-7, -3e-7])
    cx = bx + contraction_error
    native = base + cx

    centered = native + (bnew - bx)
    literal_remove_insert = native - cx + bnew
    np.testing.assert_allclose(
        centered - literal_remove_insert, contraction_error, rtol=0, atol=2e-15
    )


def test_partial_score_swap_is_not_literal_normalized_attention():
    # A and C are the registered equality roles; B is omitted background.
    weights_x = np.array([0.2, 0.3, 0.5])
    weights_y = np.array([0.4, 0.4, 0.2])
    values_x = np.eye(3)
    native = bilinear(weights_x, values_x)

    partial_centered = native + bilinear(
        weights_y[:2] - weights_x[:2], values_x[:2]
    )
    literal_full_score_swap = bilinear(weights_y, values_x)

    # The partial intervention has the requested A/C coefficients but leaves
    # recipient background fixed, so its coefficient mass is 1.3 rather than 1.
    assert partial_centered.sum() == 1.3
    assert literal_full_score_swap.sum() == 1.0
    assert not np.allclose(partial_centered, literal_full_score_swap)


def test_role_alignment_is_semantic_authority_not_an_algebraic_consequence():
    ex = np.array([0.1, 0.2])
    ey = np.array([0.8, -0.4])
    ux = np.array([[1.0, 0.0], [0.0, 3.0]])
    aligned = bilinear(ey, ux) - bilinear(ex, ux)
    donor_roles_swapped = bilinear(ey[::-1], ux) - bilinear(ex, ux)
    assert not np.allclose(aligned, donor_roles_swapped)

