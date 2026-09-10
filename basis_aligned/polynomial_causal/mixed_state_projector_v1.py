"""Exact finite-domain interaction removal; no learned circuit is assumed."""
import numpy as np
from factorial_semantic_support_v1 import signed_coefficients


def projector(corners, i, j):
    corners = [tuple(c) for c in corners]
    # Reuse complete-cube validation and coordinate convention.
    signed_coefficients(corners, np.ones(len(corners)))
    if i == j or not all(0 <= k < len(corners[0]) for k in (i, j)):
        raise ValueError('two distinct factors required')
    index = {c: k for k, c in enumerate(corners)}
    out = np.eye(len(corners))
    for row, c in enumerate(corners):
        for flips, sign in [((i,), -1), ((j,), -1), ((i, j), 1)]:
            other = list(c)
            for k in flips: other[k] *= -1
            out[row, index[tuple(other)]] += sign
    return out / 4


def controls():
    from itertools import product
    c = np.array(list(product((-1, 1), repeat=5)))
    q = projector(c, 2, 4)
    additive = 3 + c[:, 2] * 2 + c[:, 4] * 7 + c[:, 1]*c[:, 2]
    mixed = c[:, 2]*c[:, 4]*(2 + 3*c[:, 0] + c[:, 3]*c[:, 1])
    assert np.array_equal(q, q.T) and np.array_equal(q@q, q)
    assert np.max(abs(q@additive)) == 0
    assert np.array_equal(q@mixed, mixed)
    assert np.max(abs(q@(additive+mixed)-mixed)) == 0
    return {'passed': True, 'rank': int(np.linalg.matrix_rank(q)),
            'projector_error': float(np.max(abs(q@q-q))),
            'additive_annihilated': True, 'contextual_mixed_preserved': True}
