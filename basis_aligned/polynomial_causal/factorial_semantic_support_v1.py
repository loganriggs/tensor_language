"""Reusable finite-cube coordinates and token-local interaction certificates.

Reuse the existing Walsh kernel; never infer its bit convention from row order.
No native model execution, fitting, or changes to completed screen predicates.
"""
from collections import Counter
from itertools import product

import numpy as np
from dealiased_boolean_spectrum import coefficients as binary_coefficients


def signed_coefficients(corners, values):
    """Return coefficients in subset-mask order, bit i = factor i.

    The existing kernel uses (-1)^bit. Put a negative factor at bit=1;
    this handles arbitrary row order without parity or bit-reversal guesses.
    """
    corners = [tuple(c) for c in corners]
    if not corners or not corners[0]:
        raise ValueError('nonempty factorial required')
    n = len(corners[0])
    if len(corners) != 2**n or set(corners) != set(product((-1, 1), repeat=n)):
        raise ValueError('complete unique signed cube required')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(corners),):
        raise ValueError('one value per corner required')
    ordered = np.empty(len(corners))
    for c, v in zip(corners, values):
        ordered[sum((x == -1) << i for i, x in enumerate(c))] = v
    return binary_coefficients(ordered)


def score_rule(margins, labels, accuracy_bar=.75, mean_bar=1.):
    """Worlds x corners; every corner must pass, with no pooled rescue."""
    m = np.asarray(margins, dtype=float)
    labels = np.asarray(labels)
    if m.ndim != 2 or labels.shape != (m.shape[1],) or not np.isfinite(m).all():
        raise ValueError('finite worlds-by-corners margins required')
    if not np.isin(labels, [-1, 1]).all():
        raise ValueError('signed labels required')
    signed = m * labels
    accuracy = (signed > 0).mean(0)
    means = signed.mean(0)
    return {'accuracy': accuracy.tolist(), 'signed_mean_margin': means.tolist(),
            'passed': bool(np.all((accuracy >= accuracy_bar) & (means >= mean_bar))),
            'raw_correct': int((signed > 0).sum())}


def token_square_support(rows, factor_i, factor_j):
    """Certify mixed difference zero for ANY map of individual token positions.

    Diagonal token-multiset equality at each position is necessary and
    sufficient for universal token-local cancellation, independently of weights.
    """
    index = {tuple(r['factors']): r['ids'] for r in rows}
    n = len(rows[0]['factors'])
    if len(index) != len(rows) or set(index) != set(product((-1, 1), repeat=n)):
        raise ValueError('complete unique signed cube required')
    if factor_i == factor_j or not all(0 <= i < n for i in (factor_i, factor_j)):
        raise ValueError('distinct valid factors required')
    if len({len(ids) for ids in index.values()}) != 1:
        raise ValueError('equal token lengths required')
    summaries = []
    for base in index:
        if base[factor_i] != -1 or base[factor_j] != -1:
            continue
        square = []
        for a, b in product((-1, 1), repeat=2):
            c = list(base); c[factor_i] = a; c[factor_j] = b
            square.append(index[tuple(c)])
        t00, t01, t10, t11 = square
        first = [i for i, q in enumerate(zip(*square)) if q[0] != q[2] or q[1] != q[3]]
        second = [i for i, q in enumerate(zip(*square)) if q[0] != q[1] or q[2] != q[3]]
        unmatched = [i for i, q in enumerate(zip(*square))
                     if Counter((q[0], q[3])) != Counter((q[1], q[2]))]
        summaries.append({'first_edit_positions': first, 'second_edit_positions': second,
                          'unmatched_diagonal_positions': unmatched,
                          'universal_token_local_zero': not unmatched})
    return summaries
