"""Exact fixtures separating shared arithmetic, compressed state, and causal reuse.

CPU only; no trained weights or data. Necessary square-bank test is inconclusive
on a zero commutator. It does not test general overcomplete bilinear products.
"""
import json
from pathlib import Path
import sympy as s


def square_bank_obstruction(forms):
    """Nonzero [Q0^-1 Qi,Q0^-1 Qj] rules out an invertible d-square bank.

    Requires nonsingular Q0. A zero result alone is not a real-congruence proof.
    Exact symbolic arithmetic is intended for small fixtures, not 1152D weights.
    """
    inverse = forms[0].inv()
    relatives = [inverse * q for q in forms[1:]]
    return [a * b - b * a for i, a in enumerate(relatives)
            for b in relatives[i + 1:]]


def controls():
    a = s.Matrix([[1, 1], [0, 1]])
    q0 = a.T * a
    q1 = a.T * s.diag(2, 3) * a
    raw_commutator = q0 * q1 - q1 * q0
    x0, x1 = s.symbols('x0 x1', real=True)
    x = s.Matrix([x0, x1])
    z = a * x
    shared = s.Matrix([z[0] ** 2, z[1] ** 2])
    outputs = s.Matrix([(x.T * q * x)[0] for q in [q0, q1]])
    coefficients = s.Matrix([[1, 1], [2, 3]])
    q2 = s.Matrix([[0, 1], [1, 0]])
    obstruction = square_bank_obstruction([q0, q1, q2])[0]
    metric = a.inv().T * a.inv()
    q, k = s.Matrix([1, 1]), s.Matrix([1, 0])
    gauge = s.diag(2, s.Rational(1, 2))
    qg, kg = gauge*q, gauge.inv().T*k
    # Zero-epsilon limiting case already falsifies the proposed general gauge.
    normalized_squared = lambda u, v: (u.dot(v))**2/(u.dot(u)*v.dot(v))
    checks = {
        'exact_shared_square_replay': all(s.expand(v) == 0 for v in outputs-coefficients*shared),
        'full_rank_forms': q0.det() != 0 and q1.det() != 0,
        'invertible_raw_commutator_despite_shared_squares': raw_commutator.det() != 0,
        'third_consumer_rejects_two_square_bank': obstruction != s.zeros(2),
        'norm_metric_retained_exactly': s.expand((z.T*metric*z)[0]-x.dot(x)) == 0,
        'norm_is_not_sum_of_transformed_squares': s.expand(z.dot(z)-x.dot(x)) != 0,
        'attention_gauge_preserves_raw_dot': q.dot(k) == qg.dot(kg),
        'attention_gauge_changes_normalized_dot': normalized_squared(q,k) != normalized_squared(qg,kg),
    }
    return {
        'passed': all(checks.values()), 'checks': {k: bool(v) for k,v in checks.items()},
        'Q0': str(q0), 'Q1': str(q1),
        'raw_commutator_determinant': int(raw_commutator.det()),
        'third_consumer_relative_commutator': str(obstruction),
        'transformed_norm_metric': str(metric),
        'attention_normalized_dot_squared': [str(normalized_squared(q,k)), str(normalized_squared(qg,kg))],
        'scope': 'Exact small fixtures only; no native fit, discovery, extraction or causal adoption.',
        'model_forwards': 0,
    }


if __name__ == '__main__':
    result = controls()
    assert result['passed']
    path = Path(__file__).with_name('SHARED_QUADRATIC_FACTOR_MATH_V1_CONTROLS.json')
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
