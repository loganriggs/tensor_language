"""Exact dyadic rank/inertia and explicitly approximate scalar-root simplification.

Bounds apply to products of linear forms in fixed primitive coordinates p,
with linear output readout. They do not bound arbitrary arithmetic circuits.
"""
import json
from pathlib import Path
import sympy as sp
import torch
from extract_scalar_modes import evaluate, build
P = Path(__file__).resolve().parent


def rational_matrix(t):
    return sp.Matrix([[sp.Rational(float(x)) for x in row] for row in t])


def exact_inertia(matrix):
    """Rational congruence elimination, including zero-diagonal 2x2 pivots."""
    a = matrix.copy()
    positive = negative = 0
    while a.rows:
        nonzero = [i for i in range(a.rows) if a[i, i] != 0]
        if nonzero:
            i = max(nonzero, key=lambda j: abs(a[j, j]))
            order = [i] + [j for j in range(a.rows) if j != i]
            a = a.extract(order, order)
            pivot = a[0, 0]
            positive += int(bool(pivot > 0))
            negative += int(bool(pivot < 0))
            a = a[1:, 1:] - a[1:, :1] * a[:1, 1:] / pivot
        else:
            pair = next(((i, j) for i in range(a.rows)
                         for j in range(i + 1, a.rows) if a[i, j] != 0), None)
            if pair is None:
                return positive, negative, a.rows
            i, j = pair
            order = [i, j] + [k for k in range(a.rows) if k not in pair]
            a = a.extract(order, order)
            pivot = a[:2, :2]
            positive += 1
            negative += 1
            a = a[2:, 2:] - a[2:, :2] * pivot.inv() * a[:2, 2:]
    return positive, negative, 0


def main():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    source = torch.load(P / 'EXTRACTED_SCALAR_MODES_V1.pt', weights_only=True)
    s = {k: v.double() for k, v in source['program'].items()}
    # An exact nonzero minor certifies independent linear readers. Consequently
    # all six products p_i can vary independently over R with unconstrained x.
    readers = torch.cat([s['A'], s['B']])
    # Pick pivot columns numerically; verify their determinant exactly below.
    import scipy.linalg
    _, _, piv = scipy.linalg.qr(readers.numpy(), pivoting=True, mode='economic')
    minor = rational_matrix(readers[:, piv[:12].copy()])
    assert minor.det() != 0
    left, right, weights = map(rational_matrix,
                              [s['root_left'], s['root_right'], s['quartic_readout']])
    exact_q = []
    for g in range(4):
        raw = left.T * sp.diag(*list(weights.row(g))) * right
        exact_q.append((raw + raw.T) / 2)
    span_rank = sp.Matrix([list(q) for q in exact_q]).rank()
    assert span_rank == 4
    panels = torch.load(P / 'FROZEN_FRESH_VALIDATION_V1.pt', weights_only=True)['panels']
    gen = torch.Generator().manual_seed(20260920)
    probes = [('isotropic', torch.randn(1024, readers.shape[1], generator=gen, dtype=torch.float64))]
    probes += [(f'reused_context_{p["context"]}', p['rows'].double()) for p in panels]
    exports, records = {}, []
    threshold = 1e-10
    for g, q in enumerate(exact_q):
        inertia = exact_inertia(q)
        numeric = torch.tensor(q.tolist(), dtype=torch.float64)
        eigenvalues, vectors = torch.linalg.eigh(numeric)
        cutoff = eigenvalues.abs().max() * threshold
        pos = [vectors[:, i] * eigenvalues[i].sqrt() for i in range(6) if eigenvalues[i] > cutoff]
        neg = [vectors[:, i] * (-eigenvalues[i]).sqrt() for i in range(6) if eigenvalues[i] < -cutoff]
        k = max(len(pos), len(neg))
        l, r = [], []
        for i in range(k):
            a = pos[i] if i < len(pos) else torch.zeros(6, dtype=torch.float64)
            b = neg[i] if i < len(neg) else torch.zeros(6, dtype=torch.float64)
            l.append(a + b)
            r.append(a - b)
        model = {name: s[name].clone() for name in ['A', 'B']}
        model.update(root_left=torch.stack(l), root_right=torch.stack(r),
                     quartic_readout=torch.ones(1, k, dtype=torch.float64),
                     quadratic_readout=s['quadratic_readout'][g:g+1], constant=s['constant'][g:g+1])
        archive = {n: t.float() for n, t in model.items()}
        loaded = {n: t.double() for n, t in archive.items()}
        qhat = model['root_left'].T @ model['root_right']
        qhat = (qhat + qhat.T) / 2
        coefficient_error = float((qhat - numeric).norm() / numeric.norm())
        checks = []
        for name, x in probes:
            ref = evaluate(s, x)[:, g:g+1]
            err = float((evaluate(model, x) - ref).norm() / ref.norm())
            rounding = float((evaluate(loaded, x) - ref).norm() / ref.norm())
            assert err < 1e-8 and rounding < 1e-5
            checks.append(dict(panel=name, fp64_relative_error=err, fp32_archive_relative_error=rounding))
        dag, outputs = build(loaded)
        exports[g] = archive
        records.append(dict(mode=g, exact_rank=q.rank(), exact_inertia=list(inertia),
                            exact_minimum_root_products=max(inertia[:2]),
                            eigenvalues=eigenvalues.tolist(), retained_positive=len(pos), retained_negative=len(neg),
                            approximate_root_products=k, relative_eigenvalue_cutoff=threshold,
                            root_coefficient_relative_error=coefficient_error,
                            cost=dag.cost(outputs), checks=checks))
    result = dict(records=records, independent_reader_minor_columns=piv[:12].tolist(),
                  independent_reader_minor_exact_nonzero=True,
                  exact_quartic_output_span_rank=span_rank,
                  shared_root_products_lower_bound_fixed_primitives=span_rank,
                  existing_shared_root_products=4,
                  shared_root_optimal_within_restricted_class=True,
                  separate_approximate_roots_plus_shared_primitives=6+sum(r['approximate_root_products'] for r in records),
                  scope='Exact ranks/inertias refer to stored FP32 coefficients treated as dyadic rationals. '
                        'Exported scalar reductions deliberately discard eigenvalues below a stated tolerance; '
                        'they are approximations, not exact identities. Bounds require products of linear forms '
                        'in the fixed six primitive coordinates and linear output readout. No bound on arbitrary '
                        'circuits, changed primitive dictionaries, or native-model approximation error.')
    torch.save(dict(programs=exports, scope=result['scope']), P / 'SCALAR_INERTIA_PROGRAMS_V1.pt')
    (P / 'SCALAR_ROOT_INERTIA_V1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
