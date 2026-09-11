"""Dense exhaustive check of the conditional full-tensor support exchange."""
import json
from pathlib import Path
import torch
from chunked_bilinear_coefficient_v1 import dense
from folded_support_exchange_v1 import quadratic, best_exchange


def control():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    rows = []
    for seed in (813, 814, 815):
        torch.manual_seed(seed)
        basis = torch.randn(10, 7)
        basis /= basis.norm(dim=1, keepdim=True)
        na, nb, nw = torch.randn(8, 7), torch.randn(8, 7), torch.randn(4, 8)
        a, b, w = torch.randn(5, 7), torch.randn(5, 7), torch.randn(4, 5)
        j, support = 2, [0, 1, 2]
        a[j] = torch.randn(3) @ basis[support]
        target = dense(na, nb, nw)
        other_a = a.clone(); other_a[j] = 0
        residual = target - dense(other_a, b, w)
        atoms = torch.stack([dense(row[None], b[j:j+1], w[:, j:j+1]).flatten()
                             for row in basis])
        g, rhs = quadratic((na, nb, nw), a, b, w, j, basis)
        quadratic_error = max(float((g - atoms @ atoms.T).abs().max()),
                              float((rhs - atoms @ residual.flatten()).abs().max()))

        def fit(ids):
            values = torch.linalg.lstsq(atoms[ids].T, residual.flatten()).solution
            return float((residual.flatten() - values @ atoms[ids]).square().sum())

        base = fit(support)
        errors = [fit(support[:p] + [add] + support[p+1:])
                  for p in range(3) for add in range(10) if add not in support]
        new_ids, values, report = best_exchange(g, rhs, support)
        actual = fit(new_ids)
        rows.append(dict(seed=seed, quadratic_error=quadratic_error,
                         exhaustive_best_error=abs(actual - min(base, *errors)),
                         gain_error=abs(base - actual - report['gain']),
                         before=base, after=actual, **report))
    basis = torch.eye(4)
    na, nb, nw = basis[2:3], basis[3:4], torch.ones(1, 1)
    a, b, w = basis[0:1].clone(), nb.clone(), nw.clone()
    g, rhs = quadratic((na, nb, nw), a, b, w, 0, basis)
    ids, values, report = best_exchange(g, rhs, [0])
    error = float((dense(values @ basis[ids], b, w) if False else
                   dense((values @ basis[ids])[None], b, w) - dense(na, nb, nw)).square().sum())
    again_ids, _, again = best_exchange(g, rhs, ids)
    passed = dict(pred_a_dense_exhaustive=max(max(r['quadratic_error'], r['exhaustive_best_error'],
                                                 r['gain_error']) for r in rows) <= 1e-9,
                  pred_b_nonincrease=all(r['after'] <= r['before'] + 1e-9 for r in rows),
                  pred_c_planted=ids == [2] and error <= 1e-12 and report['gain'] > .1,
                  pred_d_settled=again_ids == ids and not again['changed'])
    result = dict(predictions=passed, random_controls=rows,
                  planted=dict(support=ids, error=error, **report),
                  scope='One-reader fixed-writer full-tensor control, not native or joint/global recovery.')
    with Path(__file__).with_name('FOLDED_SUPPORT_EXCHANGE_V1_CONTROL.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))
    assert all(passed.values())


if __name__ == '__main__':
    control()
