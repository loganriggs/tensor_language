"""A3 addendum: which axis is the identifiability limit on?

A3 swept only the component count K at fixed m=8 outputs and d=16 input dimensions,
and attributed the breakdown between K=24 and K=32 to K/d. Reviewer 2 pointed out
that the form family's own effective rank is exactly m at every K, so the same data
is equally consistent with a limit in K/m. THEORY.md T6 argues from Kruskal that the
binding mode is k_C <= m, i.e. the OUTPUT dimension.

This settles it by varying m and d independently. The prediction from T6:
    recovery should fail as K grows past roughly (m + 2d - 2)/2,
so doubling m should help about half as much as doubling d, and the boundary should
move with m at fixed d — which the original sweep could not see.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a3_cp as a3
from bq_common import fit_cp, interaction

DEV = a3.DEV
torch.set_default_dtype(torch.float64)


def kruskal_bound(m, d):
    best = 0
    for R in range(1, 8 * (m + d)):
        if min(R, m) + 2 * min(R, d) >= 2 * R + 2:
            best = R
    return best


def solver_recovery(K, d, m, seed=0, steps=6000):
    """Fit CP to the EXACT teacher forms at the given shape; score components."""
    a3.D, a3.M = d, m
    T = a3.teacher(K, seed)
    Q, S_true = a3.teacher_forms(T)
    ph, err = fit_cp(Q, K, steps=steps, lr=3e-2, seed=seed)
    S_hat = a3.component_forms(ph)
    mt = a3.match_components(
        S_hat, S_true / S_true.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None])
    return {'K': K, 'd': d, 'm': m, 'cp_relerr': err,
            'mean_cos': mt['mean_cos'], 'frac_above_0.99': mt['frac_above_0.99'],
            'kruskal_bound': kruskal_bound(m, d)}


def main():
    t0 = time.time()
    out = {'note': 'varies m and d independently, which the original A3 sweep did not',
           'runs': []}
    d0, m0 = a3.D, a3.M

    print('== fix d = 16, vary the number of outputs m ==')
    print('   (if the limit is in K/d this should not move; T6 says it moves with m)')
    for m in (4, 8, 16, 32):
        for K in (12, 24, 32, 48):
            r = solver_recovery(K, 16, m)
            out['runs'].append(r)
            ok = 'RECOVERED' if r['frac_above_0.99'] > 0.9 else (
                'partial' if r['frac_above_0.99'] > 0.3 else 'failed')
            print(f"  m={m:3d} d=16 K={K:3d} (K/m={K/m:5.2f} K/d={K/16:5.2f}) "
                  f"fit {r['cp_relerr']:.1e} | cos {r['mean_cos']:.3f} "
                  f"| >0.99 {r['frac_above_0.99']:.2f}  {ok:10s} "
                  f"| Kruskal bound {r['kruskal_bound']}")

    print('\n== fix m = 8, vary the input dimension d ==')
    for d in (8, 16, 32):
        for K in (12, 24, 48):
            r = solver_recovery(K, d, 8)
            out['runs'].append(r)
            ok = 'RECOVERED' if r['frac_above_0.99'] > 0.9 else (
                'partial' if r['frac_above_0.99'] > 0.3 else 'failed')
            print(f"  m=  8 d={d:3d} K={K:3d} (K/m={K/8:5.2f} K/d={K/d:5.2f}) "
                  f"fit {r['cp_relerr']:.1e} | cos {r['mean_cos']:.3f} "
                  f"| >0.99 {r['frac_above_0.99']:.2f}  {ok:10s} "
                  f"| Kruskal bound {r['kruskal_bound']}")

    a3.D, a3.M = d0, m0

    print('\n== which predicts recovery: K/m, K/d, or K vs the Kruskal bound? ==')
    import math

    def corr(u, v):
        mu, mv = sum(u) / len(u), sum(v) / len(v)
        du, dv = [a - mu for a in u], [b - mv for b in v]
        return sum(a * b for a, b in zip(du, dv)) / math.sqrt(
            sum(a * a for a in du) * sum(b * b for b in dv) + 1e-300)

    y = [r['frac_above_0.99'] for r in out['runs']]
    preds = {'-log(K/m)': [-math.log(r['K'] / r['m']) for r in out['runs']],
             '-log(K/d)': [-math.log(r['K'] / r['d']) for r in out['runs']],
             '-log(K/kruskal)': [-math.log(r['K'] / r['kruskal_bound']) for r in out['runs']]}
    out['correlations'] = {k: corr(y, v) for k, v in preds.items()}
    for k, v in out['correlations'].items():
        print(f"  correlation of recovery with {k:16s}: {v:+.3f}")
    # and the cleanest test: does recovery track K <= kruskal?
    agree = sum(1 for r in out['runs']
                if (r['frac_above_0.99'] > 0.9) == (r['K'] <= r['kruskal_bound']))
    out['kruskal_agreement'] = f'{agree}/{len(out["runs"])}'
    print(f"  recovery agrees with 'K <= Kruskal bound' on {agree}/{len(out['runs'])} cells")

    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a3_axis_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
