"""e3: computation in superposition for elementwise squares (no embed/unembed).

x in R^m sparse (each feature active w.p. p, value U(-1,1)), target y = x^2,
bilinear net y_hat = D((Lx) o (Rx)) with d_h hidden units.

Because there are no linear terms, x = v*e_i gives y_hat_i = c_i * v^2 exactly,
with c_i = D[i] . (L[:,i] * R[:,i]) -- so "feature i is computed" is the crisp
condition |c_i - 1| small, measured directly from weights.

Baselines (closed form, MSE averaged over all m outputs):
  predict-zero:      p * E[x^4] = p/5
  dedicated(d_h):    (1 - d_h/m) * p/5   (compute d_h squares exactly, drop rest)
  rank bound:        the m outputs live in the span of d_h hidden functions, so
                     total error >= sum of the Gram eigenvalues of the targets
                     beyond the top d_h. G = (p/5 - p^2/9) I + (p^2/9) 11^T, so
                     bound = (m - d_h) * (p/5 - p^2/9) / m -- i.e. dedicated
                     minus the shared-mean component, and NO superposition gain
                     is possible for a linear readout of quadratic features.

Part A: sweep d_h -- does training beat dedicated(d_h)? is #computed > d_h?
        (answer: it converges to the rank bound; #computed ~ d_h)
Part B: iterated sparsification on d_h=16 -- how sparse before degradation.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
from common import (forward, train, iterated_sparsify, remaining_frac,
                    init_params, squares_data)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
M, P_ACT = 64, 0.05
SEED = 0

E_X4 = 1.0 / 5.0
BASE_ZERO = P_ACT * E_X4


def rank_bound(d_h):
    lam_rest = P_ACT * E_X4 - (P_ACT / 3) ** 2
    return (M - d_h) * lam_rest / M


def data_fn():
    return squares_data(2048, M, P_ACT, DEV)


@torch.no_grad()
def eval_mse(p, n_batches=16):
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = data_fn()
        tot += ((forward(p, x) - y) ** 2).sum().item()
        n += y.numel()
    return tot / n


def eval_fn(p):  # protocol threshold works on this
    return eval_mse(p)


@torch.no_grad()
def coeffs(p):
    """c_i = D[i] . (L[:,i]*R[:,i]): the coefficient on x_i^2 in output i."""
    return torch.einsum('ik,ki->i', p['D'], p['L'] * p['R'])


def n_computed(p, tol=0.25):
    return int(((coeffs(p) - 1).abs() < tol).sum())


def make_model(d_h, seed=SEED):
    return init_params(M, M, d_h, M, DEV, seed, embed=False, unembed=False)


results = {'m': M, 'p_active': P_ACT, 'base_zero': BASE_ZERO, 'sweep': []}

# ---- Part A: d_h sweep
for d_h in [8, 16, 32, 64]:
    p = make_model(d_h)
    train(p, data_fn, 15000, lr=3e-3)
    train(p, data_fn, 5000, lr=3e-4)
    mse = eval_mse(p)
    ded = (1 - d_h / M) * BASE_ZERO
    row = {'d_h': d_h, 'mse': mse, 'dedicated': ded, 'rank_bound': rank_bound(d_h),
           'n_computed': n_computed(p),
           'mean_abs_c_err': float((coeffs(p) - 1).abs().mean())}
    results['sweep'].append(row)
    print(f"d_h {d_h:3d}  mse {mse:.2e}  dedicated {ded:.2e}  bound {row['rank_bound']:.2e}  "
          f"gap-to-bound {mse / max(row['rank_bound'], 1e-12) - 1:+.1%}   "
          f"computed {row['n_computed']}/{M}")
    if d_h == 16:
        p16 = {k: v.clone() for k, v in p.items()}

# ---- Part B: sparsify the d_h=16 model
# threshold = trained + 25% of the gap to predict-zero (a 1.3x-trained threshold
# would sit ABOVE predict-zero here and let pruning run to nothing)
mse16 = [r for r in results['sweep'] if r['d_h'] == 16][0]['mse']
degrade = mse16 + 0.25 * (BASE_ZERO - mse16)
print(f'\n=== sparsify d_h=16 (degrade at {degrade:.2e} = trained + 25% of gap to zero)')
p, masks, hist = iterated_sparsify(
    p16, data_fn, eval_fn, ('L', 'R', 'D'),
    l1=3e-6, lr=1e-3, steps_per_iter=1500, prune_frac=0.15,
    degrade_fvu=degrade, max_iters=40)
for h in hist:
    h['n_computed'] = None  # filled below only for final
final = {'frac_remaining': remaining_frac(masks, list(masks)),
         'degrade_threshold': degrade,
         'mse': eval_mse(p), 'n_computed': n_computed(p),
         'dedicated_16': (1 - 16 / M) * BASE_ZERO,
         'dedicated_frac_weights': 3 * 16 / (3 * 16 * M)}
results['sparsify_16'] = {'hist': hist, 'final': final}
print(f"final: remaining {final['frac_remaining']:.1%}  mse {final['mse']:.2e}  "
      f"computed {final['n_computed']}/{M} (dedicated would be 16)")
torch.save({'p': {k: v.cpu() for k, v in p.items()},
            'masks': {k: v.cpu() for k, v in masks.items()}},
           '/workspace/tensor_language/basis_aligned/e3_state_sparse16.pt')

with open('/workspace/tensor_language/basis_aligned/e3_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print('saved e3_results.json')
