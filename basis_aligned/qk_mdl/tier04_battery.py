"""Tier 0.4 planted-structure battery: ground-truth MDL (spec §5, Tier 0.4).

Three plants with KNOWN structure and known true DL; three codebooks. The
selectivity requirement: each codebook WINS (lowest DL at matched eps) on its
own plant and LOSES on the others'. Conjunction plant + its codebook (sparse
bilinear pairs) and the HODLR/tree codebook are PENDING (tick 3+) — the table
has explicit holes, not silent ones.

eps = 1.5 * noise floor of each plant (frozen convention, mdl_accounting.py).
V_hat = 512. fp64. Ground-truth DLs are computed with the same dl_* helpers.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from codebooks import CODEBOOKS
from mdl_accounting import dl_svd, dl_bicluster, dl_toeplitz_fourier

torch.manual_seed(0)
N = 512
NOISE = 0.05  # relative noise energy: ||noise||^2/||signal||^2 ~ NOISE^2... set exactly below


def add_noise(M, rel):
    """Add iid noise with ||noise||_F^2 = rel * ||M||_F^2; returns noisy M and
    the exact noise floor (FVU of the clean signal wrt the noisy matrix)."""
    noise = torch.randn_like(M)
    noise *= (rel * (M ** 2).sum() / (noise ** 2).sum()).sqrt()
    Mn = M + noise
    floor = float(((M - Mn) ** 2).sum() / (Mn ** 2).sum())
    return Mn, floor


def plant_lowrank(r=8):
    U = torch.randn(N, r, dtype=torch.float64)
    V = torch.randn(N, r, dtype=torch.float64)
    M, floor = add_noise(U @ V.T, 0.02)
    return M, floor, dl_svd(r, N, N)


def plant_bicluster(k=8):
    rows = torch.randint(0, k, (N,))
    cols = torch.randint(0, k, (N,))
    B = torch.randn(k, k, dtype=torch.float64)
    M, floor = add_noise(B[rows][:, cols], 0.02)
    return M, floor, dl_bicluster(k, k, N, N)


def plant_toeplitz(modes=6):
    n_d = 2 * N - 1
    C = torch.zeros(n_d // 2 + 1, dtype=torch.complex128)
    g = torch.Generator(); g.manual_seed(1)
    pick = torch.randperm(60, generator=g)[:modes] + 1   # low-ish frequencies
    C[pick] = torch.randn(modes, dtype=torch.float64) + \
        1j * torch.randn(modes, dtype=torch.float64)
    c = torch.fft.irfft(C, n=n_d)
    idx = torch.arange(N)[:, None] - torch.arange(N)[None, :] + (N - 1)
    M, floor = add_noise(c[idx], 0.02)
    return M, floor, dl_toeplitz_fourier(modes)


def plant_conjunction(k=8, modes=6):
    """(bicluster) ⊙ (POSITIVE toeplitz gate). Positive gate by design: a
    sign-oscillating factor is unidentifiable from the product alone (the real
    pipeline has the branches; see fit_conjunction docstring / LOG tick 3)."""
    rows = torch.randint(0, k, (N,))
    cols = torch.randint(0, k, (N,))
    B = torch.randn(k, k, dtype=torch.float64)
    n_d = 2 * N - 1
    C = torch.zeros(n_d // 2 + 1, dtype=torch.complex128)
    g = torch.Generator(); g.manual_seed(2)
    pick = torch.randperm(60, generator=g)[:modes] + 1
    C[pick] = torch.randn(modes, dtype=torch.float64) + \
        1j * torch.randn(modes, dtype=torch.float64)
    s = torch.fft.irfft(C, n=n_d)
    c = 1 + 0.8 * s / s.abs().max()                      # gate in [0.2, 1.8]
    idx = torch.arange(N)[:, None] - torch.arange(N)[None, :] + (N - 1)
    M, floor = add_noise(B[rows][:, cols] * c[idx], 0.02)
    true_dl = dl_bicluster(k, k, N, N) + dl_toeplitz_fourier(modes + 1) + 32
    return M, floor, true_dl


PLANTS = {'lowrank(svd)': plant_lowrank, 'bicluster': plant_bicluster,
          'toeplitz': plant_toeplitz, 'conjunction': plant_conjunction}
OWNER = {'lowrank(svd)': 'svd', 'bicluster': 'bicluster', 'toeplitz': 'toeplitz',
         'conjunction': 'conjunction'}

results = {'N': N, 'eps_rule': '1.5x plant noise floor', 'table': {}, 'verdicts': {}}
print(f"{'plant':16s} {'eps':>7s}  " + '  '.join(f'{c:>14s}' for c in CODEBOOKS)
      + '   true-DL  verdict')
for pname, maker in PLANTS.items():
    M, floor, true_dl = maker()
    eps = 1.5 * floor
    row = {}
    for cname, fit in CODEBOOKS.items():
        dl, fvu, meta = fit(M, eps)
        ok = fvu <= eps
        row[cname] = {'dl_bits': dl, 'fvu': fvu, 'met_eps': ok, **meta}
    results['table'][pname] = {'eps': eps, 'true_dl_bits': true_dl, **row}
    met = {c: r for c, r in row.items() if r['met_eps']}
    winner = min(met, key=lambda c: met[c]['dl_bits']) if met else None
    verdict = 'PASS' if winner == OWNER[pname] else 'FAIL'
    results['verdicts'][pname] = {'winner': winner, 'expected': OWNER[pname],
                                  'verdict': verdict}
    cells = '  '.join(
        f"{r['dl_bits'] / 1e3:8.1f}k{'*' if not r['met_eps'] else ' '}"
        + f"({str(r.get('rank', r.get('k', r.get('modes', '?')))):>3})"
        for r in row.values())
    print(f'{pname:16s} {eps:7.4f}  {cells}   {true_dl / 1e3:6.1f}k  '
          f'{verdict} (winner {winner})')
print('(* = failed to meet eps; parenthesis = rank/k/modes)')

results['selectivity'] = ('PASS' if all(v['verdict'] == 'PASS'
                                        for v in results['verdicts'].values())
                          else 'FAIL')
print(f"\nBATTERY SELECTIVITY: {results['selectivity']}")
print('pending codebooks: tree/HODLR (needs ordering machinery); '
      'shared-dictionary sparse-bilinear proper (Tier 1.3)')

with open('/workspace/tensor_language/basis_aligned/qk_mdl/tier04_battery.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print('saved tier04_battery.json')
