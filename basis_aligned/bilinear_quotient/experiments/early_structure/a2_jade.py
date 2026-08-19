"""A2 — weights-only block recovery via JADE + an explicit stratification
tolerance, with its calibration.

The commutant route (Maehara–Murota) asks for matrices that commute EXACTLY, so
it dies at any noise: calibration puts its ceiling at off-block mass ≈ 0.01,
while a grokked model sits at ≈ 0.18. This module replaces the algebraic step
with orthogonal joint approximate diagonalisation (JADE) and makes the tolerance
explicit: return the FINEST block partition whose blocks still carry (1-ε) of the
mass, and sweep ε. That is the doc's stratification cut, applied to blocks.

Scoring is strict. A frequency counts as recovered only if the union of the
blocks assigned to it spans exactly that frequency's planted 4-dim subspace —
a single 1-dim sliver sitting inside a Fourier plane does not count.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
import a2_calibrate as cal
from bq_common import (interaction, support_basis, restrict, jade, block_mass,
                       partition_from_coupling, reorder_by_partition, init_params,
                       gauge_refactor)

torch.set_default_dtype(torch.float64)
P = m.P
TOLS = (0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def score_partition(V, sizes):
    FB = {k: v.to(V.device) for k, v in m.FBLOCKS.items()}
    """V: recovered directions (d,k) grouped by `sizes`. Assign each block to its
    best planted frequency, then require the UNION of a frequency's blocks to be
    that frequency's 4-dim subspace."""
    offs = [0]
    for s in sizes:
        offs.append(offs[-1] + s)
    assign = {}
    for i, s in enumerate(sizes):
        Vb = V[:, offs[i]:offs[i + 1]]
        w = max(FB, key=lambda k: float(((Vb.T @ FB[k]) ** 2).sum()))
        ov = float(((Vb.T @ FB[w]) ** 2).sum() / Vb.shape[1])
        assign.setdefault(int(w), []).append((i, s, ov))
    full, partial = [], []
    for w, lst in assign.items():
        if w == 0:
            continue
        dim = sum(s for _, s, _ in lst)
        cols = torch.cat([torch.arange(offs[i], offs[i] + s) for i, s, _ in lst])
        Vu = V[:, cols]
        cover = float(((Vu.T @ FB[w]) ** 2).sum() / 4)     # of the 4 planted dims
        if dim == 4 and cover > 0.95 and min(o for _, _, o in lst) > 0.95:
            full.append(w)
        elif min(o for _, _, o in lst) > 0.95:
            partial.append(w)
    return {'n_freq_full': len(full), 'freqs_full': sorted(full),
            'n_freq_partial_only': len(partial)}


def frontier(Q, label, sweeps=25, verbose=True):
    U, sv = support_basis(Q, thresh=1e-3)
    cum = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    k = int((cum < 0.999).sum()) + 1
    Ur = U[:, :k].cpu()
    Qr = restrict(Q.cpu(), Ur)
    Pj, T, info = jade(Qr, sweeps=sweeps)
    rows = []
    for tol in TOLS:
        parts = partition_from_coupling(T, tol=tol)
        Pr, sizes = reorder_by_partition(Pj, parts)
        mass, _ = block_mass(Qr, Pr, sizes)
        s = score_partition(Ur @ Pr, sizes)
        rows.append({'tol': tol, 'n_blocks': len(sizes), 'in_block_mass': mass,
                     'block_sizes': sorted(sizes)[:24], **s})
        if verbose:
            print(f"    eps {tol:5.3f}: {len(sizes):2d} blocks (sizes "
                  f"{sorted(set(sizes))}) in-block {mass:.4f} | frequencies fully "
                  f"recovered {s['n_freq_full']:2d}/11 {s['freqs_full']}")
    best = max(rows, key=lambda r: (r['n_freq_full'], r['in_block_mass']))
    return {'label': label, 'support_dim_kept': k,
            'off_block': m.fourier_power(Q)['off_block'],
            'jade_off_diag_mass': info['off_diag_mass'], 'rows': rows,
            'best_n_freq_full': best['n_freq_full'], 'best_tol': best['tol'],
            'best_in_block_mass': best['in_block_mass']}


def main():
    out = {'tols': list(TOLS)}
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)

    print('== CALIBRATION: planted family + isotropic noise ==')
    Qstar = cal.planted_family(list(range(1, P // 2 + 1)))
    g = torch.Generator().manual_seed(0)
    E = torch.randn(P, 2 * P, 2 * P, generator=g).to(Qstar)
    E = 0.5 * (E + E.transpose(1, 2))
    E = E / E.norm() * Qstar.norm()
    out['calibration'] = []
    for eta in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0):
        Qn = Qstar + eta * E
        print(f"  eta {eta:.2f} (off-block {m.fourier_power(Qn)['off_block']:.4f}):")
        r = frontier(Qn, f'planted+iso{eta}')
        r['eta'] = eta
        out['calibration'].append(r)
        print(f"    --> best {r['best_n_freq_full']}/11 at eps {r['best_tol']}")

    print('\n== TRAINED MODELS ==')
    cache = torch.load('a2_cache.pt', weights_only=False)
    out['trained'] = []
    for seed in range(3):
        Q = interaction(cache[seed]['p'])
        Qid, idf = m.canonicalise(Q, basis)
        print(f"  seed {seed} (off-block {m.fourier_power(Qid)['off_block']:.4f}, test acc "
              f"{cache[seed]['hist'][-1]['test_acc']:.4f}):")
        r = frontier(Qid, f'trained{seed}')
        r['seed'] = seed
        r['identifiable_frac'] = idf
        out['trained'].append(r)
        print(f"    --> best {r['best_n_freq_full']}/11 at eps {r['best_tol']} "
              f"(in-block mass {r['best_in_block_mass']:.4f})")

    print('\n== NULL 1: random weights ==')
    out['null_random'] = []
    for seed in range(2):
        p = init_params(2 * P, m.H, P, seed=1000 + seed, device=m.DEV)
        p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
        Qid, _ = m.canonicalise(interaction(p), basis)
        print(f'  random {seed}:')
        r = frontier(Qid, f'random{seed}')
        out['null_random'].append(r)
        print(f"    --> best {r['best_n_freq_full']}/11 at eps {r['best_tol']}")

    print('\n== NULL 2: gauge scramble (must be identical) ==')
    out['null_gauge'] = []
    pg, resid = gauge_refactor(cache[0]['p'], seed=555)
    Qid, _ = m.canonicalise(interaction(pg), basis)
    print(f'  gauge-scrambled seed 0 (refactor residual {resid:.1e}):')
    r = frontier(Qid, 'gauge0')
    r['refactor_residual'] = resid
    out['null_gauge'].append(r)
    print(f"    --> best {r['best_n_freq_full']}/11 at eps {r['best_tol']} "
          f"(trained was {out['trained'][0]['best_n_freq_full']}/11 at "
          f"{out['trained'][0]['best_tol']})")

    print('\n== NULL 3: task shuffle ==')
    out['null_taskshuffle'] = []
    p, hist, _ = m.train_model(0, shuffle_labels=True, log=False)
    Qid, _ = m.canonicalise(interaction(p), basis)
    print(f"  shuffled (train {hist[-1]['train_acc']:.3f} test {hist[-1]['test_acc']:.3f}):")
    r = frontier(Qid, 'taskshuffle0')
    out['null_taskshuffle'].append(r)
    print(f"    --> best {r['best_n_freq_full']}/11 at eps {r['best_tol']}")

    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_jade_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
