"""Repair of A2-5's oracle-selected tolerance (REVIEW_RESPONSE, reviewer finding 6).

A2-5 reported "4/11 frequencies recovered at eps = 0.20" for the trained models and
called it weights-only. It was not: `a2_jade.frontier` picks the reported row by

    best = max(rows, key=lambda r: (r['n_freq_full'], ...))

and `n_freq_full` counts correctly recovered PLANTED frequencies. The tolerance was
chosen by the answer, over a 9-point sweep, so the headline is a best-of-9 oracle
number and the operating rule "eps tracks the off-block mass" is circular.

This module fixes it with a rule that needs no ground truth. After JADE has run,
its own residual off-diagonal mass is a measurable, planted-free estimate of how
much of the family cannot be block-diagonalised at all:

    eps_blind = off-diagonal mass of the JADE-rotated family

and the partition is taken at that tolerance. The same rule is applied identically
to the trained models, the synthetic surrogates and the nulls, and the oracle
best-of-sweep is reported alongside as the upper bound it always was.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
import a2_calibrate as cal
from a2_jade import score_partition, TOLS
from bq_common import (interaction, support_basis, restrict, jade, block_mass,
                       partition_from_coupling, reorder_by_partition, init_params)

torch.set_default_dtype(torch.float64)
P = m.P


def blind_and_oracle(Q, label, sweeps=25, verbose=True):
    U, sv = support_basis(Q, thresh=1e-3)
    cum = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    k = int((cum < 0.999).sum()) + 1
    Ur = U[:, :k].cpu()
    Qr = restrict(Q.cpu(), Ur)
    Pj, T, info = jade(Qr, sweeps=sweeps)

    def score_at(tol):
        parts = partition_from_coupling(T, tol=tol)
        Pr, sizes = reorder_by_partition(Pj, parts)
        mass, _ = block_mass(Qr, Pr, sizes)
        return {'tol': tol, 'n_blocks': len(sizes), 'in_block_mass': mass,
                **score_partition(Ur @ Pr, sizes)}

    eps_blind = info['off_diag_mass']          # needs no ground truth
    blind = score_at(eps_blind)
    rows = [score_at(t) for t in TOLS]
    oracle = max(rows, key=lambda r: (r['n_freq_full'], r['in_block_mass']))
    res = {'label': label, 'off_block': m.fourier_power(Q)['off_block'],
           'eps_blind': eps_blind, 'blind': blind, 'oracle': oracle,
           'sweep': rows}
    if verbose:
        print(f"  {label:28s} off-block {res['off_block']:.4f} | BLIND rule eps="
              f"{eps_blind:.3f} -> {blind['n_freq_full']:2d}/11 | oracle best-of-9 eps="
              f"{oracle['tol']:.3f} -> {oracle['n_freq_full']:2d}/11")
    return res


def main():
    t0 = time.time()
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)
    out = {'rule': 'eps = the JADE-rotated family\'s residual off-diagonal mass', 'runs': []}

    print('== calibration: planted family plus isotropic noise ==')
    Qstar = cal.planted_family(list(range(1, P // 2 + 1)))
    g = torch.Generator().manual_seed(0)
    E = torch.randn(P, 2 * P, 2 * P, generator=g).to(Qstar)
    E = 0.5 * (E + E.transpose(1, 2))
    E = E / E.norm() * Qstar.norm()
    for eta in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7):
        out['runs'].append(blind_and_oracle(Qstar + eta * E, f'planted+iso {eta}'))

    print('\n== trained models ==')
    cache = torch.load('a2_cache.pt', weights_only=False)
    for seed in range(3):
        Qid, _ = m.canonicalise(interaction(cache[seed]['p']), basis)
        out['runs'].append(blind_and_oracle(Qid, f'trained seed {seed}'))

    print('\n== trained models, symmetrised (A2-8) ==')
    S = torch.zeros(2 * P, 2 * P, device=x.device, dtype=x.dtype)
    S[:P, P:] = torch.eye(P, device=x.device, dtype=x.dtype)
    S[P:, :P] = torch.eye(P, device=x.device, dtype=x.dtype)
    for seed in range(3):
        Qid, _ = m.canonicalise(interaction(cache[seed]['p']), basis)
        Qs = 0.5 * (Qid + torch.einsum('ij,mjk,kl->mil', S, Qid, S))
        out['runs'].append(blind_and_oracle(Qs, f'symmetrised seed {seed}'))

    print('\n== nulls ==')
    for seed in range(2):
        p = init_params(2 * P, m.H, P, seed=1000 + seed, device=m.DEV)
        p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
        Qid, _ = m.canonicalise(interaction(p), basis)
        out['runs'].append(blind_and_oracle(Qid, f'null random {seed}'))
    ps, hist, _ = m.train_model(0, shuffle_labels=True, log=False)
    Qid, _ = m.canonicalise(interaction(ps), basis)
    out['runs'].append(blind_and_oracle(Qid, 'null task-shuffle'))

    print('\n== summary: what the blind rule costs ==')
    for r in out['runs']:
        gap = r['oracle']['n_freq_full'] - r['blind']['n_freq_full']
        print(f"  {r['label']:28s} blind {r['blind']['n_freq_full']:2d}/11  "
              f"oracle {r['oracle']['n_freq_full']:2d}/11  gap {gap:+d}")
    out['mean_gap'] = sum(r['oracle']['n_freq_full'] - r['blind']['n_freq_full']
                          for r in out['runs']) / len(out['runs'])
    print(f"  mean oracle-minus-blind gap: {out['mean_gap']:.2f} frequencies")

    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/fix_blind_eps_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
