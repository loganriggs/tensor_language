"""Third attempt at the open question: is the residual hard because it shares
READER-MODE structure with the signal?

Where the previous two attempts landed (`a2_coherence.py`): the damage is not
explained by magnitude, by the residual's reader-mode coherence (effective rank
19.66 of 23, nearly as incoherent as isotropic noise), or by concentration in the
coupling graph (spread over 100 of 110 block pairs). Reader-mode coherence IS a
real damage channel — a rank-1 perturbation at the same mass gives 0/11 — but the
real residual is not coherent, so that is not what is happening.

What every synthetic surrogate so far has in common is that it is INDEPENDENT of
the signal. The real residual is not: it is the off-block part of one fitted
function. The block part and the residual are orthogonal in Sym² by construction
— the projection makes them so — so any shared structure has to live in the other
mode. Both are produced by the same output-mixing matrix D, so the natural place
to look is the READER mode: the left singular subspace of the (m x dim Sym^2)
unfolding.

HYPOTHESIS. The residual's reader-mode subspace is aligned with the signal's,
whereas an injected perturbation's is independent. Aligned residual mass adds to
the coupling graph in the same reader directions the signal uses, so summing over
readers cannot average it away.

TEST. Measure the alignment for the real residual and for every surrogate; then
construct a surrogate with matched alignment and see whether it becomes hard.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
import a2_calibrate as cal
from a2_coherence import (reader_unfold, reader_coherence, off_block_part,
                          make_perturbation, scale_to_off_block, FREQS)
from fix_blind_eps import blind_and_oracle
from bq_common import interaction

torch.set_default_dtype(torch.float64)
P = m.P


def reader_subspace(Q, k):
    """Top-k left singular vectors of the reader-mode unfolding."""
    U, s, _ = torch.linalg.svd(reader_unfold(Q).double(), full_matrices=False)
    return U[:, :k], s


def alignment(E, Qb, k=6):
    """Mean captured energy of the signal's top-k reader directions inside the
    residual's top-k reader subspace. Chance for two independent k-subspaces of
    R^m is k/m."""
    Ue, _ = reader_subspace(E, k)
    Ub, _ = reader_subspace(Qb, k)
    return float(((Ue.T @ Ub) ** 2).sum() / k)


def project_reader(E, U):
    """Force E's reader mode into the subspace spanned by U (m x k)."""
    V = reader_unfold(E)
    Vp = U @ (U.T @ V)
    d = E.shape[-1]
    iu, ju = torch.triu_indices(d, d)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(E)
    out = torch.zeros_like(E)
    out[:, iu, ju] = Vp / sc
    out[:, ju, iu] = Vp / sc
    return out


def main():
    t0 = time.time()
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)
    cache = torch.load('a2_cache.pt', weights_only=False)
    allw = list(range(0, P // 2 + 1))
    out = {'hypothesis': 'the residual shares reader-mode structure with the signal'}

    Qstar = cal.planted_family(FREQS)
    g = torch.Generator().manual_seed(0)
    Eiso = torch.randn(P, 2 * P, 2 * P, generator=g).to(Qstar)
    Eiso = off_block_part(0.5 * (Eiso + Eiso.transpose(1, 2)))

    print('== 1. is the real residual reader-aligned with its own signal? ==')
    print(f'   (chance alignment for independent 6-subspaces of R^{P} is {6/P:.3f})')
    out['alignment'] = []
    for seed in range(3):
        Qid, _ = m.canonicalise(interaction(cache[seed]['p']), basis)
        Qb = m.block_project(Qid, allw)
        Res = Qid - Qb
        al = alignment(Res, Qb)
        out['alignment'].append({'model': f'trained seed {seed}', 'alignment': al})
        print(f"  trained seed {seed}: residual-to-signal reader alignment {al:.4f}")
    for tag, E in (('isotropic noise', Eiso),
                   ('reader-rank-1 synthetic', make_perturbation(1, 1, Qstar)),
                   ('reader-rank-5 synthetic', make_perturbation(5, 5, Qstar))):
        al = alignment(E, Qstar)
        out['alignment'].append({'model': tag, 'alignment': al})
        print(f"  {tag:26s}: alignment to the planted signal {al:.4f}")

    print('\n== 2. force a surrogate into the signal\'s reader subspace ==')
    Qid0, _ = m.canonicalise(interaction(cache[0]['p']), basis)
    Qb0 = m.block_project(Qid0, allw)
    target = m.fourier_power(Qid0)['off_block']
    Ub, _ = reader_subspace(Qstar, 6)
    out['forced'] = []
    for k in (2, 4, 6, 10, 23):
        Uk, _ = reader_subspace(Qstar, k)
        E = off_block_part(project_reader(Eiso, Uk))
        if float(E.norm()) < 1e-12:
            continue
        Qp = scale_to_off_block(Qstar, E, target)
        Ea = off_block_part(Qp - Qstar)
        r = blind_and_oracle(Qp, f'reader-aligned k={k}', verbose=False)
        c = reader_coherence(Ea)
        row = {'k': k, 'alignment': alignment(Ea, Qstar),
               'effective_reader_rank': c['effective_reader_rank'],
               'off_block': r['off_block'],
               'blind': r['blind']['n_freq_full'], 'oracle': r['oracle']['n_freq_full']}
        out['forced'].append(row)
        print(f"  isotropic noise forced into the signal's top-{k:2d} reader subspace: "
              f"alignment {row['alignment']:.3f} eff rank {row['effective_reader_rank']:5.2f} "
              f"-> blind {row['blind']:2d}/11  oracle {row['oracle']:2d}/11")

    print('\n== 3. the converse: strip the real residual OUT of the signal subspace ==')
    Res0 = Qid0 - Qb0
    Ub23, _ = reader_subspace(Qb0, 6)
    perp = Res0 - project_reader(Res0, Ub23)
    out['stripped'] = []
    for tag, E in (('real residual, unchanged', Res0),
                   ('real residual, reader-aligned part removed', perp)):
        Eo = off_block_part(E)
        if float(Eo.norm()) < 1e-12:
            continue
        Qp = scale_to_off_block(Qb0, Eo, target)
        r = blind_and_oracle(Qp, tag, verbose=False)
        row = {'variant': tag, 'alignment': alignment(Eo, Qb0),
               'off_block': r['off_block'], 'blind': r['blind']['n_freq_full'],
               'oracle': r['oracle']['n_freq_full']}
        out['stripped'].append(row)
        print(f"  {tag:44s} alignment {row['alignment']:.3f} -> "
              f"blind {row['blind']:2d}/11  oracle {row['oracle']:2d}/11")

    print('\n== 4. does alignment predict the damage? ==')
    pts = [(r['alignment'], r['oracle']) for r in out['forced']] + \
          [(r['alignment'], r['oracle']) for r in out['stripped']]
    if len(pts) > 2:
        u = [p[0] for p in pts]
        v = [p[1] for p in pts]
        mu, mv = sum(u) / len(u), sum(v) / len(v)
        num = sum((a_ - mu) * (b_ - mv) for a_, b_ in zip(u, v))
        den = math.sqrt(sum((a_ - mu) ** 2 for a_ in u)
                        * sum((b_ - mv) ** 2 for b_ in v) + 1e-300)
        out['corr_recovery_vs_alignment'] = num / den
        print(f"  correlation of frequencies recovered with reader alignment, at fixed "
              f"off-block mass, over {len(pts)} perturbations: {num/den:+.3f}")

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_alignment_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
