"""Fourth attempt: is it the residual's reader-energy PROFILE, not just its subspace?

Where the thread stands. Reader alignment is necessary but not sufficient
(`a2_alignment.py`): stripping the aligned part of a real residual takes recovery
from 2/11 to 7/11 blind, but forcing isotropic noise into the same subspace leaves
it harmless at 10-11/11.

The obvious difference between those two objects is HOW the energy sits inside the
shared subspace. The forced surrogate spread noise uniformly over the signal's
top-k reader directions. A real residual is produced by the same output-mixing
matrix as the signal, so it should be weighted toward the directions the signal
itself uses most — a matched profile, not a flat one.

Measured here as: project the perturbation's reader unfolding onto the SIGNAL's
reader singular vectors and read off the energy per direction. Then build a
surrogate whose profile matches the signal's and see whether it becomes damaging.

If a profile-matched surrogate is hard, the mechanism is "energy concentrated on
the reader directions the computation itself leans on". If it is still easy, the
distinguishing feature is finer than second-order statistics and the thread should
stop guessing and go structural.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
import a2_calibrate as cal
from a2_coherence import (reader_unfold, off_block_part, scale_to_off_block, FREQS)
from a2_alignment import reader_subspace, alignment
from fix_blind_eps import blind_and_oracle
from bq_common import interaction

torch.set_default_dtype(torch.float64)
P = m.P


def unvec(V, d):
    iu, ju = torch.triu_indices(d, d)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(V)
    out = torch.zeros(V.shape[0], d, d, dtype=V.dtype, device=V.device)
    out[:, iu, ju] = V / sc
    out[:, ju, iu] = V / sc
    return out


def reader_profile(E, Usig):
    """Energy of E per SIGNAL reader direction, normalised to sum to 1."""
    V = reader_unfold(E)
    c = Usig.T @ V                       # (m, dimSym2) in the signal's reader basis
    w = (c ** 2).sum(1)
    return w / w.sum().clamp_min(1e-300)


def profile_stats(w, ssig):
    """How top-heavy is the profile, and how well does it track the signal's own?"""
    k = len(w)
    top = float(w[:6].sum())
    p = ssig ** 2
    p = p / p.sum()
    # correlation of the two profiles in log space, on the directions the signal uses
    n = min(len(w), len(p))
    u = [math.log(max(float(w[i]), 1e-12)) for i in range(n)]
    v = [math.log(max(float(p[i]), 1e-12)) for i in range(n)]
    mu, mv = sum(u) / n, sum(v) / n
    num = sum((a - mu) * (b - mv) for a, b in zip(u, v))
    den = math.sqrt(sum((a - mu) ** 2 for a in u) * sum((b - mv) ** 2 for b in v) + 1e-300)
    return {'top6_share': top, 'profile_corr_with_signal': num / den,
            'effective_directions': float(w.sum() ** 2 / (w ** 2).sum().clamp_min(1e-300))}


def build_with_profile(target_w, Usig, like, seed=0):
    """Isotropic off-block noise reshaped to have a prescribed reader-energy
    profile in the signal's reader basis."""
    g = torch.Generator().manual_seed(seed)
    mm, d, _ = like.shape
    E = torch.randn(mm, d, d, generator=g).to(like)
    E = off_block_part(0.5 * (E + E.transpose(1, 2)))
    V = reader_unfold(E)
    c = Usig.T @ V
    cur = (c ** 2).sum(1).clamp_min(1e-30)
    scale = (target_w.clamp_min(1e-30) / cur).sqrt()
    c = c * scale[:, None]
    return off_block_part(unvec(Usig @ c, d))


def main():
    t0 = time.time()
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)
    cache = torch.load('a2_cache.pt', weights_only=False)
    allw = list(range(0, P // 2 + 1))
    out = {}

    Qid, _ = m.canonicalise(interaction(cache[0]['p']), basis)
    Qb = m.block_project(Qid, allw)
    Res = Qid - Qb
    target = m.fourier_power(Qid)['off_block']
    Usig, ssig = reader_subspace(Qb, P)

    print('== 1. reader-energy profiles in the SIGNAL\'s reader basis ==')
    g = torch.Generator().manual_seed(0)
    Eiso = torch.randn(P, 2 * P, 2 * P, generator=g).to(Res)
    Eiso = off_block_part(0.5 * (Eiso + Eiso.transpose(1, 2)))
    out['profiles'] = {}
    for tag, E in (('trained residual', Res), ('isotropic noise', Eiso)):
        w = reader_profile(off_block_part(E), Usig)
        st = profile_stats(w, ssig)
        out['profiles'][tag] = {**st, 'profile': [float(v) for v in w]}
        print(f"  {tag:20s} top-6 share {st['top6_share']:.3f} | effective directions "
              f"{st['effective_directions']:5.2f}/{P} | profile corr with the signal's own "
              f"{st['profile_corr_with_signal']:+.3f}")
    psig = (ssig ** 2 / (ssig ** 2).sum())
    print(f"  {'the signal itself':20s} top-6 share {float(psig[:6].sum()):.3f} | "
          f"effective directions {float(psig.sum()**2/(psig**2).sum()):5.2f}/{P}")
    out['profiles']['signal'] = {'top6_share': float(psig[:6].sum()),
                                 'profile': [float(v) for v in psig]}

    print('\n== 2. surrogates built to a prescribed reader profile ==')
    Qstar = cal.planted_family(FREQS)
    Ust, sst = reader_subspace(Qstar, P)
    out['built'] = []
    specs = {'flat (the a2_alignment surrogate)': torch.ones(P).to(Res) / P,
             "matched to the signal's own profile": (sst ** 2 / (sst ** 2).sum()).to(Res),
             "matched to the real residual's profile": reader_profile(off_block_part(Res),
                                                                     Usig).to(Res)}
    for tag, w in specs.items():
        E = build_with_profile(w, Ust, Qstar, seed=1)
        if float(E.norm()) < 1e-12:
            continue
        Qp = scale_to_off_block(Qstar, E, target)
        Ea = off_block_part(Qp - Qstar)
        wa = reader_profile(Ea, Ust)
        st = profile_stats(wa, sst)
        r = blind_and_oracle(Qp, tag, verbose=False)
        row = {'spec': tag, 'top6_share': st['top6_share'],
               'effective_directions': st['effective_directions'],
               'alignment': alignment(Ea, Qstar), 'off_block': r['off_block'],
               'blind': r['blind']['n_freq_full'], 'oracle': r['oracle']['n_freq_full']}
        out['built'].append(row)
        print(f"  {tag:38s} top-6 {row['top6_share']:.3f} eff dirs "
              f"{row['effective_directions']:5.2f} align {row['alignment']:.3f} -> "
              f"blind {row['blind']:2d}/11  oracle {row['oracle']:2d}/11")

    print('\n== 3. reference points at the same off-block mass ==')
    out['reference'] = []
    for tag, E, base in (('real residual on the real signal', Res, Qb),
                         ('real residual transplanted onto the planted signal',
                          Res, Qstar),
                         ('isotropic noise on the planted signal', Eiso, Qstar)):
        Qp = scale_to_off_block(base, off_block_part(E), target)
        r = blind_and_oracle(Qp, tag, verbose=False)
        row = {'case': tag, 'off_block': r['off_block'],
               'blind': r['blind']['n_freq_full'], 'oracle': r['oracle']['n_freq_full']}
        out['reference'].append(row)
        print(f"  {tag:52s} -> blind {row['blind']:2d}/11  oracle {row['oracle']:2d}/11")

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_profile_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
