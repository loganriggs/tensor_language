"""Why is a trained residual harder for block recovery than matched noise?

The open question left by A2-5/A2-8 and flagged in THEORY.md. At the same
off-block mass, a trained model's residual leaves 1-2 of 11 frequencies
recoverable while isotropic, structured and memorisation-shaped surrogates all
leave 11. Symmetry-breaking explains about half. This tests a mechanism for the
rest.

HYPOTHESIS. JADE reads the block structure off the coupling graph

    W_ij = sqrt( sum_m T_mij^2 )

summed over the m reader slices. What matters is therefore not how big a
perturbation is but how COHERENT it is across readers:

  * a perturbation whose slices are independent across m adds incoherently. Its
    contribution to every off-block entry concentrates around the same value with
    relative fluctuation ~1/sqrt(m), so it raises the coupling FLOOR roughly
    uniformly — and a uniform floor is exactly what the tolerance search is built
    to absorb.
  * a perturbation that is low-rank in the reader mode (the same cross-block
    pattern in every slice, up to scale) adds coherently. Its contribution is
    concentrated on SPECIFIC off-block entries, which merges specific pairs of
    blocks no matter how the threshold is chosen.

PREDICTION. At fixed off-block mass, recovery should collapse as the
perturbation's effective reader-mode rank falls, and the trained residual should
sit at the low-rank end of that curve.

This is a mechanism test, not a proof, but it makes the proof obligation precise:
a bound on the coupling graph's off-block entries in terms of the perturbation's
reader-mode spectrum.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
import a2_calibrate as cal
from fix_blind_eps import blind_and_oracle
from bq_common import interaction

torch.set_default_dtype(torch.float64)
P = m.P
FREQS = list(range(1, P // 2 + 1))


def reader_unfold(Q):
    """The (m, dim Sym^2) unfolding whose singular spectrum is the reader mode."""
    d = Q.shape[-1]
    iu, ju = torch.triu_indices(d, d)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Q)
    return Q[:, iu, ju] * sc


def reader_coherence(Q):
    """Effective reader-mode rank (participation ratio of the squared singular
    values) and the share of mass in the leading reader direction. Low effective
    rank = the same pattern repeated across readers = coherent."""
    s = torch.linalg.svdvals(reader_unfold(Q).double())
    p = s ** 2
    eff = float(p.sum() ** 2 / (p ** 2).sum().clamp_min(1e-300))
    return {'effective_reader_rank': eff, 'n_readers': int(Q.shape[0]),
            'top_share': float(p[0] / p.sum()),
            'normalised_eff_rank': eff / Q.shape[0]}


def coupling_spread(E):
    """The second hypothesis: maybe the damage comes from the off-block mass being
    concentrated on FEW pairs of blocks (a few strong couplings merge a few
    specific blocks) rather than spread thinly over all of them. Participation
    ratio over the 110 ordered pairs of distinct frequency blocks."""
    B = [m.FBLOCKS[w] for w in FREQS]
    n = len(B)
    G = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            G[i, j] = float((torch.einsum('ip,mij,jq->mpq', B[i], E, B[j]) ** 2).sum())
    off = G[G > 0]
    return {'pairs_spread_over': float(off.sum() ** 2 / (off ** 2).sum()),
            'n_pairs': int((G > 0).sum()),
            'top_pair_share': float(off.max() / off.sum())}


def off_block_part(E):
    """Keep only the part of E that lies OUTSIDE the planted frequency blocks."""
    keep = torch.zeros_like(E)
    for w, B in m.FBLOCKS.items():
        Pb = B @ B.T
        keep = keep + torch.einsum('ij,mjk,kl->mil', Pb, E, Pb)
    return E - keep


def make_perturbation(rank, seed, like):
    """A perturbation with a controlled effective reader-mode rank, supported off
    the planted blocks. rank=1 means every reader slice is the same pattern up to
    scale; rank=m means independent slices."""
    g = torch.Generator().manual_seed(seed)
    mm, d, _ = like.shape
    basis = []
    for _ in range(rank):
        A = torch.randn(d, d, generator=g).to(like)
        basis.append(0.5 * (A + A.T))
    basis = torch.stack(basis)
    coef = torch.randn(mm, rank, generator=g).to(like)
    E = torch.einsum('mr,rij->mij', coef, basis)
    return off_block_part(E)


def scale_to_off_block(Qb, E, target):
    lo, hi = 0.0, 50.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if m.fourier_power(Qb + mid * E)['off_block'] < target:
            lo = mid
        else:
            hi = mid
    return Qb + lo * E


def main():
    t0 = time.time()
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)
    cache = torch.load('a2_cache.pt', weights_only=False)
    out = {'hypothesis': 'reader-mode coherence, not magnitude, governs the damage'}

    print('== 1. how coherent is the real residual, versus the surrogates? ==')
    Qid, _ = m.canonicalise(interaction(cache[0]['p']), basis)
    Qb = m.block_project(Qid, list(range(0, P // 2 + 1)))
    Res = Qid - Qb
    target = m.fourier_power(Qid)['off_block']
    Qstar = cal.planted_family(FREQS)

    g = torch.Generator().manual_seed(0)
    Eiso = torch.randn(P, 2 * P, 2 * P, generator=g).to(Qstar)
    Eiso = off_block_part(0.5 * (Eiso + Eiso.transpose(1, 2)))

    cands = {'trained residual (seed 0)': Res,
             'isotropic noise': Eiso,
             'symmetry-breaking half of the residual': None,
             'symmetry-preserving half of the residual': None}
    S = torch.zeros(2 * P, 2 * P, device=Res.device, dtype=Res.dtype)
    S[:P, P:] = torch.eye(P, device=Res.device, dtype=Res.dtype)
    S[P:, :P] = torch.eye(P, device=Res.device, dtype=Res.dtype)
    Rsym = 0.5 * (Res + torch.einsum('ij,mjk,kl->mil', S, Res, S))
    cands['symmetry-preserving half of the residual'] = Rsym
    cands['symmetry-breaking half of the residual'] = Res - Rsym

    out['coherence'] = {}
    for tag, E in cands.items():
        Eo = off_block_part(E)
        c = reader_coherence(Eo)
        c.update(coupling_spread(Eo))
        out['coherence'][tag] = c
        print(f"  {tag:42s} eff reader rank {c['effective_reader_rank']:6.2f}/{c['n_readers']}"
              f"  | coupling spread over {c['pairs_spread_over']:5.1f}/{c['n_pairs']} "
              f"block pairs (top {c['top_pair_share']:.3f})")
    Er1 = make_perturbation(1, 1, Qstar)
    c1 = reader_coherence(Er1)
    c1.update(coupling_spread(Er1))
    out['coherence']['reader-rank-1 synthetic (damaging)'] = c1
    print(f"  {'reader-rank-1 synthetic (damaging)':42s} eff reader rank "
          f"{c1['effective_reader_rank']:6.2f}/{c1['n_readers']}  | coupling spread over "
          f"{c1['pairs_spread_over']:5.1f}/{c1['n_pairs']} block pairs "
          f"(top {c1['top_pair_share']:.3f})")

    print('\n== 2. sweep the reader-mode rank at FIXED off-block mass ==')
    print(f'   (all perturbations rescaled to off-block {target:.4f}, the trained level)')
    out['rank_sweep'] = []
    for rank in (1, 2, 3, 5, 8, 12, 23):
        E = make_perturbation(rank, seed=rank, like=Qstar)
        Qp = scale_to_off_block(Qstar, E, target)
        c = reader_coherence(off_block_part(Qp - Qstar))
        r = blind_and_oracle(Qp, f'reader-rank {rank}', verbose=False)
        row = {'planted_rank': rank, 'effective_reader_rank': c['effective_reader_rank'],
               'off_block': r['off_block'], 'blind': r['blind']['n_freq_full'],
               'oracle': r['oracle']['n_freq_full']}
        out['rank_sweep'].append(row)
        print(f"  planted rank {rank:2d} (effective {c['effective_reader_rank']:5.2f}) "
              f"off-block {row['off_block']:.4f} -> blind {row['blind']:2d}/11  "
              f"oracle {row['oracle']:2d}/11")

    print('\n== 3. the real residual and its halves, at the same off-block mass ==')
    out['real'] = []
    for tag, E in cands.items():
        if E is None:
            continue
        Qp = scale_to_off_block(Qstar, off_block_part(E), target)
        c = reader_coherence(off_block_part(Qp - Qstar))
        r = blind_and_oracle(Qp, tag, verbose=False)
        row = {'source': tag, 'effective_reader_rank': c['effective_reader_rank'],
               'off_block': r['off_block'], 'blind': r['blind']['n_freq_full'],
               'oracle': r['oracle']['n_freq_full']}
        out['real'].append(row)
        print(f"  {tag:42s} eff rank {c['effective_reader_rank']:5.2f} -> "
              f"blind {row['blind']:2d}/11  oracle {row['oracle']:2d}/11")

    print('\n== 4. does coherence predict the damage? ==')
    pts = [(r['effective_reader_rank'], r['oracle']) for r in out['rank_sweep']] + \
          [(r['effective_reader_rank'], r['oracle']) for r in out['real']]
    u = [math.log(p[0]) for p in pts]
    v = [p[1] for p in pts]
    mu, mv = sum(u) / len(u), sum(v) / len(v)
    num = sum((a_ - mu) * (b_ - mv) for a_, b_ in zip(u, v))
    den = math.sqrt(sum((a_ - mu) ** 2 for a_ in u) * sum((b_ - mv) ** 2 for b_ in v) + 1e-300)
    out['corr_recovery_vs_log_eff_rank'] = num / den
    print(f"  correlation of frequencies recovered with log(effective reader rank), "
          f"at fixed off-block mass, over {len(pts)} perturbations: {num/den:+.3f}")

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_coherence_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
