"""A budget-free replacement for the statistic §12 showed is budget-dependent.

§12 established that "individual directions explain X% of the layer's effect" is not a
property of the layer: on MLP1, unchanged, X runs from 64% at 4 removed directions to 9%
at 32 to 14% at 82. The cause is that solo ablation and joint ablation are two arbitrary
points on a curve, and the gap between them grows with the size of the ablation set. Both
this program and the repository quoted numbers from that curve as though they measured
the layer.

There is a standard fix, and the reason it is the right one here is precise. The Shapley
value of direction i is its marginal contribution averaged over ALL coalition sizes,

    phi_i = E_pi [ v(P_i^pi + i) - v(P_i^pi) ]

with pi a uniformly random ordering and P_i^pi the directions preceding i. Averaging over
orderings averages over budgets, so the answer does not depend on a budget the analyst
picked. And the efficiency axiom gives exactly what the solo-sum lacked:

    sum_i phi_i  =  v(all)  -  v(none)     EXACTLY, by construction.

The "9% attributable, 91% unexplained" gap cannot occur. Whatever interference exists is
distributed back onto the directions responsible for it rather than left in a residual.

So the question §74 wanted to ask -- is this layer's causal content concentrated in a few
directions or spread across many? -- becomes answerable without a budget choice: fit the
Shapley values and look at how concentrated they are. Reported three ways, none of which
can be moved by choosing a different ablation-set size:

    participation ratio  (sum phi)^2 / sum phi^2, the scale-free count of directions
                         actually carrying the effect; 1 = all in one, 32 = perfectly even
    top-k share          what fraction of the total the largest few carry
    sign structure       negative phi_i means a direction whose removal HELPS once the
                         others are gone -- invisible to solo ablation, and a thing the
                         "irreducibly distributed" reading has no room for

GATE. The efficiency axiom is checked numerically against a direct measurement of
v(all): a Monte-Carlo Shapley estimate satisfies it only up to sampling error, so the gap
is the honest error bar on everything else here.

Compared against two references measured on the same directions: the solo attribution
(§74's numerator) and the joint effect (its denominator).
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import (fwd, held, collect_out, orth, TRAIN, HELD,
                                   LAYER, DEV, PATCH)

NDIR = 32
N_PERM = 20
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_shapley_results.json')


def value(Q, cols, Ybar, base):
    """v(S) = increase in held-out CE from mean-ablating the span of the chosen columns."""
    if len(cols) == 0:
        return 0.0
    Qs = Q[:, cols]
    PATCH[LAYER] = (Qs, Ybar @ Qs)
    try:
        return float((held() - base).mean())
    finally:
        PATCH.pop(LAYER)


def main():
    t0 = time.time()
    base = held(); BASE = float(base.mean())
    Ytr = collect_out(TRAIN, LAYER)
    Ybar = Ytr.mean(0)
    Yc = Ytr - Ybar
    _, Sv, Vh = torch.linalg.svd(Yc, full_matrices=False)
    Q = orth(Vh[:NDIR].T)
    print(f'base CE {BASE:.4f} | {NDIR} directions of MLP{LAYER} | '
          f'{N_PERM} permutations = {N_PERM * NDIR} evaluations\n')

    v_all = value(Q, list(range(NDIR)), Ybar, base)
    solo = [value(Q, [i], Ybar, base) for i in range(NDIR)]
    solo_sum = sum(solo)
    print(f'reference points: joint removal of all {NDIR} = {v_all:+.4f} nats | '
          f'sum of solos = {solo_sum:+.4f} ({100*solo_sum/v_all:.1f}%)\n')

    g = torch.Generator().manual_seed(0)
    phi = torch.zeros(NDIR, N_PERM, dtype=torch.float64)
    for p in range(N_PERM):
        order = torch.randperm(NDIR, generator=g).tolist()
        prev_v = 0.0
        cur = []
        for pos, i in enumerate(order):
            cur.append(i)
            v = v_all if pos == NDIR - 1 else value(Q, cur, Ybar, base)
            phi[i, p] = v - prev_v
            prev_v = v
        est = phi[:, :p + 1].mean(1)
        print(f'  permutation {p+1:2d}/{N_PERM}: running sum of Shapley values '
              f'{float(est.sum()):+.4f} (target {v_all:+.4f}, '
              f'gap {float(est.sum())-v_all:+.4f})', flush=True)

    est = phi.mean(1)
    se = phi.std(1) / N_PERM ** 0.5
    tot = float(est.sum())
    pr = float(est.sum() ** 2 / (est ** 2).sum())
    srt = est.sort(descending=True).values
    top4 = float(srt[:4].sum() / tot)
    top8 = float(srt[:8].sum() / tot)
    neg = int((est < 0).sum())

    # the same three statistics computed on the solo attribution, for contrast
    st = torch.tensor(solo, dtype=torch.float64)
    pr_solo = float(st.sum() ** 2 / (st ** 2).sum())

    out = {'base_ce': BASE, 'n_dir': NDIR, 'n_perm': N_PERM, 'v_all': v_all,
           'solo': solo, 'solo_sum': solo_sum,
           'shapley': est.tolist(), 'shapley_se': se.tolist(),
           'shapley_sum': tot, 'efficiency_gap': tot - v_all,
           'participation_ratio': pr, 'participation_ratio_solo': pr_solo,
           'top4_share': top4, 'top8_share': top8, 'n_negative': neg}

    print(f'\n== the gate ==')
    print(f'  sum of Shapley values {tot:+.4f} against a directly measured '
          f'{v_all:+.4f}: gap {tot-v_all:+.4f} '
          f'({100*abs(tot-v_all)/abs(v_all):.1f}% -- this is the sampling error bar)')
    print(f'\n== how concentrated is MLP{LAYER}\'s causal content, with no budget choice? ==')
    print(f'  directions actually carrying the effect (participation ratio): '
          f'{pr:.1f} of {NDIR}')
    print(f'  largest 4 carry {100*top4:.0f}% of the total; largest 8 carry '
          f'{100*top8:.0f}%')
    print(f'  directions with NEGATIVE contribution (removal helps, given the rest): '
          f'{neg} of {NDIR}')
    print(f'  for contrast, the same participation ratio on the solo attribution: '
          f'{pr_solo:.1f}')
    print(f'\n  top 6 directions by Shapley value:')
    idx = est.argsort(descending=True)[:6]
    for r, i in enumerate(idx.tolist()):
        print(f'    #{r+1} direction {i:2d}: {est[i]:+.4f} +/- {se[i]:.4f} nats '
              f'({100*float(est[i])/tot:4.1f}%)   solo was {solo[i]:+.4f}')

    if pr < 8:
        v = (f'concentrated: {pr:.1f} effective directions of {NDIR} carry the layer, '
             f'which is NOT what "irreducibly distributed" describes')
    elif pr > 0.6 * NDIR:
        v = (f'genuinely spread: {pr:.1f} effective directions of {NDIR}, so the '
             f'distributed reading survives a budget-free measurement')
    else:
        v = (f'intermediate: {pr:.1f} effective directions of {NDIR} -- neither a few '
             f'nameable parts nor an even smear')
    out['verdict'] = v
    print(f'\nVERDICT: {v}')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
