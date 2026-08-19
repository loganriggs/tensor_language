"""Sample-size control for §13-§15: does the answer survive 4.8x more fitting data?

Everything downstream of the Shapley run rests on a PCA basis fit on 32,768 positions
(FW[0:256, :128]). The recorded basis-rotation trap in §15 already shows the tail of
that basis is corpus-sensitive, so the concern is concrete, not hypothetical: with a
better-estimated basis, (a) the attribution's shape (ten-ish directions, 27% leader)
could change, and (b) the weight-derived basis -- which cannot overfit a sample, since
it never sees one -- could close the gap on the data basis, whose recorded 70.5% output
energy was measured IN-SAMPLE.

Fit data: FW[0:300, :513] = 153,900 positions (4.8x), rows disjoint from the held-out
eval rows FW[300:452] that every v(S) is scored on. (The obvious 512-row corpus would
overlap the eval rows -- basis fit on the evaluation data -- so it is not used.)

Four questions, in order of cost:

1. BASIS STABILITY. Principal angles between the old and new top-32 subspaces; cosine
   between old and new direction i for the Shapley leaders. If the leaders are stable
   and only the tail rotates, §15's naming stands as-is.
2. HELD-OUT ENERGY -- the exoneration test. Output energy captured by each basis on
   positions neither basis was fit on: data-32k, data-154k, weight SVD of Down. The
   in-sample 70.5% vs 44.3% comparison was unfair to the weights exactly insofar as
   the 70.5% was overfit; held-out, the gap is the true gap.
3. SHAPLEY ON THE REFIT BASIS. Full rerun, 20 permutations, same protocol as §13.
   Compare participation ratio, leader share, and the per-direction values carried
   over through the direction correspondence (new dir j matched to old dir i by
   |cosine|).
4. THE WEIGHT VERDICT, updated. Joint causal effect of the weight span, re-scored,
   with the held-out energies alongside, so the reply to "might exonerate the weights"
   is a measurement rather than an expectation.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import (fwd, held, orth, m, FW, LAYER, DEV, PATCH)
from bilin18_shapley import value

NDIR = 32
N_PERM = 20
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_shapley_bigdata_results.json')


@torch.no_grad()
def collect(seqs):
    accs = []
    for i in range(0, seqs.shape[0], 6):
        acc = []
        fwd(seqs[i:i + 6].to(DEV), collect=LAYER, acc=acc)
        accs.append(acc[0])
    return torch.cat(accs, 0)


def main():
    t0 = time.time()
    base = held(); BASE = float(base.mean())
    out = {'base_ce': BASE}

    # the three bases
    Y_small = collect(FW[0:256, :128])            # §13's fit set, 32k positions
    Y_big = collect(FW[0:300, :513])              # 4.8x, rows disjoint from eval rows
    Y_test = collect(FW[452:512, :513])           # held-out for energy, disjoint from both
    print(f'fit sets: small {Y_small.shape[0]:,} | big {Y_big.shape[0]:,} | '
          f'energy test {Y_test.shape[0]:,} positions')

    def pca(Y):
        Yb = Y.mean(0)
        _, Sv, Vh = torch.linalg.svd(Y - Yb, full_matrices=False)
        return orth(Vh[:NDIR].T), Yb

    Q_small, _ = pca(Y_small)
    Q_big, Ybar_big = pca(Y_big)
    Dw = m.transformer.h[LAYER].mlp.Down.weight.detach().float()
    U, S, _ = torch.linalg.svd(Dw, full_matrices=False)
    Q_w = orth(U[:, :NDIR])

    # ---- 1. basis stability ----
    sv = torch.linalg.svdvals(Q_small.T @ Q_big)          # cosines of principal angles
    R_old = json.load(open('bilin18_shapley_results.json'))
    phi_old = torch.tensor(R_old['shapley'])
    order_old = phi_old.argsort(descending=True)
    # match each old direction to its best new direction
    C = (Q_small.T @ Q_big).abs()
    match = C.argmax(1)
    out['stability'] = {'principal_cosines_min': float(sv.min()),
                        'principal_cosines_median': float(sv.median()),
                        'n_angles_above_0p9': int((sv > 0.9).sum())}
    print(f'\n== 1. how much did the basis move with 4.8x data? ==')
    print(f'  principal cosines of the two 32-dim subspaces: median '
          f'{float(sv.median()):.3f}, min {float(sv.min()):.3f}, '
          f'{int((sv > 0.9).sum())}/32 above 0.9')
    print(f'  the §13 Shapley leaders, old direction -> best new direction:')
    lead = []
    for r in range(6):
        i = int(order_old[r])
        j = int(match[i]); c = float(C[i, j])
        lead.append({'old': i, 'new': j, 'cos': c})
        print(f'    #{r+1} old dir {i:2d} -> new dir {j:2d}  |cos| {c:.3f}')
    out['stability']['leaders'] = lead

    # ---- 2. held-out energy: the exoneration test ----
    Yt = Y_test - Y_test.mean(0)
    tot = float(Yt.pow(2).sum())
    def en(Q): return float((Yt @ Q).pow(2).sum()) / tot
    e_small, e_big, e_w = en(Q_small), en(Q_big), en(Q_w)
    out['heldout_energy'] = {'data_32k': e_small, 'data_154k': e_big, 'weight_svd': e_w}
    print(f'\n== 2. output energy on held-out positions (fit-set overfitting test) ==')
    print(f'  data basis, 32k fit:   {100*e_small:5.1f}%')
    print(f'  data basis, 154k fit:  {100*e_big:5.1f}%')
    print(f'  weight SVD (no fit):   {100*e_w:5.1f}%')

    # ---- 3. Shapley on the refit basis ----
    print(f'\n== 3. Shapley rerun on the big-data basis ({N_PERM} permutations) ==')
    v_all = value(Q_big, list(range(NDIR)), Ybar_big, base)
    g = torch.Generator().manual_seed(0)
    phi = torch.zeros(NDIR, N_PERM, dtype=torch.float64)
    for p in range(N_PERM):
        perm = torch.randperm(NDIR, generator=g).tolist()
        prev = 0.0; cur = []
        for pos, i in enumerate(perm):
            cur.append(i)
            v = v_all if pos == NDIR - 1 else value(Q_big, cur, Ybar_big, base)
            phi[i, p] = v - prev; prev = v
        if (p + 1) % 5 == 0:
            print(f'  permutation {p+1}/{N_PERM}', flush=True)
    est = phi.mean(1); se = phi.std(1) / N_PERM ** 0.5
    pr = float(est.sum() ** 2 / (est ** 2).sum())
    srt = est.sort(descending=True).values
    tot_phi = float(est.sum())
    out['shapley_big'] = {'v_all': v_all, 'phi': est.tolist(), 'se': se.tolist(),
                          'participation_ratio': pr,
                          'leader_share': float(srt[0] / tot_phi),
                          'top4_share': float(srt[:4].sum() / tot_phi),
                          'top8_share': float(srt[:8].sum() / tot_phi)}
    print(f'\n  joint effect of the refit span: {v_all:+.4f} '
          f'(was {R_old["v_all"]:+.4f})')
    print(f'  participation ratio {pr:.1f} of 32   (was '
          f'{R_old["participation_ratio"]:.1f})')
    print(f'  leader share {100*srt[0]/tot_phi:.0f}%  (was 27%) | top-8 '
          f'{100*srt[:8].sum()/tot_phi:.0f}%  (was 67%)')
    # carry the correspondence through: does the old leader map to the new leader?
    new_order = est.argsort(descending=True)
    print(f'  new leaders (by refit Shapley): '
          f'{[int(i) for i in new_order[:6]]}')
    print(f'  old leaders mapped through the basis correspondence: '
          f'{[l["new"] for l in lead]}')

    # ---- 4. the weight verdict, updated ----
    v_w = value(Q_w, list(range(NDIR)), Y_big.mean(0), base)
    out['weight_joint_dce'] = v_w
    print(f'\n== 4. the weight basis, re-scored ==')
    print(f'  joint causal effect: weight span {v_w:+.4f} vs refit data span '
          f'{v_all:+.4f}  ({100*v_w/v_all:.0f}%)')
    print(f'  held-out energy:     weight {100*e_w:.1f}% vs data-154k '
          f'{100*e_big:.1f}%')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
