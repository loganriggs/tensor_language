"""What structure does MLP1's output distribution actually have?

The question that decides which compression is right. An MDL-optimal description of the
layer depends on the data's shape, and four different shapes call for four different
codes:

    dense low-rank subspace      -> PCA + Gaussian code (what §12-16 implicitly assumed)
    heavy-tailed / sparse        -> sparse dictionary code (atoms + few active coeffs)
    mixture / clustered          -> per-cluster codes (the §16 heterogeneity hints here)
    hierarchical / gated         -> leader-conditioned code (tail coded given leader)

Each hypothesis has a cheap sufficient statistic, measured here on 153,900 positions
(fit rows) with the held-out rows for anything that could overfit:

  1. SPECTRUM. Eigenvalue decay of the full 1152-dim output covariance: effective rank,
     dims for 50/90/99% of energy, and the held-out energy-vs-k curve. Decides "is it a
     32-dim subspace at all".
  2. SPARSITY. Excess kurtosis of coefficients along top PCA directions vs random
     directions. Gaussian (dense) ~ 0; sparse features >> 0. This is the direct test of
     "should the code be sparse".
  3. DOCUMENT MIXTURE. Intra-class correlation by document for the leading coefficients:
     share of each coefficient's variance that is between-document rather than
     within-document. §16 found row-group heterogeneity; this quantifies it per
     direction. High ICC -> mixture structure -> per-cluster or contextual code.
  4. HIERARCHY. Spearman correlation of |c_i| across the top 8 directions, and of the
     leader's |c_0| against the tail's total energy. Gating (tail active only when the
     leader is) shows up as strong positive dependence of magnitudes; independent parts
     show ~0.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, FW, LAYER, DEV

OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_data_structure_results.json')


@torch.no_grad()
def collect(seqs):
    accs = []
    for i in range(0, seqs.shape[0], 6):
        acc = []
        fwd(seqs[i:i + 6].to(DEV), collect=LAYER, acc=acc)
        accs.append(acc[0])
    return torch.cat(accs, 0)


def spearman_mat(A):
    """Spearman correlation matrix of columns of A."""
    R = A.argsort(0).argsort(0).double()
    R = R - R.mean(0, keepdim=True)
    R = R / R.norm(dim=0, keepdim=True).clamp_min(1e-30)
    return R.T @ R


def main():
    t0 = time.time()
    n_seq, T = 300, 513
    Y = collect(FW[0:n_seq, :T])
    Yt = collect(FW[452:512, :T])
    Ybar = Y.mean(0)
    Yc = (Y - Ybar).float()
    Ytc = (Yt - Yt.mean(0)).float()
    n, d = Yc.shape
    print(f'{n:,} fit positions, {Ytc.shape[0]:,} held-out positions, dim {d}\n')
    out = {}

    # ---- 1. spectrum ----
    C = Yc.T @ Yc / n
    ev = torch.linalg.eigvalsh(C.double()).flip(0).clamp_min(0)
    share = ev / ev.sum()
    cum = share.cumsum(0)
    er = float(ev.sum() ** 2 / (ev ** 2).sum())
    dims = {q: int((cum < q).sum()) + 1 for q in (0.5, 0.9, 0.99)}
    _, _, Vh = torch.linalg.svd(Yc, full_matrices=False)
    tot_t = float(Ytc.pow(2).sum())
    curve = {}
    for k in (8, 16, 32, 64, 128, 256, 512):
        Q = orth(Vh[:k].T)
        curve[k] = float((Ytc @ Q).pow(2).sum()) / tot_t
    out['spectrum'] = {'effective_rank': er, 'dims_for': dims,
                       'top1_share': float(share[0]),
                       'heldout_energy_vs_k': curve}
    print('== 1. spectrum ==')
    print(f'  effective rank of the full output: {er:.0f} of {d}')
    print(f'  dims for 50/90/99% of energy (in-sample): {dims[0.5]}/{dims[0.9]}/'
          f'{dims[0.99]}')
    print(f'  held-out energy vs k: ' +
          ' '.join(f'k={k}:{100*v:.0f}%' for k, v in curve.items()))

    # ---- 2. sparsity ----
    def kurt(c):
        c = c - c.mean(0, keepdim=True)
        return ((c ** 4).mean(0) / (c ** 2).mean(0).clamp_min(1e-30) ** 2) - 3.0

    c_pca = Yc @ Vh[:32].T
    g = torch.Generator(device=DEV).manual_seed(0)
    Qr = orth(torch.randn(d, 32, device=DEV, generator=g))
    c_rnd = Yc @ Qr
    k_pca, k_rnd = kurt(c_pca), kurt(c_rnd)
    out['sparsity'] = {'excess_kurtosis_pca_median': float(k_pca.median()),
                       'excess_kurtosis_pca_max': float(k_pca.max()),
                       'excess_kurtosis_pca_per_dir_top8': [round(float(v), 1)
                                                            for v in k_pca[:8]],
                       'excess_kurtosis_random_median': float(k_rnd.median())}
    print('\n== 2. sparsity (excess kurtosis; Gaussian/dense = 0) ==')
    print(f'  top-32 PCA dirs: median {float(k_pca.median()):.1f}, max '
          f'{float(k_pca.max()):.0f}')
    print(f'  per-direction, top 8: '
          f'{[round(float(v), 1) for v in k_pca[:8]]}')
    print(f'  random dirs: median {float(k_rnd.median()):.1f}')

    # ---- 3. document mixture ----
    doc = torch.arange(n, device=DEV) // (T - 1)
    n_doc = int(doc.max()) + 1
    icc = []
    for j in range(8):
        c = c_pca[:, j]
        dm = torch.zeros(n_doc, device=DEV).index_add_(0, doc, c)
        cnt = torch.zeros(n_doc, device=DEV).index_add_(0, doc,
                                                        torch.ones_like(c))
        dm = dm / cnt
        between = float((dm[doc] - c.mean()).pow(2).mean())
        total = float(c.var())
        icc.append(between / max(total, 1e-30))
    out['document_mixture'] = {'icc_top8': [round(v, 3) for v in icc]}
    print('\n== 3. document mixture (share of variance that is between-document) ==')
    print(f'  ICC, top 8 directions: {[round(v, 2) for v in icc]}')

    # ---- 4. hierarchy ----
    S = spearman_mat(c_pca[:, :8].abs())
    off = S - torch.eye(8, dtype=S.dtype, device=S.device)
    tail_energy = (Yc @ Vh[8:32].T).pow(2).sum(1)
    lead = c_pca[:, 0].abs()
    r_lead_tail = float(spearman_mat(torch.stack([lead, tail_energy], 1))[0, 1])
    out['hierarchy'] = {'mean_abs_offdiag_spearman': float(off.abs().mean()),
                        'max_offdiag_spearman': float(off.max()),
                        'leader_vs_tail_energy': r_lead_tail}
    print('\n== 4. hierarchy (magnitude co-activation of the top 8) ==')
    print(f'  mean |off-diagonal| Spearman of |c_i|: {float(off.abs().mean()):.2f} '
          f'(max {float(off.max()):.2f})')
    print(f'  leader |c_0| vs tail energy (dims 9-32): {r_lead_tail:+.2f}')

    # ---- verdict ----
    v = []
    if curve[32] < 0.5:
        v.append(f'NOT a 32-dim story: 32 dims hold {100*curve[32]:.0f}% held-out; '
                 f'90% needs ~{dims[0.9]} dims in-sample')
    if float(k_pca.median()) > 5:
        v.append('strongly heavy-tailed along its principal directions -> a sparse '
                 'code fits the coefficients better than a Gaussian one')
    if max(icc) > 0.3:
        v.append(f'document-mixture structure is real (ICC up to {max(icc):.2f}) '
                 f'-> part of the "activation" is document identity, not token '
                 f'computation')
    if r_lead_tail > 0.3:
        v.append('leader and tail magnitudes co-activate -> hierarchical/gated '
                 'code is plausible')
    elif r_lead_tail < 0.1:
        v.append('no leader-tail gating; directions activate independently')
    out['verdict'] = v
    print('\nVERDICT:')
    for s in v:
        print(f'  - {s}')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
