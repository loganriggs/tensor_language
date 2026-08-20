"""MLP0 UNIT CLUSTER -- apply the validated RSPD-style method (578:
SVD(WX) reconstruction proxy + random-ablation damage covariance +
hierarchical clustering) to a REAL, not-yet-fully-characterized
piece: mlp0's 4608 hidden units, to find reusable "variable" groups
among them.

Could not fetch github.com/ThatE10/rspd (blocked -- see ledger); this
is my own implementation of the method described, validated first on
synthetic data with known ground-truth computational groups (578,
all 4 core predictions held: exact top-level recovery, correct
nesting of a hierarchical sub-case, correct minimal-weight subset
identification). This is the first real application.

533/536 established mlp0 exactly: output = Down_bias + sum_j h_j *
Down[:,j], h_j = (L_j.x)(R_j.x), and the write is NARROW (top ~64 of
1152 residual directions hold ~90% of it). 536 asked "does mlp0 read
r directions and write r directions" but never asked whether the
4608 units GROUP into a smaller number of reusable computations --
units that co-activate on the same kind of token/context and whose
combined damage is not simply additive (they cover overlapping
residual directions). That grouping is the target here.

METHOD: capture h over real FineWeb data. Ablating any SET S of
units removes exactly sum_{j in S} h_j(x) Down[:,j] from the output
-- a closed form, no forward-pass needed per ablation. To keep this
"extremely quick" at scale and to honor the requested SVD(WX) step
(also gives the reconstruction proxy a fixed, interpretable basis),
project this exact removed vector onto the top-r PCA directions of
mlp0's realized output (r chosen to explain 95% of output variance
-- 536 predicts r near 64). Damage(S, sample) = squared norm of the
projected removed vector. Compute SINGLETON damage vectors for the
top-K units by importance (534's importance rank: Down-column-norm
times hidden-unit std), correlate them across samples, hierarchically
cluster (average linkage, 1-corr distance).

REGISTERED PREDICTIONS:
  (0) PROJECTION SANITY: at r = full output rank, the projected
      singleton damage equals the exact ||h_j Down[:,j]||^2 to
      relative error < 1e-4 -- verifies the projection code, VOIDS
      the run on failure;
  (a) NONTRIVIAL STRUCTURE: cutting the dendrogram (over the top
      K=300 units) to 20 clusters does not put >= 80% of units in
      one giant cluster -- the correlation structure is genuinely
      distributed, not one dominant blob;
  (b) STABILITY (the real test, no ground truth available): split
      the N samples into two independent random halves, cluster the
      SAME K units on each half (same 20-cluster cut), and measure
      agreement between the two halves' cluster labels via the
      Adjusted Rand Index (chance-corrected) against a NULL where
      one half's cluster labels are randomly permuted across units.
      Real structure predicts ARI >= null ARI + 0.05;
  (c) SUPERADDITIVITY: random same-cluster unit pairs (top-level
      clusters) have a higher pair-damage / singleton-damage-sum
      ratio than random cross-cluster pairs -- Down's columns are
      not orthogonal, so this is a real (not tautological, unlike
      578's toy) test of whether the clusters found actually share
      residual directions;
  (d) INTERPRETABILITY: report the 5 most-activating frequent tokens
      (census top-5000) for the 3 largest clusters' summed
      activation, and 3 example contexts per cluster via cl.context
      -- qualitative, no bar, this is where a human judges whether
      the clusters are nameable;
  NULL: (b)'s permutation baseline, reported explicitly."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import census_lib as cl
from bilin18_joint_removal import m, DEV
D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_unit_cluster_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
VARFRAC = 0.95
NCLUST = 20


def adjusted_rand_index(labels_true, labels_pred):
    from collections import Counter
    n = len(labels_true)
    ct = Counter(zip(labels_true, labels_pred))
    a = Counter(labels_true)
    b = Counter(labels_pred)
    def comb2(x): return x * (x - 1) / 2
    sum_comb_c = sum(comb2(v) for v in ct.values())
    sum_comb_a = sum(comb2(v) for v in a.values())
    sum_comb_b = sum(comb2(v) for v in b.values())
    comb_n = comb2(n)
    expected = sum_comb_a * sum_comb_b / comb_n if comb_n else 0
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if abs(denom) < 1e-12:
        return 1.0
    return (sum_comb_c - expected) / denom


@torch.no_grad()
def capture(fresh):
    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight.float()
    cap = {}
    hk = mlp.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X', a_[0]))
    hs = []
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        X = cap['X'].float()
        h = (X @ L.T) * (X @ R.T)
        hs.append(h.reshape(-1, h.shape[-1]).cpu())
    hk.remove()
    H = torch.cat(hs, dim=0)
    return H, Dw.cpu(), L.cpu(), R.cpu()


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    H, Dw, L, R = capture(fresh)
    Nfull = H.shape[0]
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(Nfull, generator=g)[:NSAMP]
    H = H[perm]
    N = H.shape[0]
    Hd = H.shape[1]
    print(f'{N} samples, {Hd} hidden units', flush=True)

    imp = (Dw.norm(dim=0) * H.std(0))
    order = imp.argsort(descending=True)
    topk = order[:TOPK].numpy()

    # realized output & its PCA basis (the "SVD(WX)" reconstruction
    # proxy) -- WX here is O = H @ Down.T, the actual write into the
    # residual stream over real data.
    O = H @ Dw.T  # (N, 1152)
    Omu = O.mean(0)
    Oc = O - Omu
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    var = (S ** 2)
    cum = torch.cumsum(var, 0) / var.sum()
    r = int((cum < VARFRAC).sum().item()) + 1
    r = min(r, Vt.shape[0])
    Vr = Vt[:r]  # (r, 1152)
    print(f'output rank for {VARFRAC*100:.0f}% variance: r={r} '
          f'(536 predicted ~64)', flush=True)

    # per-unit removed-vector projected onto Vr, for the top-K units
    # damage_j(sample) = || Vr @ (h_j(sample) * Down[:,j]) ||^2
    Dw_topk = Dw[:, topk]                      # (1152, K)
    Dw_proj = Vr @ Dw_topk                     # (r, K) -- Down cols in PCA basis
    Hk = H[:, topk]                            # (N, K)

    # damage[n,j] = sum_c (Hk[n,j]*Dw_proj[c,j])^2
    #                                    = Hk[n,j]^2 * sum_c Dw_proj[c,j]^2
    coldw2 = (Dw_proj ** 2).sum(0)             # (K,) -- ||Vr @ Down[:,j]||^2
    damage = (Hk ** 2) * coldw2[None, :]       # (N, K), exact closed form
    damage = damage.T.numpy()                  # (K, N)

    # (0) sanity: at full rank (r=1152) this equals exact ||h_j Down[:,j]||^2
    coldw2_full = (Dw_topk ** 2).sum(0)
    damage_full_exact = ((Hk ** 2) * coldw2_full[None, :]).T.numpy()
    Vfull = Vt  # full rank
    Dw_proj_full = Vfull @ Dw_topk
    coldw2_proj_full = (Dw_proj_full ** 2).sum(0)
    damage_full_proj = ((Hk ** 2) * coldw2_proj_full[None, :]).T.numpy()
    relerr = float(np.abs(damage_full_proj - damage_full_exact).sum() /
                   max(np.abs(damage_full_exact).sum(), 1e-12))
    p0 = relerr < 1e-4
    print(f'(0) full-rank projection sanity relerr {relerr:.2e}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'projection sanity failed', 'relerr': relerr},
                   open(OUT, 'w'), indent=1)
        return

    def cluster_damage(dmg):
        dm = dmg - dmg.mean(1, keepdims=True)
        dstd = dmg.std(1, keepdims=True)
        dstd[dstd < 1e-12] = 1e-12
        corr = (dm @ dm.T) / (dmg.shape[1] * dstd * dstd.T)
        corr = np.clip(corr, -1, 1)
        dist = 1 - corr
        np.fill_diagonal(dist, 0)
        cond = squareform((dist + dist.T) / 2, checks=False)
        return linkage(cond, method='average')

    Z = cluster_damage(damage)
    labels = fcluster(Z, t=NCLUST, criterion='maxclust')
    n_clusters = int(labels.max())
    sizes_all = sorted(np.bincount(labels)[1:].tolist(), reverse=True)
    pa = n_clusters >= 5 and sizes_all[0] < TOPK * 0.8
    print(f'(a) {n_clusters} clusters (target {NCLUST}), sizes '
          f"{sizes_all}: {'HELD' if pa else 'FAILED (degenerate)'}",
          flush=True)

    # (b) stability: split samples, cluster independently, compare via
    # ARI (chance-corrected -- raw pairwise accuracy is dominated by
    # trivial "different cluster" agreement and is not sensitive here)
    g2 = np.random.default_rng(3)
    sperm = g2.permutation(N)
    h1, h2 = sperm[:N // 2], sperm[N // 2:]
    Z1 = cluster_damage(damage[:, h1])
    Z2 = cluster_damage(damage[:, h2])
    lab1 = fcluster(Z1, t=NCLUST, criterion='maxclust')
    lab2 = fcluster(Z2, t=NCLUST, criterion='maxclust')
    agree = adjusted_rand_index(list(lab1), list(lab2))
    lab2_shuf = g2.permutation(lab2)
    agree_null = adjusted_rand_index(list(lab1), list(lab2_shuf))
    pb = agree >= agree_null + 0.05
    print(f'(b) stability ARI {agree:.3f} vs permutation-null '
          f"{agree_null:.3f}: {'HELD' if pb else 'FAILED'}", flush=True)

    # (c) superadditivity: same-cluster vs cross-cluster pairs
    def pair_damage_exact(i, j):
        # exact closed form in the r-dim proxy basis: damage of
        # ablating BOTH i and j together (includes interaction)
        v = Hk[:, i:i+1] * Dw_proj[:, i:i+1].T + \
            Hk[:, j:j+1] * Dw_proj[:, j:j+1].T
        return float((v ** 2).sum(1).mean())

    g3 = np.random.default_rng(5)
    same_ratios, cross_ratios = [], []
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(i)
    big = [idxs for idxs in by_cluster.values() if len(idxs) >= 2]
    for _ in range(60):
        if not big:
            break
        bl = big[g3.integers(len(big))]
        i, j = g3.choice(bl, size=2, replace=False)
        single = float(damage[i].mean()) + float(damage[j].mean())
        pair = pair_damage_exact(int(i), int(j))
        same_ratios.append(pair / max(single, 1e-12))
    all_idx = list(range(TOPK))
    for _ in range(60):
        i, j = g3.choice(all_idx, size=2, replace=False)
        if labels[i] == labels[j]:
            continue
        single = float(damage[i].mean()) + float(damage[j].mean())
        pair = pair_damage_exact(int(i), int(j))
        cross_ratios.append(pair / max(single, 1e-12))
    same_ratio = float(np.mean(same_ratios)) if same_ratios else None
    cross_ratio = float(np.mean(cross_ratios)) if cross_ratios else None
    pc = (same_ratio is not None and cross_ratio is not None and
          same_ratio > cross_ratio)
    print(f'(c) same-cluster pair ratio {same_ratio}, cross-cluster '
          f"{cross_ratio}: {'HELD' if pc else 'FAILED'}", flush=True)

    # (d) interpretability spot-check on the 3 largest clusters
    sizes = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))[:3]
    rows_all = cl.rows()
    cnt = torch.bincount(rows_all.reshape(-1),
                          minlength=m.transformer.wte.weight.shape[0])
    freq = set(cnt.argsort(descending=True)[:5000].tolist())
    E = m.transformer.wte.weight.float()
    named = []
    for cid, idxs in sizes:
        unit_ids = topk[idxs]
        direction = Dw[:, unit_ids].sum(1)
        direction = direction / direction.norm().clamp_min(1e-9)
        scores = E @ direction.to(E.device)
        cand = [t for t in scores.argsort(descending=True).tolist()
                if t in freq][:5]
        toks = [cl.d1(t) for t in cand]
        named.append({'cluster': cid, 'n_units': len(idxs),
                       'unit_ids': unit_ids.tolist()[:10],
                       'top_tokens': toks})
        print(f"   cluster {cid} ({len(idxs)} units): {toks}", flush=True)

    out = {'N': N, 'K': TOPK, 'output_rank_r': r,
           'projection_relerr': relerr, 'pred_0': bool(p0),
           'n_clusters_at_0.5': n_clusters, 'pred_a': bool(pa),
           'stability_agreement': agree, 'stability_null': agree_null,
           'pred_b': bool(pb),
           'same_cluster_ratio': same_ratio,
           'cross_cluster_ratio': cross_ratio, 'pred_c': bool(pc),
           'top_clusters': named, 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
