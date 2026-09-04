"""RSPD TOY VALIDATE -- sanity-check the ablation-covariance-clustering
method on synthetic data with KNOWN computational groups, before
applying it to a real early-layer component (MLP0 decoder / attn QK
or OV -- "the earliest component we don't quite understand").

User's proposed method: SVD(W X) on a component's output over a lot
of data (local reconstruction as a fast proxy for the real nonlinear
computation), then random ablations of singleton/pair/group
components, measure reconstruction-loss damage per datapoint, check
covariance of damage across components, and hierarchically cluster
to recover "variable names" -- reusable computational sub-blocks.

Could not fetch github.com/ThatE10/rspd (404 / blocked credential
use against a third-party repo) -- this is my own implementation of
the method as described, not their code. This script's only job is
to check that implementation is sound on a problem with ground
truth, per the request: "make your own dataset of things that use
separate computational components ... be sure your understanding of
the tool can separate them."

TOY CONSTRUCTION: three "computational components" A, B, C write to
disjoint blocks of a shared output space via a block-structured
weight matrix W (rows = output dims, cols = latent dims; each latent
only feeds its own block, mixed by a random rotation WITHIN the
block, so no single output coordinate is a clean giveaway -- SVD/
clustering must recover the grouping from statistics, not from
reading off dimensions). Each component is independently "active"
(Bernoulli gate) on ~half of samples and near-zero on the rest,
mimicking real circuits that fire on a subset of data. Component A
is further split into two sub-latents A1/A2 that share A's gate
(always co-active with each other) but are otherwise independent --
a hierarchical case: A1/A2 should merge into A before either merges
with B or C.

REGISTERED PREDICTIONS:
  (0) IDENTITY: full-rank reconstruction (all components kept) has
      near-zero loss (< 1e-6) -- the SVD basis is exact;
  (a) TOP-LEVEL RECOVERY: hierarchical clustering of components by
      damage-covariance, cut to 3 clusters, exactly recovers the
      {A1+A2}, {B}, {C} partition (adjusted Rand index == 1.0);
  (b) HIERARCHY: in the dendrogram, A1's and A2's components merge
      with each other before either merges with any B or C
      component -- the nested structure is recovered, not just the
      flat partition;
  (c) MINIMAL WEIGHTS: reconstructing block A's output using ONLY
      the components clustered into A (zeroing all B/C components)
      loses < 5% relative reconstruction fidelity on block A
      specifically, vs >= 50% loss if instead using only B+C's
      components -- the cluster found is the minimal sufficient set;
  (d) PAIR SUPERADDITIVITY: ablating a random PAIR of components
      from the SAME cluster produces reconstruction damage more than
      the sum of their two singleton damages (they explain
      overlapping variance); a cross-cluster pair's damage is close
      to additive (independent variance) -- report both ratios;
  NULL: shuffling which sample each component's damage vector
      belongs to (breaking the datapoint alignment) destroys the
      clustering (ARI drops to ~0) -- the recovery depends on the
      real per-datapoint co-activation pattern, not on marginal
      damage magnitude alone."""
import json, time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_toy_validate_results.json'
N = 4000
rng = np.random.default_rng(0)


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


def make_block(d_latent, d_out, n, gate_p, rng):
    """returns (latent (n,d_latent), gate (n,) bool, W (d_out,d_latent))"""
    latent = rng.normal(size=(n, d_latent))
    gate = rng.random(n) < gate_p
    latent = latent * gate[:, None]
    W = rng.normal(size=(d_out, d_latent)) / np.sqrt(d_latent)
    return latent, gate, W


def main():
    t0 = time.time()
    # block A: two sub-latents (A1 dims 2, A2 dims 2) sharing one gate,
    # each independently mixed into its OWN 3-dim sub-block (no cross
    # mixing between A1/A2's output dims -- but both still "block A").
    gA = rng.random(N) < 0.5
    lA1 = rng.normal(size=(N, 2)) * gA[:, None]
    lA2 = rng.normal(size=(N, 2)) * gA[:, None]
    WA1 = rng.normal(size=(3, 2)) / np.sqrt(2)
    WA2 = rng.normal(size=(3, 2)) / np.sqrt(2)
    outA1 = lA1 @ WA1.T
    outA2 = lA2 @ WA2.T

    lB, gB, WB = make_block(3, 5, N, 0.5, rng)
    outB = lB @ WB.T

    lC, gC, WC = make_block(1, 4, N, 0.5, rng)
    outC = lC @ WC.T

    NOISE = 0.03
    WX = np.concatenate([outA1, outA2, outB, outC], axis=1)
    WX += rng.normal(scale=NOISE, size=WX.shape)
    D = WX.shape[1]  # 3+3+5+4 = 15
    dims = {'A1': slice(0, 3), 'A2': slice(3, 6), 'B': slice(6, 11),
            'C': slice(11, 15)}
    true_block = (['A1'] * 3 + ['A2'] * 3 + ['B'] * 5 + ['C'] * 4)
    true_top = (['A'] * 6 + ['B'] * 5 + ['C'] * 4)

    # ---- SVD(WX) ----
    mu = WX.mean(0)
    Xc = WX - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # signal rank via scree elbow (largest S[i]/S[i+1] ratio); components
    # beyond the elbow are noise floor -- kept fixed in reconstruction,
    # never ablated/clustered (a degenerate near-equal-eigenvalue
    # subspace has no canonical basis, so "components" there are not
    # meaningful units to ablate).
    ratios = S[:-1] / np.maximum(S[1:], 1e-9)
    k = int(np.argmax(ratios)) + 1
    scores = U * S  # (N, D) -- per-sample component activations
    V = Vt  # (D, D) -- component i's direction is V[i]
    # ground-truth block label for each retained component, from which
    # block's output dims carry most of that component's energy --
    # used only for SCORING the clustering, not by the algorithm.
    comp_true_block = []
    for i in range(k):
        e = {b: float((V[i, sl] ** 2).sum()) for b, sl in dims.items()}
        comp_true_block.append(max(e, key=e.get))
    comp_true_top = ['A' if b in ('A1', 'A2') else b
                      for b in comp_true_block]

    def reconstruct(comp_mask):
        """comp_mask: bool (k,) which of the top-k components are kept;
        components beyond k (noise floor) are always kept."""
        s = scores.copy()
        drop = np.zeros(D, dtype=bool)
        drop[:k] = ~comp_mask
        s[:, drop] = 0.0
        return mu + s @ V

    # (0) identity
    full = reconstruct(np.ones(k, dtype=bool))
    ident_err = float(np.mean((full - WX) ** 2))

    # per-sample, per-block loss under full reconstruction (should be ~0)
    def block_loss(recon, blk):
        sl = dims[blk]
        return np.mean((recon[:, sl] - WX[:, sl]) ** 2, axis=1)

    # ---- singleton damage vectors ----
    base_loss = np.mean((full - WX) ** 2, axis=1)  # ~0
    damage = np.zeros((k, N))
    for i in range(k):
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        r = reconstruct(mask)
        damage[i] = np.mean((r - WX) ** 2, axis=1) - base_loss

    # ---- covariance/correlation of damage across components ----
    dm = damage - damage.mean(1, keepdims=True)
    dstd = damage.std(1, keepdims=True)
    dstd[dstd < 1e-12] = 1e-12
    corr = (dm @ dm.T) / (N * dstd * dstd.T)
    corr = np.clip(corr, -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    condensed = squareform((dist + dist.T) / 2, checks=False)
    Z = linkage(condensed, method='average')

    # (a) top-level recovery: cut to 3 clusters
    labels3 = fcluster(Z, t=3, criterion='maxclust')
    true_top_id = {b: i for i, b in enumerate(sorted(set(comp_true_top)))}
    true_top_lab = [true_top_id[b] for b in comp_true_top]
    ari = adjusted_rand_index(true_top_lab, list(labels3))

    # (b) hierarchy: do A1 and A2 components merge with each other
    # before either merges with a B or C component?
    # walk the linkage tree: find the merge height where the full A1
    # set and full A2 set first become connected, vs the height where
    # A (A1 u A2) first connects to any B/C component.
    n_leaves = k
    parent = {}  # cluster id -> set of leaf indices
    for i in range(n_leaves):
        parent[i] = {i}
    a1_idx = {i for i in range(k) if comp_true_block[i] == 'A1'}
    a2_idx = {i for i in range(k) if comp_true_block[i] == 'A2'}
    bc_idx = {i for i in range(k) if comp_true_block[i] in ('B', 'C')}
    merge_a1a2 = None
    merge_a_bc = None
    for row, (c1, c2, h, cnt) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        s1 = parent[c1]
        s2 = parent[c2]
        news = s1 | s2
        cid = n_leaves + row
        parent[cid] = news
        if merge_a1a2 is None and s1 and s2:
            if (s1 <= a1_idx and s2 <= a2_idx) or \
               (s1 <= a2_idx and s2 <= a1_idx):
                merge_a1a2 = h
        if merge_a_bc is None:
            a_full = a1_idx | a2_idx
            if news & a_full and news & bc_idx and \
               (s1 <= a_full or s1 <= bc_idx) and \
               (s2 <= a_full or s2 <= bc_idx) and \
               not (s1 <= a_full and s2 <= a_full) and \
               not (s1 <= bc_idx and s2 <= bc_idx):
                merge_a_bc = h
    hierarchy_ok = (merge_a1a2 is not None and merge_a_bc is not None
                     and merge_a1a2 < merge_a_bc)

    # (c) minimal weights: reconstruct block A using only its own
    # cluster's components vs using only B+C's components
    cluster_of = {i: int(labels3[i]) for i in range(k)}
    # which flat cluster id corresponds to "A" = majority vote of A1/A2 idx
    from collections import Counter
    a_true_idx = sorted(a1_idx | a2_idx)
    a_cluster = Counter(labels3[i] for i in a_true_idx).most_common(1)[0][0]
    own_mask = np.array([cluster_of[i] == a_cluster for i in range(k)])
    other_mask = ~own_mask
    r_own = reconstruct(own_mask)
    r_other = reconstruct(other_mask)
    lossA_own = float(np.mean(block_loss(r_own, 'A1')) +
                       np.mean(block_loss(r_own, 'A2')))
    lossA_other = float(np.mean(block_loss(r_other, 'A1')) +
                         np.mean(block_loss(r_other, 'A2')))
    varA = float(np.var(np.concatenate([WX[:, dims['A1']],
                                         WX[:, dims['A2']]], axis=1)))
    rel_own = lossA_own / max(varA, 1e-9)
    rel_other = lossA_other / max(varA, 1e-9)

    # (d) pair superadditivity: same-cluster vs cross-cluster pairs
    def pair_damage(i, j):
        mask = np.ones(k, dtype=bool)
        mask[i] = False; mask[j] = False
        r = reconstruct(mask)
        return float(np.mean((r - WX) ** 2))
    rng2 = np.random.default_rng(1)
    same_ratios, cross_ratios = [], []
    idx_by_block = {b: [i for i in range(k) if comp_true_block[i] == b]
                     for b in ('A1', 'A2', 'B', 'C')}
    blocks_l = [bl for bl in idx_by_block.values() if len(bl) >= 2]
    for _ in range(40):
        bl = blocks_l[rng2.integers(len(blocks_l))]
        if len(bl) >= 2:
            i, j = rng2.choice(bl, size=2, replace=False)
            single = float(damage[i].mean()) + float(damage[j].mean())
            pair = pair_damage(int(i), int(j))
            same_ratios.append(pair / max(single, 1e-9))
    all_blocks_l = list(idx_by_block.values())
    for _ in range(40):
        b1, b2 = rng2.choice(4, size=2, replace=False)
        i = rng2.choice(all_blocks_l[b1]); j = rng2.choice(all_blocks_l[b2])
        single = float(damage[i].mean()) + float(damage[j].mean())
        pair = pair_damage(int(i), int(j))
        cross_ratios.append(pair / max(single, 1e-9))
    same_ratio = float(np.mean(same_ratios))
    cross_ratio = float(np.mean(cross_ratios))

    # NULL: shuffle sample alignment of each component's damage vector
    dshuf = damage.copy()
    for i in range(k):
        dshuf[i] = rng.permutation(dshuf[i])
    dsm = dshuf - dshuf.mean(1, keepdims=True)
    dsstd = dshuf.std(1, keepdims=True); dsstd[dsstd < 1e-12] = 1e-12
    corr_s = (dsm @ dsm.T) / (N * dsstd * dsstd.T)
    corr_s = np.clip(corr_s, -1, 1)
    dist_s = 1 - corr_s
    np.fill_diagonal(dist_s, 0)
    cond_s = squareform((dist_s + dist_s.T) / 2, checks=False)
    Zs = linkage(cond_s, method='average')
    labels3_s = fcluster(Zs, t=3, criterion='maxclust')
    ari_null = adjusted_rand_index(true_top_lab, list(labels3_s))

    p0 = ident_err < 1e-6
    pa = ari >= 0.999
    pb = hierarchy_ok
    pc = rel_own < 0.05 and rel_other >= 0.50
    pd_report = {'same_cluster_ratio': round(same_ratio, 3),
                 'cross_cluster_ratio': round(cross_ratio, 3)}
    null_ok = ari_null < 0.3

    print(f"(0) identity reconstruction error {ident_err:.2e} < 1e-6: "
          f"{'HELD' if p0 else 'FAILED'}")
    print(f"(a) 3-cluster ARI vs {{A1+A2,B,C}}: {ari:.3f} "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) hierarchy A1/A2 merge ({merge_a1a2}) before A-BC merge "
          f"({merge_a_bc}): {'HELD' if pb else 'FAILED'}")
    print(f"(c) minimal weights: block-A loss using own cluster "
          f"{rel_own:.4f} rel-var, using other cluster {rel_other:.4f} "
          f"rel-var: {'HELD' if pc else 'FAILED'}")
    print(f"(d) pair damage/singleton-sum ratio: same-cluster "
          f"{same_ratio:.2f}x, cross-cluster {cross_ratio:.2f}x")
    print(f"NULL (shuffled-sample ARI {ari_null:.3f} < 0.3): "
          f"{'ok' if null_ok else 'CHECK'}")

    out = {'N': N, 'D': D, 'identity_err': ident_err, 'ari_3cluster': ari,
           'merge_height_a1a2': merge_a1a2, 'merge_height_a_bc': merge_a_bc,
           'hierarchy_ok': bool(hierarchy_ok),
           'lossA_own_cluster_rel': rel_own,
           'lossA_other_cluster_rel': rel_other,
           'pair_damage_ratios': pd_report,
           'ari_null_shuffled': ari_null,
           'pred_0': bool(p0), 'pred_a': bool(pa), 'pred_b': bool(pb),
           'pred_c': bool(pc), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.1f}s)')


if __name__ == '__main__':
    main()
