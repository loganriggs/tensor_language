"""ARTICLE CIRCUIT DEPTH -- depth-first on the one fully-confirmed
circuit (mlp0 cluster 8, the a/an-vs-the article circuit), doing the
two things the user asked for: (1) FOLD ATTN0 INTO MLP0 to see what
actually feeds the article decision, and (2) trace it FORWARD to test
whether its downstream echo (mlp1's article cluster, 595) fires on
the SAME data points.

Context: the program has been breadth-first (mlp0->mlp1->mlp2 unit
clustering). Coverage is broad but shallow -- only the top-300 units
per layer, only the 3 largest clusters named, only cluster 8 fully
causally traced. This is the complementary depth-first pass on the
best circuit, per the user's steer to intersperse the two.

PHASE 1 -- BACKWARD (fold attn0 into mlp0). mlp0 sits in block 0,
immediately after attn0, so mlp0's input is exactly the embedding
path PLUS attn0's contribution (attn0 is an exact bigram table, its
input exactly the token embedding -- 254). By zeroing attn0's write
(its c_proj output) and re-measuring cluster 8's firing, we learn
whether the article decision is made from the CURRENT TOKEN alone
(cluster 8 unchanged) or needs the PREVIOUS-TOKEN context attn0
carries (cluster 8 changes). Control: zeroing attn1 (a LATER block,
which cannot affect mlp0's input at all) must leave cluster 8
exactly unchanged -- a clean proof the measurement only picks up
genuine upstream dependence.

PHASE 2 -- FORWARD (same data points). If mlp1's article cluster
(595, causally confirmed as a 13% echo of cluster 8) is genuinely the
same circuit continued, it should FIRE ON THE SAME POSITIONS. Measure
per-position firing energy (sum of squared unit activations) of
mlp0-cluster8 and mlp1-article-cluster across all real positions, and
correlate. NULL: an UNRELATED mlp1 cluster (the 86-unit tokenization-
artifact "replacement-char" detector, 587) should NOT co-fire with
cluster 8's positions.

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY: both mlp0 and mlp1 reclusterings reproduce
      their established sizes exactly -- VOIDS on failure;
  (1a) BACKWARD CONTROL (clean null): zeroing attn1 leaves cluster 8
      activation identical (correlation > 0.999) -- proves only
      block-0-and-earlier can feed mlp0's input;
  (1b) BACKWARD FINDING: zeroing attn0 changes cluster 8 activation
      at article-target positions -- report the full-vs-ablated
      correlation. Prediction: attn0 IS load-bearing (correlation
      < 0.9), because article choice depends on preceding context,
      not just the current token. Report the number regardless;
  (2a) FORWARD FINDING: mlp0-cluster8 and mlp1-article-cluster
      per-position firing energies correlate positively (Pearson
      > 0.2) across all positions -- they fire on similar data;
  (2b) FORWARD NULL: mlp0-cluster8 vs the unrelated mlp1 artifact
      cluster correlates much less (< half of 2a's correlation) --
      the co-firing is specific to the redundant article cluster,
      not any large mlp1 cluster;
  (2c) TOP-POSITION OVERLAP (no bar, report): Jaccard overlap of the
      top-100 firing positions, cluster8 vs mlp1-article vs
      mlp1-artifact."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_circuit_depth_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383


@torch.no_grad()
def capture_h(fresh, LJ):
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
    return torch.cat(hs, dim=0), Dw.cpu()


def recover_clusters(fresh, LJ):
    H, Dw = capture_h(fresh, LJ)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(H.shape[0], generator=g)[:NSAMP]
    Hs = H[perm]
    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    topk = order[:TOPK].numpy()
    O = Hs @ Dw.T
    Oc = O - O.mean(0)
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    r = min(int((cum < 0.95).sum().item()) + 1, Vt.shape[0])
    Vr = Vt[:r]
    Dw_topk = Dw[:, topk]
    Dw_proj = Vr @ Dw_topk
    Hk = Hs[:, topk]
    coldw2 = (Dw_proj ** 2).sum(0)
    damage = ((Hk ** 2) * coldw2[None, :]).T.numpy()
    dm = damage - damage.mean(1, keepdims=True)
    dstd = damage.std(1, keepdims=True)
    dstd[dstd < 1e-12] = 1e-12
    corr = np.clip((dm @ dm.T) / (damage.shape[1] * dstd * dstd.T), -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform((dist + dist.T) / 2, checks=False), method='average')
    labels = fcluster(Z, t=NCLUST, criterion='maxclust')
    sizes = sorted(np.bincount(labels)[1:].tolist(), reverse=True)
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    ranked = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
    return ranked, sizes


@torch.no_grad()
def cluster8_activation(fresh, cluster8, ablate=None):
    """Per-position summed signed activation of mlp0 cluster8 units.
    ablate: None | 'attn0' | 'attn1' -- zero that attn block's c_proj."""
    mlp = m.transformer.h[0].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    cap = {}
    hk = mlp.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X', a_[0]))
    abl_hooks = []
    if ablate is not None:
        li = int(ablate[-1])
        cp = m.transformer.h[li].attn.c_proj
        abl_hooks.append(cp.register_forward_hook(
            lambda mo_, a_, o_: torch.zeros_like(o_)))
    out = []
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
        out.append(h[:, :, cluster8].sum(-1).reshape(-1).cpu())
    hk.remove()
    for hh in abl_hooks:
        hh.remove()
    return torch.cat(out)


@torch.no_grad()
def cluster_energy(fresh, LJ, units):
    """Per-position firing energy (sum of squared unit activations)."""
    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    cap = {}
    hk = mlp.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X', a_[0]))
    out = []
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
        out.append((h[:, :, units] ** 2).sum(-1).reshape(-1).cpu())
    hk.remove()
    return torch.cat(out)


def jaccard_top(a, b, k=100):
    ta = set(np.argsort(a.numpy())[::-1][:k].tolist())
    tb = set(np.argsort(b.numpy())[::-1][:k].tolist())
    return len(ta & tb) / len(ta | tb)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)

    r0, s0 = recover_clusters(fresh, 0)
    r1, s1 = recover_clusters(fresh, 1)
    exp0 = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    exp1 = [86, 46, 42, 37, 12, 11, 11, 11, 10, 9, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1]
    p0 = (s0 == exp0 and s1 == exp1)
    print(f'(0) mlp0 sizes match {s0 == exp0}, mlp1 sizes match {s1 == exp1}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'reclustering mismatch', 's0': s0, 's1': s1},
                   open(OUT, 'w'), indent=1)
        return
    cluster8 = r0[0][1]          # mlp0 largest = article cluster (101)
    mlp1_article = r1[1][1]      # mlp1 2nd-largest = determiner cluster (46)
    mlp1_artifact = r1[0][1]     # mlp1 largest = tokenization artifact (86)
    print(f'cluster8 n={len(cluster8)}, mlp1_article n={len(mlp1_article)}, '
          f'mlp1_artifact n={len(mlp1_artifact)}', flush=True)

    # article-target positions
    nxt = fresh[:, 1:257].reshape(-1)
    art_mask = ((nxt == TOK_A) | (nxt == TOK_AN) |
                (nxt == TOK_THE) | (nxt == TOK_THE2)).numpy()
    print(f'{art_mask.sum()} article-target positions', flush=True)

    # PHASE 1 backward
    base = cluster8_activation(fresh, cluster8, ablate=None)
    abl0 = cluster8_activation(fresh, cluster8, ablate='attn0')
    abl1 = cluster8_activation(fresh, cluster8, ablate='attn1')
    am = art_mask
    c_attn1 = pearsonr(base[am].numpy(), abl1[am].numpy())[0]
    c_attn0 = pearsonr(base[am].numpy(), abl0[am].numpy())[0]
    p1a = c_attn1 > 0.999
    p1b_context = c_attn0 < 0.9
    print(f'(1a) attn1-ablated correlation {c_attn1:.4f} (control, want '
          f">0.999): {'HELD' if p1a else 'FAILED'}", flush=True)
    print(f'(1b) attn0-ablated correlation {c_attn0:.4f}: '
          f"{'attn0 load-bearing (context-driven)' if p1b_context else 'attn0 NOT load-bearing (current-token-driven)'}",
          flush=True)

    # PHASE 2 forward
    e8 = cluster_energy(fresh, 0, cluster8)
    e1a = cluster_energy(fresh, 1, mlp1_article)
    e1x = cluster_energy(fresh, 1, mlp1_artifact)
    r_article = pearsonr(e8.numpy(), e1a.numpy())[0]
    r_artifact = pearsonr(e8.numpy(), e1x.numpy())[0]
    p2a = r_article > 0.2
    p2b = abs(r_artifact) < 0.5 * abs(r_article)
    print(f'(2a) mlp0-c8 vs mlp1-article energy correlation {r_article:.4f} '
          f"(bar >0.2): {'HELD' if p2a else 'FAILED'}", flush=True)
    print(f'(2b) mlp0-c8 vs mlp1-artifact correlation {r_artifact:.4f} '
          f"(null, want <half): {'HELD' if p2b else 'FAILED'}", flush=True)
    j_article = jaccard_top(e8, e1a)
    j_artifact = jaccard_top(e8, e1x)
    print(f'(2c) top-100 Jaccard: article {j_article:.3f}, artifact '
          f'{j_artifact:.3f}', flush=True)

    out = {'pred_0': bool(p0), 'n_article_positions': int(am.sum()),
           'backward': {'attn1_control_corr': float(c_attn1),
                        'attn0_corr': float(c_attn0),
                        'pred_1a_control': bool(p1a),
                        'attn0_context_driven': bool(p1b_context)},
           'forward': {'corr_article': float(r_article),
                       'corr_artifact': float(r_artifact),
                       'pred_2a': bool(p2a), 'pred_2b': bool(p2b),
                       'jaccard_article': float(j_article),
                       'jaccard_artifact': float(j_artifact)},
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
