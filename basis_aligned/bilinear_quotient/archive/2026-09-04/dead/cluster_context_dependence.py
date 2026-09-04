"""CLUSTER CONTEXT DEPENDENCE -- which of mlp0's named clusters read
the CURRENT TOKEN, and which need the PREVIOUS-token CONTEXT that attn0
folds in? Maps the input side of six circuits with one validated
method, avoiding the reversal ambiguity entirely (this is about what
FEEDS each cluster, not which way it pushes).

597 established the method on the article cluster: mlp0 sits right
after attn0, so zeroing attn0's write and re-measuring a cluster's
firing reveals whether it is driven by the current token's embedding
alone (firing unchanged) or the previous-token context attn0 carries
(firing changes). The article cluster came out strongly context-driven
(correlation 0.24). Control: zeroing attn1 (a later block) leaves any
mlp0 cluster's firing exactly unchanged.

This applies it as a BATCH across the six clean nameable clusters found
so far (600/601): article, punctuation, aux-contraction (top-300);
newline, pronoun, number-word (ranks 300-600). The result is a map of
which early decisions are lexical (current-token) vs contextual
(previous-token), a clean model-level statement about how mlp0
allocates its work.

REGISTERED PREDICTIONS:
  (0) CONTROL + REPRODUCIBILITY: both bands reproduce their sizes;
      zeroing attn1 leaves every cluster's firing correlation > 0.999
      (only block-0-and-earlier can reach mlp0's input) -- VOIDS on
      failure;
  (a) ARTICLE RE-CONFIRMS CONTEXT-DRIVEN: the article cluster's attn0-
      ablated correlation is < 0.5 (597 got 0.24), re-validating the
      method on the known case;
  (b) A SPLIT EXISTS (the finding): report each cluster's attn0-
      ablated firing correlation. Prediction: at least one cluster is
      clearly CURRENT-TOKEN-driven (correlation > 0.8 -- the number-
      word cluster is the natural candidate, since a spelled-out
      number is identifiable from the current token) and at least one
      besides the article is CONTEXT-driven (< 0.5). A split across
      the six is the result; a map, not a pass/fail;
  NULL: the attn1 control (b's within-run null) -- every cluster's
      attn1-ablated correlation stays > 0.999, so any low attn0
      correlation is genuine upstream dependence, not measurement
      noise."""
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
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cluster_context_dependence_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20


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
    return torch.cat(hs, dim=0), Dw.cpu()


def recover_band(fresh, band, expect):
    H, Dw = capture(fresh)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(H.shape[0], generator=g)[:NSAMP]
    Hs = H[perm]
    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    topk = order[band[0]:band[1]].numpy()
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
    return ranked, sizes == expect


@torch.no_grad()
def cluster_signed(fresh, units, ablate=None):
    mlp = m.transformer.h[0].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    cap = {}
    hk = mlp.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X', a_[0]))
    abl = []
    if ablate is not None:
        li = int(ablate[-1])
        cp = m.transformer.h[li].attn.c_proj
        abl.append(cp.register_forward_hook(
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
        out.append(h[:, :, units].sum(-1).reshape(-1).cpu())
    hk.remove()
    for a in abl:
        a.remove()
    return torch.cat(out)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    exp_top = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    exp_mid = [81, 54, 37, 31, 22, 12, 10, 8, 7, 6, 6, 5, 4, 4, 4, 3, 3, 1, 1, 1]
    rtop, ok_t = recover_band(fresh, (0, TOPK), exp_top)
    rmid, ok_m = recover_band(fresh, (TOPK, 2 * TOPK), exp_mid)
    p0 = ok_t and ok_m
    print(f'(0) top-300 {ok_t}, 300-600 {ok_m}: '
          f'{"HELD" if p0 else "FAILED -- VOID"}', flush=True)
    if not p0:
        json.dump({'void': 'reclustering mismatch'}, open(OUT, 'w'), indent=1)
        return

    clusters = {
        'article': rtop[0][1],       # 101
        'punctuation': rtop[1][1],   # 76
        'aux_contraction': rtop[2][1],  # 29
        'newline': rmid[0][1],       # 81
        'pronoun': rmid[1][1],       # 54
        'number_word': rmid[2][1],   # 37
    }
    for k, u in clusters.items():
        print(f'  {k}: n={len(u)}', flush=True)

    results = {}
    for name, units in clusters.items():
        base = cluster_signed(fresh, units, ablate=None).numpy()
        abl0 = cluster_signed(fresh, units, ablate='attn0').numpy()
        abl1 = cluster_signed(fresh, units, ablate='attn1').numpy()
        # measure at the cluster's own strong-firing positions
        thresh = np.quantile(np.abs(base), 0.9)
        mask = np.abs(base) >= thresh
        c0 = pearsonr(base[mask], abl0[mask])[0]
        c1 = pearsonr(base[mask], abl1[mask])[0]
        driven = ('current-token' if c0 > 0.8
                  else 'context (attn0)' if c0 < 0.5 else 'mixed')
        results[name] = {'attn0_corr': float(c0), 'attn1_corr': float(c1),
                         'driven_by': driven, 'n_units': len(units)}
        print(f'  {name:>16}: attn0-corr {c0:+.3f} attn1-corr {c1:+.3f} '
              f'-> {driven}', flush=True)

    control_ok = all(r['attn1_corr'] > 0.999 for r in results.values())
    pa = results['article']['attn0_corr'] < 0.5
    ctx = [k for k, r in results.items() if r['attn0_corr'] < 0.5]
    tok = [k for k, r in results.items() if r['attn0_corr'] > 0.8]
    pb = len(tok) >= 1 and len(ctx) >= 2
    print(f'\n(0-control) all attn1-corr > 0.999: '
          f'{"HELD" if control_ok else "FAILED"}', flush=True)
    print(f'(a) article context-driven (attn0-corr < 0.5): '
          f'{"HELD" if pa else "FAILED"}', flush=True)
    print(f'(b) split -- context-driven {ctx}, current-token {tok}: '
          f'{"HELD" if pb else "FAILED"}', flush=True)

    out = {'pred_0': bool(p0), 'control_ok': bool(control_ok),
           'clusters': results, 'context_driven': ctx, 'token_driven': tok,
           'pred_a': bool(pa), 'pred_b': bool(pb),
           'null_ok': bool(control_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
