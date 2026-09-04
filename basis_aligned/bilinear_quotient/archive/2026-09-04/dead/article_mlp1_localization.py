"""ARTICLE MLP1 LOCALIZATION -- 608 revealed mlp1 as the single
heaviest article-margin layer (+0.120 whole-layer, 8x mlp0). Is that
article role CONCENTRATED in mlp1's article echo cluster (46 units,
595), the way mlp0's is in cluster 8, or DIFFUSE across the whole
layer?

Method: at real article positions, mean-fill (remove the write of)
(a) the WHOLE mlp1 layer, (b) just mlp1's 46-unit article cluster,
(c) a size-matched random 46-unit set, and measure the article-margin
shift for each. The fraction (b)/(a) is the localization: near 1 =
the cluster carries mlp1's whole article role (concentrated); near 0 =
mlp1's article work is spread across the layer (diffuse).

601 already showed the article decision is DISTRIBUTED across multiple
mlp0 clusters at different importance ranks, so the prediction is
diffuse -- but the number is the finding.

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY + IDENTITY: mlp1 reclustering reproduces its
      sizes and the article cluster is the 2nd largest (46 units); no
      ablation reproduces the true margin -- VOIDS on failure;
  (a) WHOLE-LAYER SANITY: mean-filling all of mlp1 reproduces 608's
      +0.120 shift (within 0.02) -- confirms the same measurement;
  (b) LOCALIZATION FRACTION (the finding, no pass/fail): the 46-unit
      article cluster's shift as a fraction of the whole-mlp1 shift.
      Prediction: DIFFUSE (< 0.4) -- the cluster is a minority of
      mlp1's article work, consistent with 601's distributed picture;
  (c) SPECIFICITY: the article cluster's |shift| exceeds a random
      46-unit set's |shift| by >= 3x -- the cluster does carry more
      article work than an arbitrary same-size slice, even if it's a
      minority of the whole;
  NULL: the random 46-unit set's |shift| is small relative to the
      whole layer (< 0.2 of it) -- an arbitrary slice is not itself a
      big article contributor."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
LJ = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_mlp1_localization_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383


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


def recover_article_cluster(fresh):
    H, Dw = capture(fresh)
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
    expect = [86, 46, 42, 37, 12, 11, 11, 11, 10, 9, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1]
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    ranked = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
    return ranked[1][1], sizes == expect  # 2nd largest = article echo (46)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    article, repro = recover_article_cluster(fresh)
    print(f'(0a) reproduced + article cluster n={len(article)}: '
          f'{"HELD" if repro and len(article) == 46 else "FAILED"}', flush=True)
    if not (repro and len(article) == 46):
        json.dump({'void': 'reclustering mismatch', 'n': len(article)},
                   open(OUT, 'w'), indent=1)
        return

    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight
    b = mlp.Down_bias
    H_full = Dw.shape[1]

    Hall, _ = capture(fresh)
    hmean = Hall.mean(0).to(DEV)
    out_mean = (hmean @ Dw.T.float() + b.float())  # mean output vector

    nxt = fresh[:, 1:257].reshape(-1)
    art = ((nxt == TOK_A) | (nxt == TOK_AN) |
           (nxt == TOK_THE) | (nxt == TOK_THE2)).numpy()
    amask = torch.tensor(art)

    @torch.no_grad()
    def margin(mode=None, units=None):
        """mode: None | 'whole' | 'units'."""
        hook = None
        if mode == 'whole':
            def fh(mo, args, o_):
                return out_mean.expand_as(o_).to(o_.dtype)
            hook = mlp.register_forward_hook(fh)
        elif mode == 'units':
            msk = torch.zeros(H_full, device=DEV)
            msk[torch.tensor(units, device=DEV)] = 1.0

            def fh(mo, args, o_):
                X = args[0].float()
                h = (X @ L.T) * (X @ R.T)
                h = h * (1 - msk) + hmean[None, None, :] * msk
                return (h @ Dw.T.float() + b.float()).to(o_.dtype)
            hook = mlp.register_forward_hook(fh)
        out = []
        for i in range(0, fresh.shape[0], 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,))
            x0 = x
            v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            p = F.softmax(lg, dim=-1)
            mg = p[..., TOK_A] + p[..., TOK_AN] - p[..., TOK_THE] - p[..., TOK_THE2]
            out.append(mg.reshape(-1).cpu())
        if hook is not None:
            hook.remove()
        return torch.cat(out)

    base = margin(None)
    base_art = float(base[amask].mean())
    ident = float(margin(None)[amask].mean()) - base_art
    p0 = abs(ident) < 1e-9
    print(f'(0b) identity {ident:+.7f}: {"HELD" if p0 else "FAILED"} | '
          f'baseline margin {base_art:.4f}', flush=True)

    whole = float(margin('whole')[amask].mean()) - base_art
    clust = float(margin('units', article)[amask].mean()) - base_art
    rng = np.random.default_rng(21)
    rand_units = rng.choice(list(range(H_full)), size=len(article),
                            replace=False).tolist()
    rand = float(margin('units', rand_units)[amask].mean()) - base_art

    pa = abs(whole - 0.120) < 0.02
    frac = clust / whole if whole != 0 else None
    localized = 'concentrated' if (frac is not None and abs(frac) >= 0.4) else 'diffuse'
    pc = abs(clust) >= 3 * abs(rand) if rand != 0 else abs(clust) > abs(rand)
    null_ok = abs(rand) < 0.2 * abs(whole)
    print(f'(a) whole-mlp1 shift {whole:+.4f} (608 was +0.120): '
          f'{"HELD" if pa else "FAILED"}', flush=True)
    print(f'(b) article cluster shift {clust:+.4f}, fraction of whole '
          f'{frac:.2f} -> {localized}', flush=True)
    print(f'(c) random 46-unit shift {rand:+.4f} (cluster/random '
          f'{abs(clust)/max(abs(rand),1e-9):.2f}x): '
          f'{"HELD" if pc else "FAILED"}', flush=True)
    print(f'NULL (random {abs(rand):.4f} < 0.2*whole {0.2*abs(whole):.4f}): '
          f'{"ok" if null_ok else "CHECK"}', flush=True)

    out = {'repro': bool(repro), 'n_article': len(article),
           'baseline_margin': base_art, 'pred_0': bool(p0),
           'whole_mlp1_shift': whole, 'article_cluster_shift': clust,
           'random_shift': rand, 'localization_fraction': frac,
           'localization': localized, 'pred_a': bool(pa), 'pred_c': bool(pc),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
