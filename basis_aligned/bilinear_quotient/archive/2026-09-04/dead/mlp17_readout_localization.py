"""MLP17 READOUT LOCALIZATION -- are READOUT circuits localizable even
though FEATURE-BUILDING circuits are not? A pointed contrast to 610.

610-613 established that mlp0's feature-building decisions are causally
DIFFUSE over units (clusters carry ~0-4% of the whole-layer effect),
compact only in a low-rank output subspace. But 615 found mlp17 (the
readout end) is a DIFFERENT kind of layer -- its output is rank-8, and
it clusters cleanly into direct next-token readouts. Does that make
its clusters NECESSARY (localizable) rather than merely sufficient?
This runs the identical ablation-fraction test on mlp17's newline
readout cluster (88 units, the largest top-300 cluster, 615): mean-
fill (a) the whole mlp17 layer, (b) just the readout cluster, (c) a
random 88-unit set, at newline positions, and report the cluster's
shift as a fraction of the whole-mlp17 shift.

If the readout cluster carries a LARGE fraction (localized), then
readout circuits ARE necessary bottlenecks -- a real contrast with the
diffuse feature-building layers, and evidence the rank-8 readout end
is genuinely concentrated. If it is also ~0 (diffuse), redundancy is
universal even at the readout end.

Margin = P(newline) - P(sentence-end), at newline/sentence-end
positions.

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY + IDENTITY: mlp17 top-300 reproduces its sizes
      and the newline readout cluster is the LARGEST (88 units); no
      ablation reproduces the true margin -- VOIDS on failure;
  (a) WHOLE-LAYER effect: report the whole-mlp17 newline-margin shift;
  (b) LOCALIZATION FRACTION (the finding): the readout cluster's shift
      as a fraction of the whole-mlp17 shift. Prediction, CONTRASTING
      610: CONCENTRATED (>= 0.3) -- readout circuits are localizable,
      unlike feature-building ones. Whether it confirms this or comes
      out diffuse like mlp0 is the decisive result;
  (c) SPECIFICITY: the readout cluster's |shift| exceeds a random
      88-unit set's |shift| by >= 3x;
  NULL: the random 88-unit set's |shift| is small relative to the
      whole layer (< 0.2 of it)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
LJ = 17
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_readout_localization_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
NL1, NL2, PER, QUO = 198, 628, 13, 1  # newline vs sentence-end


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
    topk = order[:TOPK].numpy()  # top-300
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
    expect = [88, 67, 67, 18, 13, 9, 8, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1]
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    ranked = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
    return ranked[0][1], sizes == expect  # largest = mlp17 newline readout (88)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    article, repro = recover_article_cluster(fresh)
    print(f'(0a) reproduced + article cluster n={len(article)}: '
          f'{"HELD" if repro and len(article) == 88 else "FAILED"}', flush=True)
    if not (repro and len(article) == 88):
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
    art = ((nxt == NL1) | (nxt == NL2) |
           (nxt == PER) | (nxt == QUO)).numpy()
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
            mg = p[..., NL1] + p[..., NL2] - p[..., PER] - p[..., QUO]
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

    pa = abs(whole) > 1e-3  # whole-layer has a real newline effect to fraction against
    frac = clust / whole if whole != 0 else None
    localized = 'concentrated' if (frac is not None and abs(frac) >= 0.3) else 'diffuse'
    pc = abs(clust) >= 3 * abs(rand) if rand != 0 else abs(clust) > abs(rand)
    null_ok = abs(rand) < 0.2 * abs(whole)
    print(f'(a) whole-mlp17 newline-margin shift {whole:+.4f}: '
          f'{"HELD (real effect)" if pa else "FAILED (no whole-layer effect)"}',
          flush=True)
    print(f'(b) newline cluster shift {clust:+.4f}, fraction of whole '
          f'{frac:.2f} -> {localized}', flush=True)
    print(f'(c) random 88-unit shift {rand:+.4f} (cluster/random '
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
