"""MLP0 CLUSTER13/7 LOGIT MARGIN -- causally verify the other two
named clusters from 581, using the properly-powered metric from the
start (583 showed aggregate CE is too diluted; logit margin resolved
a real, partial effect for cluster 8 on the first correctly-scoped
try).

581 named three clusters from real activating context: cluster 8
(a/an-vs-the, causally tested in 582/583 -- confirmed as a selective
a/an promoter, NOT the full bidirectional discriminator claimed).
This tests the remaining two the same way, directly with logit
margins (no aggregate-CE detour this time):
  CLUSTER 13 (76 units): period/exclamation vs dash. margin =
    logit('.') + logit('!') - logit(dash tokens: '--','—',' --',' —').
  CLUSTER 7 (29 units): first-person auxiliary vs contraction. margin
    = logit(' am') + logit(' had') + logit(' wasn') - logit("'s").
Same ablation protocol as 582/583: mean-fill ONLY the named cluster's
units in mlp0's hidden vector, everything else exact; compare the
margin shift at each cluster's own positive-class target positions
and negative-class target positions against a size-matched random-
unit control, using DIFFERENCES (not ratios of small numbers, 582's
lesson).

REGISTERED PREDICTIONS (per cluster, same structure as 583):
  (0) IDENTITY: empty ablation leaves margins unchanged;
  (a) POSITIVE-CLASS SHRINKAGE: at the cluster's positive-class
      target positions, ablation shrinks the margin by >= 3x the
      random-unit control's shrinkage;
  (b) NEGATIVE-CLASS MIRROR: at the negative-class target positions,
      ablation shifts the margin toward 0 by >= 3x the random
      control -- 583's finding predicts this FAILS for at least one
      side (cluster 8's did), so this is a real test, not a foregone
      pass;
  (c) MAGNITUDES: report both deltas in absolute logits, no bar;
  NULL: at a class unrelated to either cluster (digit-target
      positions), the cluster's delta-margin is not larger in
      magnitude than the random control's."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV
D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_cluster13_7_logit_margin_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20

CLUSTERS = {
    'cluster13': {'expect_n': 76, 'pos_tok': [13, 0],  # '.', '!'
                  'neg_tok': [438, 960, 1377, 851]},   # '--','—',' --',' —'
    'cluster7': {'expect_n': 29, 'pos_tok': [716, 550, 2492],  # am,had,wasn
                 'neg_tok': [338]},                    # 's
}


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


def recover_clusters(fresh):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
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
    expect = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    ranked = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
    return ranked, sizes == expect, sizes, H


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    ranked, repro_ok, sizes, Hall = recover_clusters(fresh)
    print(f'(0a) reproduced sizes {sizes}: '
          f"{'HELD' if repro_ok else 'FAILED'}", flush=True)
    named = {'cluster13': ranked[1][1], 'cluster7': ranked[2][1]}
    for k, cfg in CLUSTERS.items():
        assert len(named[k]) == cfg['expect_n'], \
            f"{k} size mismatch: {len(named[k])} vs {cfg['expect_n']}"

    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight
    b = mlp.Down_bias
    H_full = Dw.shape[1]
    hmean = Hall.mean(0).to(DEV)

    nxt = fresh[:, 1:257]

    def target_mask(tok_ids):
        msk = torch.zeros(NFRESH, T, dtype=torch.bool)
        for t in tok_ids:
            msk |= (nxt == t)
        return msk

    digit_mask = torch.zeros(NFRESH, T, dtype=torch.bool)
    for r_ in range(NFRESH):
        for q in range(T):
            s = cl.d1(int(nxt[r_, q])).strip()
            if s and s[0].isdigit():
                digit_mask[r_, q] = True

    def margins(unit_ids, pos_tok, neg_tok):
        msk = torch.zeros(H_full, device=DEV)
        if unit_ids:
            msk[torch.tensor(unit_ids, device=DEV)] = 1.0
        out = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            B = bb.shape[0]

            def fh(mo, args, o_, msk=msk):
                X = args[0].float()
                h = (X @ L.T) * (X @ R.T)
                h = h * (1 - msk) + hmean[None, None, :] * msk
                return (h @ Dw.T.float() + b.float()).to(o_.dtype)
            hh = mlp.register_forward_hook(fh)
            x = F.rms_norm(m.transformer.wte(idx), (D,))
            x0 = x
            v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            pos = sum(lg[..., t] for t in pos_tok)
            neg = sum(lg[..., t] for t in neg_tok)
            out[i:i + B] = (pos - neg).cpu()
            hh.remove()
        return out

    rng = np.random.default_rng(5)
    results = {}
    for k, cfg in CLUSTERS.items():
        units = named[k]
        pos_mask = target_mask(cfg['pos_tok'])
        neg_mask = target_mask(cfg['neg_tok'])
        n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
        print(f'\n=== {k} (n_units={len(units)}) n_pos={n_pos} n_neg={n_neg} ===',
              flush=True)

        base_mg = margins([], cfg['pos_tok'], cfg['neg_tok'])
        ident = float(margins([], cfg['pos_tok'], cfg['neg_tok']).mean()
                      - base_mg.mean())
        p0 = abs(ident) < 1e-3
        print(f'(0b) identity {ident:+.6f}: {"HELD" if p0 else "FAILED"}',
              flush=True)

        mg_c = margins(units, cfg['pos_tok'], cfg['neg_tok'])
        rand_units = rng.choice(list(range(H_full)), size=len(units),
                                replace=False).tolist()
        mg_r = margins(rand_units, cfg['pos_tok'], cfg['neg_tok'])

        def cls_delta(mg, mask):
            d = (mg - base_mg)[mask]
            return float(d.mean()) if d.numel() else None

        dc_pos = cls_delta(mg_c, pos_mask)
        dc_neg = cls_delta(mg_c, neg_mask)
        dc_dig = cls_delta(mg_c, digit_mask)
        dr_pos = cls_delta(mg_r, pos_mask)
        dr_neg = cls_delta(mg_r, neg_mask)
        dr_dig = cls_delta(mg_r, digit_mask)
        base_pos = float(base_mg[pos_mask].mean()) if n_pos else None
        base_neg = float(base_mg[neg_mask].mean()) if n_neg else None

        ratio_a = (dc_pos / dr_pos) if dr_pos not in (None, 0) else None
        ratio_b = (dc_neg / dr_neg) if dr_neg not in (None, 0) else None
        pa = dc_pos is not None and dc_pos < 0 and ratio_a is not None and ratio_a >= 3.0
        pb = dc_neg is not None and dc_neg > 0 and ratio_b is not None and ratio_b >= 3.0
        null_ok = (dc_dig is not None and dr_dig is not None and
                   abs(dc_dig) <= abs(dr_dig) + 0.05)
        print(f'(a) pos-class: base {base_pos:.3f}, cluster delta {dc_pos:+.4f}, '
              f'random delta {dr_pos:+.4f}, ratio {ratio_a}: '
              f"{'HELD' if pa else 'FAILED'}", flush=True)
        print(f'(b) neg-class: base {base_neg:.3f}, cluster delta {dc_neg:+.4f}, '
              f'random delta {dr_neg:+.4f}, ratio {ratio_b}: '
              f"{'HELD' if pb else 'FAILED'}", flush=True)
        print(f'NULL: digit delta {dc_dig} vs random {dr_dig}: '
              f"{'ok' if null_ok else 'CHECK'}", flush=True)

        results[k] = {'n_units': len(units), 'n_pos': n_pos, 'n_neg': n_neg,
                      'identity': ident, 'pred_0': bool(p0),
                      'base_pos': base_pos, 'base_neg': base_neg,
                      'delta_pos': dc_pos, 'delta_neg': dc_neg,
                      'delta_digit': dc_dig,
                      'random_delta_pos': dr_pos, 'random_delta_neg': dr_neg,
                      'random_delta_digit': dr_dig,
                      'ratio_pos': ratio_a, 'ratio_neg': ratio_b,
                      'pred_a': bool(pa), 'pred_b': bool(pb),
                      'null_ok': bool(null_ok)}

    out = {'repro_ok': bool(repro_ok), 'sizes': sizes, 'clusters': results,
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
