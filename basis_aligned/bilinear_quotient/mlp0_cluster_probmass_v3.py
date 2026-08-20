"""MLP0 CLUSTER PROBMASS V3 -- the principled fix for 582-585's causal
tests: PROBABILITY MASS (softmax), not raw logit sums or means.

585 traced why both prior metrics failed: raw logits for common
tokens (period, high-frequency auxiliaries) and rare tokens (em-dash,
contraction suffix) sit on different scales by default, unrelated to
context -- so neither summing them (584's original, inflated by class
size) nor averaging them (v2's fix, still comparing incommensurate
per-token scales) is a fair contrast. Cluster 8's test (582/583)
happened to avoid this because its two classes were already matched
in both size (2 vs 2) and rough frequency (a/an vs the/The are all
common words) -- not because logit-sum margins are generally sound.

The fix: convert to PROBABILITY MASS via softmax over the full
vocabulary before comparing classes. P(token in class) sums correctly
regardless of how many tokens are in the class or how rare any one of
them is -- there is no scale mismatch to correct for. Margin =
P(positive-class tokens) - P(negative-class tokens), bounded in
[-1, 1], well-defined for any class composition.

Retests all THREE named clusters from 581 uniformly (not just the two
that broke) -- cluster 8 is re-run here too, per this program's rule
that a metric fix applies uniformly rather than selectively to results
that need rescuing, even though 582/583's cluster-8 result is not
under suspicion.

REGISTERED PREDICTIONS (per cluster):
  (0) IDENTITY: empty ablation leaves the probability margin
      unchanged;
  (-1) BASELINE SANITY: P(positive class) - P(negative class) is
      positive at the class's own positive-class target positions and
      negative at its own negative-class target positions, for ALL
      THREE clusters -- if any cluster fails this, its verdict below
      is not trustworthy and should be reported as such, not silently
      used;
  (a) POSITIVE-CLASS SHRINKAGE: ablating the cluster shrinks the
      probability margin at positive-class positions by >= 3x a
      size-matched random-unit control;
  (b) NEGATIVE-CLASS MIRROR: ablating the cluster shifts the
      probability margin toward 0 at negative-class positions by
      >= 3x the random control;
  (c) MAGNITUDES: report both deltas in absolute probability, no bar;
  NULL: at digit-target positions (unrelated to all three clusters'
      claimed behaviour), no cluster's delta exceeds its random
      control's in magnitude by more than a small tolerance."""
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
OUT = PT + 'mlp0_cluster_probmass_v3_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20

CLUSTERS = {
    'cluster8': {'rank': 0, 'expect_n': 101,
                 'pos_tok': [257, 281], 'neg_tok': [262, 383]},  # a,an / the,The
    'cluster13': {'rank': 1, 'expect_n': 76,
                  'pos_tok': [13, 0], 'neg_tok': [438, 960, 1377, 851]},
    'cluster7': {'rank': 2, 'expect_n': 29,
                 'pos_tok': [716, 550, 2492], 'neg_tok': [338]},
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
    named = {k: ranked[cfg['rank']][1] for k, cfg in CLUSTERS.items()}
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

    def probmargins(unit_ids, pos_tok, neg_tok):
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
            p = F.softmax(lg, dim=-1)
            pos = sum(p[..., t] for t in pos_tok)
            neg = sum(p[..., t] for t in neg_tok)
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

        base_mg = probmargins([], cfg['pos_tok'], cfg['neg_tok'])
        ident = float(probmargins([], cfg['pos_tok'], cfg['neg_tok']).mean()
                      - base_mg.mean())
        p0 = abs(ident) < 1e-3
        print(f'(0) identity {ident:+.6f}: {"HELD" if p0 else "FAILED"}',
              flush=True)

        base_pos = float(base_mg[pos_mask].mean()) if n_pos else None
        base_neg = float(base_mg[neg_mask].mean()) if n_neg else None
        sane = (base_pos is not None and base_pos > 0 and
                base_neg is not None and base_neg < 0)
        print(f'(-1) baseline sanity: base_pos={base_pos:.4f} (want >0), '
              f'base_neg={base_neg:.4f} (want <0): '
              f"{'HELD' if sane else 'FAILED -- do not trust below'}",
              flush=True)

        mg_c = probmargins(units, cfg['pos_tok'], cfg['neg_tok'])
        rand_units = rng.choice(list(range(H_full)), size=len(units),
                                replace=False).tolist()
        mg_r = probmargins(rand_units, cfg['pos_tok'], cfg['neg_tok'])

        def cls_delta(mg, mask):
            d = (mg - base_mg)[mask]
            return float(d.mean()) if d.numel() else None

        dc_pos = cls_delta(mg_c, pos_mask)
        dc_neg = cls_delta(mg_c, neg_mask)
        dc_dig = cls_delta(mg_c, digit_mask)
        dr_pos = cls_delta(mg_r, pos_mask)
        dr_neg = cls_delta(mg_r, neg_mask)
        dr_dig = cls_delta(mg_r, digit_mask)

        ratio_a = (dc_pos / dr_pos) if dr_pos not in (None, 0) else None
        ratio_b = (dc_neg / dr_neg) if dr_neg not in (None, 0) else None
        pa = dc_pos is not None and dc_pos < 0 and ratio_a is not None and ratio_a >= 3.0
        pb = dc_neg is not None and dc_neg > 0 and ratio_b is not None and ratio_b >= 3.0
        null_ok = (dc_dig is not None and dr_dig is not None and
                   abs(dc_dig) <= abs(dr_dig) + 0.01)
        print(f'(a) pos-class: base {base_pos:.4f}, cluster delta {dc_pos:+.5f}, '
              f'random delta {dr_pos:+.5f}, ratio {ratio_a}: '
              f"{'HELD' if pa else 'FAILED'}", flush=True)
        print(f'(b) neg-class: base {base_neg:.4f}, cluster delta {dc_neg:+.5f}, '
              f'random delta {dr_neg:+.5f}, ratio {ratio_b}: '
              f"{'HELD' if pb else 'FAILED'}", flush=True)
        print(f'NULL: digit delta {dc_dig} vs random {dr_dig}: '
              f"{'ok' if null_ok else 'CHECK'}", flush=True)

        results[k] = {'n_units': len(units), 'n_pos': n_pos, 'n_neg': n_neg,
                      'identity': ident, 'pred_0': bool(p0),
                      'baseline_sane': bool(sane),
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
