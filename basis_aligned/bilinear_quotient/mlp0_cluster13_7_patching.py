"""MLP0 CLUSTER13/7 PATCHING -- extend 592's validated activation-
patching method to the two clusters margin/mean-ablation testing
(582-586) never resolved.

592 showed activation patching -- replacing a real activation from an
opposite-favoring context, not deleting to a mean -- gives a clean,
specific, correctly-signed causal confirmation for cluster 8, where
three margin-based designs gave metric-unstable readings. This
applies the exact same method to cluster 13 (period/exclamation vs
dash) and cluster 7 (first-person auxiliary vs contraction), the two
clusters 586 left unresolved.

592's registered absolute-magnitude bar for (a) was set without a
real sense of scale and failed even though the comparative checks
(b)/(c)/NULL all confirmed a real, specific, correctly-signed effect
-- so this run treats (a)'s sign (not a fixed magnitude) as the
primary bar, and leans on the comparative checks as the load-bearing
evidence, per that lesson.

REGISTERED PREDICTIONS (per cluster, mirroring 592's structure):
  (0) IDENTITY: patching a target's own activation into itself
      changes nothing (< 1e-4) -- sanity, VOIDS that cluster's run;
  (a) CORRECT SIGN: patching a positive-class-favoring source into a
      negative-class-favoring target moves the margin toward the
      positive class (delta > 0) -- the primary bar this time, not a
      fixed magnitude (592's lesson);
  (b) SPECIFICITY vs RANDOM: the cluster's patch effect is >= 3x a
      size-matched random-unit patch;
  (c) SPECIFICITY vs SOURCE CLASS: a same-class (negative-into-
      negative) control patch produces a much smaller shift than the
      real cross-class patch;
  NULL: the same positive-class-favoring-source patch, applied at
      digit-target positions, produces a shift under 30% the size of
      the real-target effect."""
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
OUT = PT + 'mlp0_cluster13_7_patching_results.json'
NFRESH = 64  # must match every other cluster script for exact reproduction
NSAMP = 4000
TOPK = 300
NCLUST = 20
NPAIRS = 80

CLUSTERS = {
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
    return ranked, sizes == expect, sizes


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    ranked, repro_ok, sizes = recover_clusters(fresh)
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

    idx_all = fresh[:, :256].to(DEV)
    nxt = fresh[:, 1:257]
    N_docs = fresh.shape[0]

    digit_mask = torch.zeros(N_docs, T, dtype=torch.bool)
    for r_ in range(N_docs):
        for q in range(T):
            s = cl.d1(int(nxt[r_, q])).strip()
            if s and s[0].isdigit():
                digit_mask[r_, q] = True

    @torch.no_grad()
    def forward_capture_h_and_margin(pos_tok, neg_tok):
        cap = {}
        hk = mlp.register_forward_pre_hook(
            lambda mo_, a_: cap.__setitem__('X', a_[0]))
        x = F.rms_norm(m.transformer.wte(idx_all), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        hk.remove()
        X = cap['X'].float()
        h = (X @ L.T) * (X @ R.T)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        pos = sum(p[..., t] for t in pos_tok)
        neg = sum(p[..., t] for t in neg_tok)
        margin = pos - neg
        return h.reshape(N_docs, T, H_full), margin.reshape(N_docs, T)

    def patched_margin(mlp, L, R, Dw, b, tgt_rq, patch_units, src_h_vec,
                       pos_tok, neg_tok):
        tr, tq = tgt_rq
        msk = torch.zeros(H_full, device=DEV)
        msk[torch.tensor(patch_units, device=DEV)] = 1.0

        def fh(mo, args, o_):
            X_in = args[0].float()
            h = (X_in @ L.T) * (X_in @ R.T)
            h[:, tq, :] = h[:, tq, :] * (1 - msk) + src_h_vec[None, :] * msk
            return (h @ Dw.T.float() + b.float()).to(o_.dtype)
        hh = mlp.register_forward_hook(fh)
        idx1 = idx_all[tr:tr + 1]
        x = F.rms_norm(m.transformer.wte(idx1), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        hh.remove()
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        pos = sum(p[..., t] for t in pos_tok)
        neg = sum(p[..., t] for t in neg_tok)
        return float((pos - neg)[0, tq])

    rng2 = np.random.default_rng(13)
    results = {}
    for k, cfg in CLUSTERS.items():
        units = named[k]
        pos_tok, neg_tok = cfg['pos_tok'], cfg['neg_tok']
        print(f'\n=== {k} (n_units={len(units)}) ===', flush=True)
        h_all, margin_all = forward_capture_h_and_margin(pos_tok, neg_tok)

        src_pos = (margin_all > 0.02).nonzero()
        tgt_pos = (margin_all < -0.02).nonzero()
        print(f'source positions: {src_pos.shape[0]}, target positions: '
              f'{tgt_pos.shape[0]}', flush=True)
        n0 = min(len(src_pos), len(tgt_pos))
        if n0 < 20:
            print(f'  void: too few positions ({n0})', flush=True)
            results[k] = {'void': 'too few positions', 'n0': n0}
            continue

        g = np.random.default_rng(9)
        pairs = [(tuple(src_pos[g.integers(len(src_pos))].tolist()),
                  tuple(tgt_pos[g.integers(len(tgt_pos))].tolist()))
                 for _ in range(NPAIRS)]
        same_class_pairs = [(tuple(tgt_pos[g.integers(len(tgt_pos))].tolist()),
                             tuple(tgt_pos[g.integers(len(tgt_pos))].tolist()))
                            for _ in range(NPAIRS)]

        ident_deltas = []
        for s_rq, t_rq in pairs[:10]:
            base = float(margin_all[t_rq])
            own_h = h_all[t_rq[0], t_rq[1]].to(DEV)
            pm = patched_margin(mlp, L, R, Dw, b, t_rq, units, own_h,
                                pos_tok, neg_tok)
            ident_deltas.append(pm - base)
        ident = float(np.mean(np.abs(ident_deltas)))
        p0 = ident < 1e-4
        print(f'(0) identity mean |delta| {ident:.6f}: '
              f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
        if not p0:
            results[k] = {'void': 'identity failed', 'ident': ident}
            continue

        rand_units = rng2.choice(list(range(H_full)), size=len(units),
                                 replace=False).tolist()
        a_deltas, a_rand_deltas = [], []
        for s_rq, t_rq in pairs:
            base = float(margin_all[t_rq])
            src_h = h_all[s_rq[0], s_rq[1]].to(DEV)
            pm_c = patched_margin(mlp, L, R, Dw, b, t_rq, units, src_h,
                                  pos_tok, neg_tok)
            pm_r = patched_margin(mlp, L, R, Dw, b, t_rq, rand_units, src_h,
                                  pos_tok, neg_tok)
            a_deltas.append(pm_c - base)
            a_rand_deltas.append(pm_r - base)
        a_mean = float(np.mean(a_deltas))
        a_rand_mean = float(np.mean(a_rand_deltas))
        pa = a_mean > 0
        pb = (a_mean >= 3.0 * a_rand_mean if a_rand_mean > 0
              else a_mean > a_rand_mean)
        print(f'(a) cluster patch mean delta {a_mean:+.5f} (sign correct): '
              f"{'HELD' if pa else 'FAILED'}", flush=True)
        print(f'(b) random-unit patch mean delta {a_rand_mean:+.5f} '
              f'(ratio {a_mean/max(a_rand_mean,1e-9):.2f}): '
              f"{'HELD' if pb else 'FAILED'}", flush=True)

        c_deltas = []
        for t1_rq, t2_rq in same_class_pairs:
            base = float(margin_all[t2_rq])
            src_h = h_all[t1_rq[0], t1_rq[1]].to(DEV)
            pm = patched_margin(mlp, L, R, Dw, b, t2_rq, units, src_h,
                                pos_tok, neg_tok)
            c_deltas.append(pm - base)
        c_mean = float(np.mean(c_deltas))
        pc = a_mean >= 3.0 * abs(c_mean) if c_mean != 0 else a_mean > 0
        print(f'(c) same-class patch mean delta {c_mean:+.5f} vs '
              f'cross-class {a_mean:+.5f}: {"HELD" if pc else "FAILED"}',
              flush=True)

        dig_pos = digit_mask.nonzero()
        null_deltas = []
        if len(dig_pos) >= 10:
            for _ in range(min(NPAIRS, len(dig_pos))):
                dr = tuple(dig_pos[g.integers(len(dig_pos))].tolist())
                sr = tuple(src_pos[g.integers(len(src_pos))].tolist())
                base = float(margin_all[dr])
                src_h = h_all[sr[0], sr[1]].to(DEV)
                pm = patched_margin(mlp, L, R, Dw, b, dr, units, src_h,
                                    pos_tok, neg_tok)
                null_deltas.append(pm - base)
        null_mean = float(np.mean(null_deltas)) if null_deltas else None
        null_ok = (null_mean is not None and
                   abs(null_mean) < 0.3 * abs(a_mean))
        print(f'NULL: digit-position delta {null_mean} vs real {a_mean:+.5f}: '
              f"{'ok' if null_ok else 'CHECK'}", flush=True)

        results[k] = {'n_units': len(units), 'n_source': int(src_pos.shape[0]),
                      'n_target': int(tgt_pos.shape[0]), 'pred_0': bool(p0),
                      'identity_abs_delta': ident, 'cluster_mean_delta': a_mean,
                      'random_mean_delta': a_rand_mean, 'pred_a': bool(pa),
                      'pred_b': bool(pb), 'same_class_mean_delta': c_mean,
                      'pred_c': bool(pc), 'null_mean_delta': null_mean,
                      'null_ok': bool(null_ok)}

    out = {'repro_ok': bool(repro_ok), 'sizes': sizes, 'clusters': results,
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
