"""MLP0 CLUSTER8 LOGIT MARGIN -- the properly-powered causal test 582
queued: replace aggregate cross-entropy (diluted by 50000 competing
vocabulary items, every comparison came back UNEVALUABLE) with a
DIRECT logit margin tied to the specific claim.

581's correlational reading: mlp0 hidden-unit cluster 8 (101 units,
579) is a SIGNED a/an-vs-the discriminator -- positive activation
contexts predict 'a'/'an', negative predict 'the'/'The'. 582's causal
test (mean-fill cluster 8, measure whole-model CE split by target
class) was UNEVALUABLE across the board: the effect on aggregate CE
was too small relative to itself to read a direction, let alone
confirm or refute selectivity.

This redoes the causal test with a metric that isolates the specific
claim instead of diluting it across the whole vocabulary:
    margin(position) = (logit[' a'] + logit[' an'])
                      - (logit[' the'] + logit[' The'])
computed at every position, before and after mean-filling cluster 8's
units. If cluster 8 is really the discriminator, ablating it should
DEGRADE the model's article discrimination specifically -- shrink the
margin toward 0 -- at both indefinite-target positions (where margin
should start positive) and definite-target positions (where margin
should start negative), a two-sided mirror-image signature. Margins
are DIFFERENCES, not ratios of small numbers, so 582's near-zero-
denominator failure mode does not apply here -- delta-margin is
well-defined and interpretable even when small.

REGISTERED PREDICTIONS:
  (0) IDENTITY: ablating an empty unit set leaves the margin exactly
      unchanged (sanity);
  (a) INDEFINITE-TARGET SHRINKAGE: at indefinite-article-target
      positions, ablating cluster 8 reduces the (positive) margin by
      an amount that is at least 3x the reduction from ablating a
      size-matched random unit subset;
  (b) DEFINITE-TARGET MIRROR: at definite-article-target positions,
      ablating cluster 8 INCREASES the (negative) margin toward 0 by
      at least 3x the shift from the random control -- the mirror-
      image signature of a shared discriminator, not just noise in
      one direction;
  (c) MAGNITUDE: report both deltas in absolute nats/logits, since
      (a)/(b) are ratios against a control and could pass on tiny
      absolute numbers -- no bar, just the numbers for the record;
  NULL: at digit-target positions (unrelated class), cluster 8's
      delta-margin is no larger in magnitude than the random control's
      -- the effect (if any) is article-specific, not a generic
      disruption from removing 101 units."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV
D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_cluster8_logit_margin_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383  # ' a',' an',' the',' The'


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
    return H, Dw.cpu()


def recover_cluster8(fresh):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    H, Dw = capture(fresh)
    Nfull = H.shape[0]
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(Nfull, generator=g)[:NSAMP]
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
    c8 = max(by_cluster, key=lambda k: len(by_cluster[k]))
    return by_cluster[c8], sizes == expect, sizes


def isart_indef(s):
    return s.strip() in ('a', 'an', 'A', 'An')


def isart_def(s):
    return s.strip() in ('the', 'The')


def isdig(s):
    z = s.strip()
    return bool(z) and z[0].isdigit()


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cluster8, repro_ok, sizes = recover_cluster8(fresh)
    print(f'(0a) reproduced sizes {sizes}: '
          f"{'HELD' if repro_ok else 'FAILED'} | cluster8 n={len(cluster8)}",
          flush=True)

    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight
    b = mlp.Down_bias
    H_full = Dw.shape[1]

    cur = fresh[:, :256]
    nxt = fresh[:, 1:257]
    indef = torch.zeros(NFRESH, T, dtype=torch.bool)
    defi = torch.zeros(NFRESH, T, dtype=torch.bool)
    digit = torch.zeros(NFRESH, T, dtype=torch.bool)
    for r_ in range(NFRESH):
        for q in range(T):
            s = cl.d1(int(nxt[r_, q]))
            if isart_indef(s):
                indef[r_, q] = True
            elif isart_def(s):
                defi[r_, q] = True
            elif isdig(s):
                digit[r_, q] = True
    print(f'n indef={int(indef.sum())} n defi={int(defi.sum())} '
          f'n digit={int(digit.sum())}', flush=True)

    Hall, _ = capture(fresh)
    hmean = Hall.mean(0).to(DEV)

    def margins(unit_ids):
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
            mg = (lg[..., TOK_A] + lg[..., TOK_AN]
                  - lg[..., TOK_THE] - lg[..., TOK_THE2])
            out[i:i + B] = mg.cpu()
            hh.remove()
        return out

    base_mg = margins([])
    ident = float(margins([]).mean() - base_mg.mean())
    p0 = abs(ident) < 1e-3
    print(f'(0b) identity {ident:+.6f}: {"HELD" if p0 else "FAILED"}',
          flush=True)

    mg8 = margins(cluster8)
    rng = np.random.default_rng(5)
    rand_units = rng.choice(list(range(H_full)), size=len(cluster8),
                            replace=False).tolist()
    mg_rand = margins(rand_units)

    def cls_delta(mg, mask):
        d = (mg - base_mg)[mask]
        return float(d.mean()) if d.numel() else None

    d8_indef = cls_delta(mg8, indef)
    d8_defi = cls_delta(mg8, defi)
    d8_digit = cls_delta(mg8, digit)
    dr_indef = cls_delta(mg_rand, indef)
    dr_defi = cls_delta(mg_rand, defi)
    dr_digit = cls_delta(mg_rand, digit)
    base_indef = float(base_mg[indef].mean())
    base_defi = float(base_mg[defi].mean())

    ratio_a = (d8_indef / dr_indef) if dr_indef not in (None, 0) else None
    ratio_b = (d8_defi / dr_defi) if dr_defi not in (None, 0) else None
    pa = (d8_indef is not None and d8_indef < 0 and
          ratio_a is not None and ratio_a >= 3.0)
    pb = (d8_defi is not None and d8_defi > 0 and
          ratio_b is not None and ratio_b >= 3.0)
    print(f'(a) indef: base margin {base_indef:.3f}, cluster8 delta '
          f'{d8_indef:+.4f}, random delta {dr_indef:+.4f}, ratio {ratio_a}: '
          f"{'HELD' if pa else 'FAILED'}", flush=True)
    print(f'(b) defi: base margin {base_defi:.3f}, cluster8 delta '
          f'{d8_defi:+.4f}, random delta {dr_defi:+.4f}, ratio {ratio_b}: '
          f"{'HELD' if pb else 'FAILED'}", flush=True)
    print(f'(c) magnitudes -- indef delta {d8_indef}, defi delta {d8_defi}',
          flush=True)

    null_ok = (d8_digit is not None and dr_digit is not None and
               abs(d8_digit) <= abs(dr_digit) + 0.05)
    print(f'NULL: cluster8 digit delta {d8_digit} vs random {dr_digit}: '
          f"{'ok' if null_ok else 'CHECK'}", flush=True)

    out = {'repro_ok': bool(repro_ok), 'cluster8_n': len(cluster8),
           'identity': ident, 'pred_0': bool(p0),
           'base_indef_margin': base_indef, 'base_defi_margin': base_defi,
           'cluster8': {'indef_delta': d8_indef, 'defi_delta': d8_defi,
                        'digit_delta': d8_digit},
           'random_control': {'indef_delta': dr_indef, 'defi_delta': dr_defi,
                              'digit_delta': dr_digit},
           'ratio_indef': ratio_a, 'ratio_defi': ratio_b,
           'pred_a': bool(pa), 'pred_b': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
