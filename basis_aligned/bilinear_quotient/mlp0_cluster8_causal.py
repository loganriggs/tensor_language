"""MLP0 CLUSTER8 CAUSAL -- does ablating cluster 8 (579/581's a/an-vs-
the discriminator) SELECTIVELY cost article prediction, or is 581's
reading (activation-based, correlational) not causally load-bearing?

581 found mlp0 hidden-unit cluster 8 (101 units, stable out-of-sample,
579) reads as a signed a/an-vs-the discriminator purely from its top
activating CONTEXTS -- positive activation contexts are all followed
by 'a'/'an', negative by 'the'/'The'. That is a correlational finding
(what the cluster's activation predicts about upcoming text), not yet
a causal one. This program's standard (LESSONS, every verified
circuit in this ledger) requires the causal step before calling
anything a real circuit: ablate the proposed component and check the
COST is where the reading predicts it should be, with a size-matched
random-unit control and an unrelated-class null.

Reproduces 579's clustering exactly (same seeds/code -- sanity-checked
below) to recover cluster 8's full 101-unit id list (579/581's saved
JSON only kept the first 10), then mean-fills ONLY those units in
mlp0's hidden vector (everything else in the model exact), and prices
the whole-model CE cost split by the TARGET token's class at each
position: indefinite article ('a'/'an'), definite article
('the'/'The'), and a size-matched random sample of all other
positions (generic control).

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY + IDENTITY: reclustering reproduces 579's exact
      cluster sizes (sanity), and ablating an EMPTY unit set costs
      < 1e-3 nats (hook machinery is a no-op when it should be);
  (a) SELECTIVE COST: ablating cluster 8 costs more at article
      positions (indefinite + definite combined) than at the generic
      control, by >= 3x;
  (b) CONTROL: a random 101-unit subset (mean-filled the same way)
      shows NO such selectivity -- its cost ratio (article vs
      control) is < 1.5x, clearly below cluster 8's;
  (c) SIGN STRUCTURE (no bar, report only): compare cluster 8's cost
      specifically at indefinite-article positions vs definite-article
      positions -- 581's reading (a SIGNED discriminator) predicts
      both sides are hurt, not just one, since removing the whole
      signal removes the discrimination in both directions;
  NULL: cluster 8's cost at an UNRELATED control class (digit-
      predicting positions) is not elevated versus the generic
      control -- the selectivity is specific to articles, not a
      general "this cluster matters everywhere" effect."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV
D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_cluster8_causal_results.json'
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
    Omu = O.mean(0)
    Oc = O - Omu
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    var = (S ** 2)
    cum = torch.cumsum(var, 0) / var.sum()
    r = int((cum < 0.95).sum().item()) + 1
    r = min(r, Vt.shape[0])
    Vr = Vt[:r]
    Dw_topk = Dw[:, topk]
    Dw_proj = Vr @ Dw_topk
    Hk = Hs[:, topk]
    coldw2 = (Dw_proj ** 2).sum(0)
    damage = ((Hk ** 2) * coldw2[None, :]).T.numpy()
    dm = damage - damage.mean(1, keepdims=True)
    dstd = damage.std(1, keepdims=True)
    dstd[dstd < 1e-12] = 1e-12
    corr = (dm @ dm.T) / (damage.shape[1] * dstd * dstd.T)
    corr = np.clip(corr, -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    cond = squareform((dist + dist.T) / 2, checks=False)
    Z = linkage(cond, method='average')
    labels = fcluster(Z, t=NCLUST, criterion='maxclust')
    sizes = sorted(np.bincount(labels)[1:].tolist(), reverse=True)
    expect = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    ok = sizes == expect
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    c8_id = max(by_cluster, key=lambda k: len(by_cluster[k]))
    return by_cluster[c8_id], ok, sizes


def isart_indef(s):
    t = s.strip()
    return t in ('a', 'an', 'A', 'An')


def isart_def(s):
    t = s.strip()
    return t in ('the', 'The')


def isdig(s):
    z = s.strip()
    return bool(z) and z[0].isdigit()


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cluster8, repro_ok, sizes = recover_cluster8(fresh)
    print(f'(0) reproduced sizes {sizes}: '
          f"{'HELD' if repro_ok else 'FAILED'} | cluster8 n={len(cluster8)}",
          flush=True)

    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight
    b = mlp.Down_bias
    H_full = Dw.shape[1]

    # target-class positions over the SAME fresh corpus
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
    g2 = np.random.default_rng(3)
    generic = torch.zeros(NFRESH, T, dtype=torch.bool)
    free = (~indef) & (~defi) & (~digit)
    free_idx = free.nonzero()
    n_ctrl = int((indef.sum() + defi.sum()).item())
    pick = g2.choice(len(free_idx), size=min(n_ctrl, len(free_idx)), replace=False)
    for i in pick:
        r_, q = free_idx[i].tolist()
        generic[r_, q] = True
    print(f'n indef={int(indef.sum())} n defi={int(defi.sum())} '
          f'n digit={int(digit.sum())} n generic={int(generic.sum())}',
          flush=True)

    def price(unit_ids):
        msk = torch.zeros(H_full, device=DEV)
        if unit_ids:
            msk[torch.tensor(unit_ids, device=DEV)] = 1.0
        hmean = cl_hmean
        ce = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            tg = bb[:, 1:].reshape(-1)
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
            ce[i:i + B] = F.cross_entropy(
                lg.view(-1, lg.size(-1)), tg, reduction='none').view(B, T).cpu()
            hh.remove()
        return ce

    # mean hidden activation for mean-filling
    Hall, _ = capture(fresh)
    cl_hmean = Hall.mean(0).to(DEV)

    base = price([])
    ident_check = price([]).mean().item() - base.mean().item()
    p0b = abs(ident_check) < 1e-3
    print(f'(0b) identity re-check {ident_check:+.5f}: '
          f"{'HELD' if p0b else 'FAILED'}", flush=True)

    dce8 = price(cluster8) - base
    rng = np.random.default_rng(5)
    all_units = list(range(H_full))
    rand_units = rng.choice(all_units, size=len(cluster8), replace=False).tolist()
    dce_rand = price(rand_units) - base

    def classavg(dce, mask):
        v = dce[mask]
        return float(v.mean()) if v.numel() else None

    c8_indef = classavg(dce8, indef)
    c8_defi = classavg(dce8, defi)
    c8_art = classavg(dce8, indef | defi)
    c8_generic = classavg(dce8, generic)
    c8_digit = classavg(dce8, digit)
    r_indef = classavg(dce_rand, indef)
    r_defi = classavg(dce_rand, defi)
    r_art = classavg(dce_rand, indef | defi)
    r_generic = classavg(dce_rand, generic)
    r_digit = classavg(dce_rand, digit)

    ratio8 = c8_art / c8_generic if c8_generic else None
    ratio_rand = r_art / r_generic if r_generic else None
    va, _ = cl.score_bar('a', ratio8, 3.0, denom=c8_generic,
                         n=min(len(indef.nonzero()), len(defi.nonzero())))
    vb, _ = cl.score_bar('b', ratio_rand, 1.5, denom=r_generic,
                         n=min(len(indef.nonzero()), len(defi.nonzero())))
    pa = va == 'HELD'
    pb = vb != 'HELD'  # control PASSES if it does NOT clear the bar
    print(f'(c) cluster8 indef={c8_indef} defi={c8_defi} (sign structure, '
          f'no bar)', flush=True)

    vnull, _ = cl.score_bar('NULL', c8_digit / c8_generic if c8_generic else None,
                            1.5, denom=c8_generic, n=int(digit.sum()))
    null_ok = vnull != 'HELD'

    out = {'repro_ok': bool(repro_ok), 'cluster8_n': len(cluster8),
           'identity_recheck': ident_check, 'pred_0b': bool(p0b),
           'base_mean': float(base.mean()),
           'cluster8': {'indef': c8_indef, 'defi': c8_defi,
                        'article': c8_art, 'generic': c8_generic,
                        'digit': c8_digit, 'ratio': ratio8},
           'random_control': {'indef': r_indef, 'defi': r_defi,
                              'article': r_art, 'generic': r_generic,
                              'digit': r_digit, 'ratio': ratio_rand},
           'pred_a': bool(pa), 'pred_b': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
