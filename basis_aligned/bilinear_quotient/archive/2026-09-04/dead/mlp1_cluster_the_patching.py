"""MLP1 CLUSTER "THE" PATCHING -- is mlp1's article-adjacent cluster
(587's open lead) causally redundant with mlp0's cluster 8, or doing
something unrelated?

587 found mlp1's own unit-clustering has a 46-unit group that fires
strongly before " the" specifically (concentration 8/8 "determiner"),
distinct from mlp0's cluster 8 (which reads the full a/an-vs-the
CONTRAST). Flagged then as an untested open lead: does the model have
genuinely redundant/distributed article-choice machinery across
layers 0 and 1, or is mlp1's group doing something that only looks
similar from its top examples?

Now that 592/593 validated activation patching as the right tool for
this exact kind of claim (causal, specific, correctly-signed effects
for mlp0's cluster 8; a real reversed effect for cluster 7), apply it
to mlp1's cluster directly: does patching a real a/an-favoring
source's activation (captured from mlp1's hidden units, not mlp0's)
into a real the-favoring target shift the SAME whole-model a/an-vs-
the margin, the way mlp0's cluster 8 did?

REGISTERED PREDICTIONS:
  (0) IDENTITY + REPRODUCIBILITY: patching a target's own activation
      into itself changes nothing; reclustering mlp1's top-300 units
      reproduces 587's cluster sizes exactly -- both VOID the run on
      failure;
  (a) CORRECT SIGN: patching an a/an-favoring source into a the-
      favoring target moves the margin toward a/an (delta > 0) --
      the primary bar (592's lesson: sign first, not a fixed
      magnitude);
  (b) SPECIFICITY vs RANDOM: >= 3x a size-matched random-unit patch;
  (c) SPECIFICITY vs SOURCE CLASS: a same-class control patch
      produces a much smaller shift than the real cross-class patch;
  (d) THE REDUNDANCY QUESTION (no bar, report only): compare mlp1's
      cluster effect size directly against mlp0's cluster-8 effect
      size (592: +0.00219) -- comparable magnitude suggests genuinely
      distributed/redundant machinery; much smaller suggests mlp1's
      group is a minor or unrelated echo;
  NULL: the same patch at digit-target positions produces a shift
      under 30% the size of the real-target effect."""
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
OUT = PT + 'mlp1_cluster_the_patching_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383
NPAIRS = 80


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


def recover_the_cluster(fresh):
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
    # 587: the "determiner" (the-predicting) cluster was the SECOND
    # largest (46 units)
    the_cluster = ranked[1][1]
    return the_cluster, sizes == expect, sizes


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    the_cluster, repro_ok, sizes = recover_the_cluster(fresh)
    print(f'(0a) reproduced sizes {sizes}: '
          f"{'HELD' if repro_ok else 'FAILED'} | cluster n={len(the_cluster)}",
          flush=True)
    if not repro_ok:
        json.dump({'void': 'reclustering did not reproduce 587', 'sizes': sizes},
                   open(OUT, 'w'), indent=1)
        return

    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight
    b = mlp.Down_bias
    H_full = Dw.shape[1]

    @torch.no_grad()
    def forward_capture_h_and_margin(idx):
        cap = {}
        hk = mlp.register_forward_pre_hook(
            lambda mo_, a_: cap.__setitem__('X', a_[0]))
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        hk.remove()
        X = cap['X'].float()
        h = (X @ L.T) * (X @ R.T)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        margin = (p[..., TOK_A] + p[..., TOK_AN] - p[..., TOK_THE] - p[..., TOK_THE2])
        return h, margin

    idx_all = fresh[:, :256].to(DEV)
    nxt = fresh[:, 1:257]
    N_docs = fresh.shape[0]

    h_all, margin_all = forward_capture_h_and_margin(idx_all)
    h_all = h_all.reshape(N_docs, T, H_full)
    margin_all = margin_all.reshape(N_docs, T)

    digit_mask = torch.zeros(N_docs, T, dtype=torch.bool)
    for r_ in range(N_docs):
        for q in range(T):
            s = cl.d1(int(nxt[r_, q])).strip()
            if s and s[0].isdigit():
                digit_mask[r_, q] = True

    src_pos = (margin_all > 0.05).nonzero()
    tgt_pos = (margin_all < -0.05).nonzero()
    print(f'source positions: {src_pos.shape[0]}, target positions: '
          f'{tgt_pos.shape[0]}', flush=True)
    if min(len(src_pos), len(tgt_pos)) < 30:
        json.dump({'void': 'too few positions'}, open(OUT, 'w'), indent=1)
        return

    g = np.random.default_rng(9)
    pairs = [(tuple(src_pos[g.integers(len(src_pos))].tolist()),
              tuple(tgt_pos[g.integers(len(tgt_pos))].tolist()))
             for _ in range(NPAIRS)]
    same_class_pairs = [(tuple(tgt_pos[g.integers(len(tgt_pos))].tolist()),
                         tuple(tgt_pos[g.integers(len(tgt_pos))].tolist()))
                        for _ in range(NPAIRS)]

    def patched_margin(tgt_rq, patch_units, src_h_vec):
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
        mg = (p[..., TOK_A] + p[..., TOK_AN] - p[..., TOK_THE] - p[..., TOK_THE2])
        return float(mg[0, tq])

    ident_deltas = []
    for s_rq, t_rq in pairs[:10]:
        base = float(margin_all[t_rq])
        own_h = h_all[t_rq[0], t_rq[1]].to(DEV)
        pm = patched_margin(t_rq, the_cluster, own_h)
        ident_deltas.append(pm - base)
    ident = float(np.mean(np.abs(ident_deltas)))
    p0 = ident < 1e-4
    print(f'(0b) identity mean |delta| {ident:.6f}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'identity patch failed', 'ident': ident},
                   open(OUT, 'w'), indent=1)
        return

    rng2 = np.random.default_rng(13)
    rand_units = rng2.choice(list(range(H_full)), size=len(the_cluster),
                             replace=False).tolist()
    a_deltas, a_rand_deltas = [], []
    for s_rq, t_rq in pairs:
        base = float(margin_all[t_rq])
        src_h = h_all[s_rq[0], s_rq[1]].to(DEV)
        pm_c = patched_margin(t_rq, the_cluster, src_h)
        pm_r = patched_margin(t_rq, rand_units, src_h)
        a_deltas.append(pm_c - base)
        a_rand_deltas.append(pm_r - base)
    a_mean = float(np.mean(a_deltas))
    a_rand_mean = float(np.mean(a_rand_deltas))
    pa = a_mean > 0
    pb = (a_mean >= 3.0 * a_rand_mean if a_rand_mean > 0
          else a_mean > a_rand_mean)
    print(f'(a) cluster patch mean delta {a_mean:+.5f} (sign correct): '
          f"{'HELD' if pa else 'FAILED'}", flush=True)
    print(f'(b) random-unit patch mean delta {a_rand_mean:+.5f}: '
          f"{'HELD' if pb else 'FAILED'}", flush=True)

    c_deltas = []
    for t1_rq, t2_rq in same_class_pairs:
        base = float(margin_all[t2_rq])
        src_h = h_all[t1_rq[0], t1_rq[1]].to(DEV)
        pm = patched_margin(t2_rq, the_cluster, src_h)
        c_deltas.append(pm - base)
    c_mean = float(np.mean(c_deltas))
    pc = a_mean >= 3.0 * abs(c_mean) if c_mean != 0 else a_mean > 0
    print(f'(c) same-class patch mean delta {c_mean:+.5f} vs cross-class '
          f'{a_mean:+.5f}: {"HELD" if pc else "FAILED"}', flush=True)

    mlp0_cluster8_delta = 0.00219  # 592
    print(f'(d) mlp1 effect {a_mean:+.5f} vs mlp0 cluster-8 effect '
          f'{mlp0_cluster8_delta:+.5f} (ratio {a_mean/mlp0_cluster8_delta:.2f}x)',
          flush=True)

    dig_pos = digit_mask.nonzero()
    null_deltas = []
    if len(dig_pos) >= 10:
        for _ in range(min(NPAIRS, len(dig_pos))):
            dr = tuple(dig_pos[g.integers(len(dig_pos))].tolist())
            sr = tuple(src_pos[g.integers(len(src_pos))].tolist())
            base = float(margin_all[dr])
            src_h = h_all[sr[0], sr[1]].to(DEV)
            pm = patched_margin(dr, the_cluster, src_h)
            null_deltas.append(pm - base)
    null_mean = float(np.mean(null_deltas)) if null_deltas else None
    null_ok = (null_mean is not None and abs(null_mean) < 0.3 * abs(a_mean))
    print(f'NULL: digit-position delta {null_mean} vs real {a_mean:+.5f}: '
          f"{'ok' if null_ok else 'CHECK'}", flush=True)

    out = {'repro_ok': bool(repro_ok), 'cluster_n': len(the_cluster),
           'n_source': int(src_pos.shape[0]), 'n_target': int(tgt_pos.shape[0]),
           'pred_0': bool(p0), 'identity_abs_delta': ident,
           'cluster_mean_delta': a_mean, 'random_mean_delta': a_rand_mean,
           'pred_a': bool(pa), 'pred_b': bool(pb),
           'same_class_mean_delta': c_mean, 'pred_c': bool(pc),
           'mlp0_cluster8_delta': mlp0_cluster8_delta,
           'redundancy_ratio': a_mean / mlp0_cluster8_delta,
           'null_mean_delta': null_mean, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
