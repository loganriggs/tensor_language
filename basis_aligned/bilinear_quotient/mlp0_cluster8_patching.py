"""MLP0 CLUSTER8 PATCHING -- the genuinely different causal probe 586
said this thread needed, after three margin/mean-ablation designs
(582-586) all gave metric-unstable, directionally-inconsistent
verdicts for cluster 8 (and the other two named clusters).

Mean-ablation (582/583/585) replaces a cluster's units with their
corpus-wide AVERAGE everywhere -- a blunt intervention that removes
information but replaces it with an arbitrary constant, and 586 found
the resulting margin shifts flip sign depending on which reasonable
scoring function is used. ACTIVATION PATCHING is the standard
alternative for exactly this kind of claim: instead of deleting
information, SWAP IN a real activation from a different, real context
-- if cluster 8 really carries "this context wants a/an", patching a
real a/an-favoring context's cluster-8 activation into a real
the-favoring context's forward pass (same downstream machinery,
different history) should shift that target position's prediction
toward a/an, and patching a the-favoring source into a the-favoring
target should not.

Method: find real positions where the model's own next-token
prediction strongly favors indefinite articles (SOURCE, margin
P(a/an)-P(the/The) > 0) and real positions where it strongly favors
definite articles (TARGET, margin < 0), reusing the same real
fineweb corpus and the same cluster-8 unit set 579/581 found. For
each TARGET position, run its own sequence normally up to mlp0, but
at that single position overwrite cluster 8's 101 hidden-unit values
with a SOURCE position's real values (captured from the source's own
independent forward pass, same units, same relative slot), leave
every other position and every other unit exactly as the target's own
forward pass computed it, and measure the shift in the probability
margin at that target position.

REGISTERED PREDICTIONS:
  (0) IDENTITY: patching a target's OWN cluster-8 activation into
      itself changes nothing (< 1e-4 margin shift) -- sanity, VOIDS
      on failure;
  (a) PATCHING SHIFTS TOWARD THE SOURCE: patching a real a/an-
      favoring source into a the-favoring target increases the
      target's margin (moves toward a/an) by a clear amount;
  (b) SPECIFICITY, cluster vs random: the same patch restricted to a
      random size-matched unit set produces a shift at least 3x
      smaller than cluster 8's -- the effect is about THESE units,
      not "patching anything from a different context helps";
  (c) SPECIFICITY, source class: patching a the-favoring source
      (matched, real, different sequence) into a the-favoring target
      produces a much smaller shift than an a/an-favoring source does
      -- the shift is about the SOURCE'S content, not just "patching
      disturbs the residual stream";
  NULL: the same a/an-favoring-source patch, applied at a digit-
      prediction TARGET position instead of a the-favoring one (same
      margin measured, now irrelevant to the actual task there),
      produces a shift under 30% the size of (a)'s real-target
      effect -- the effect is about genuine article-choice positions,
      not a generic consequence of patching these units anywhere."""
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
OUT = PT + 'mlp0_cluster8_patching_results.json'
NFRESH = 64  # must match every other cluster8 script -- NSAMP subsampling
             # with a fixed seed depends on the total pool size (NFRESH*256),
             # so changing this silently breaks exact cluster reproduction
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


def recover_cluster8(fresh):
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
    c8 = max(by_cluster, key=lambda k: len(by_cluster[k]))
    return by_cluster[c8], sizes == expect, sizes


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
    print(f'source (a/an-favoring) positions: {src_pos.shape[0]}, '
          f'target (the-favoring) positions: {tgt_pos.shape[0]}', flush=True)
    n0 = min(len(src_pos), len(tgt_pos))
    p0v = n0 >= 30
    if not p0v:
        json.dump({'void': 'too few source/target positions', 'n0': n0},
                   open(OUT, 'w'), indent=1)
        return

    g = np.random.default_rng(9)
    pairs = []
    for _ in range(NPAIRS):
        s = src_pos[g.integers(len(src_pos))].tolist()
        t = tgt_pos[g.integers(len(tgt_pos))].tolist()
        pairs.append((tuple(s), tuple(t)))
    same_class_pairs = []
    for _ in range(NPAIRS):
        t1 = tgt_pos[g.integers(len(tgt_pos))].tolist()
        t2 = tgt_pos[g.integers(len(tgt_pos))].tolist()
        same_class_pairs.append((tuple(t1), tuple(t2)))

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

    # (0) identity: patch target's own h into itself
    ident_deltas = []
    for s_rq, t_rq in pairs[:10]:
        base = float(margin_all[t_rq])
        own_h = h_all[t_rq[0], t_rq[1]].to(DEV)
        pm = patched_margin(t_rq, cluster8, own_h)
        ident_deltas.append(pm - base)
    ident = float(np.mean(np.abs(ident_deltas)))
    p0 = ident < 1e-4
    print(f'(0) identity mean |delta| {ident:.6f}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'identity patch failed', 'ident': ident},
                   open(OUT, 'w'), indent=1)
        return

    rng2 = np.random.default_rng(13)
    rand_units = rng2.choice(list(range(H_full)), size=len(cluster8),
                             replace=False).tolist()

    a_deltas, a_rand_deltas = [], []
    for s_rq, t_rq in pairs:
        base = float(margin_all[t_rq])
        src_h = h_all[s_rq[0], s_rq[1]].to(DEV)
        pm_c = patched_margin(t_rq, cluster8, src_h)
        pm_r = patched_margin(t_rq, rand_units, src_h)
        a_deltas.append(pm_c - base)
        a_rand_deltas.append(pm_r - base)
    a_mean = float(np.mean(a_deltas))
    a_rand_mean = float(np.mean(a_rand_deltas))
    pa = a_mean > 0.01
    pb = a_rand_mean == 0 or a_mean >= 3.0 * a_rand_mean if a_rand_mean > 0 else a_mean > a_rand_mean
    print(f'(a) cluster8 patch mean delta {a_mean:+.5f}: '
          f"{'HELD' if pa else 'FAILED'}", flush=True)
    print(f'(b) random-unit patch mean delta {a_rand_mean:+.5f} '
          f'(cluster8/random ratio {a_mean/max(a_rand_mean,1e-9):.2f}): '
          f"{'HELD' if pb else 'FAILED'}", flush=True)

    c_deltas = []
    for t1_rq, t2_rq in same_class_pairs:
        base = float(margin_all[t2_rq])
        src_h = h_all[t1_rq[0], t1_rq[1]].to(DEV)
        pm = patched_margin(t2_rq, cluster8, src_h)
        c_deltas.append(pm - base)
    c_mean = float(np.mean(c_deltas))
    pc = a_mean >= 3.0 * abs(c_mean) if c_mean != 0 else a_mean > 0
    print(f'(c) same-class (the->the) patch mean delta {c_mean:+.5f} '
          f'vs cross-class {a_mean:+.5f}: '
          f"{'HELD' if pc else 'FAILED'}", flush=True)

    # NULL: same patch (a/an-favoring source into cluster 8), but the
    # TARGET is a digit-prediction position instead of a the-favoring
    # one -- reuses the same margin function (still measuring the
    # a/an-vs-the split, just at a position where that split is
    # irrelevant to the actual task); an article-specific effect
    # predicts this shift is much smaller than (a)'s real effect.
    dig_pos = digit_mask.nonzero()
    null_deltas = []
    if len(dig_pos) >= 10:
        for _ in range(min(NPAIRS, len(dig_pos))):
            dr = tuple(dig_pos[g.integers(len(dig_pos))].tolist())
            sr = tuple(src_pos[g.integers(len(src_pos))].tolist())
            base = float(margin_all[dr])
            src_h = h_all[sr[0], sr[1]].to(DEV)
            pm = patched_margin(dr, cluster8, src_h)
            null_deltas.append(pm - base)
    null_mean = float(np.mean(null_deltas)) if null_deltas else None
    null_ok = (null_mean is not None and abs(null_mean) < 0.3 * abs(a_mean))
    print(f'NULL: digit-position patch mean delta {null_mean} vs real-target '
          f'delta {a_mean:+.5f}: {"ok" if null_ok else "CHECK"}', flush=True)

    out = {'repro_ok': bool(repro_ok), 'cluster8_n': len(cluster8),
           'n_source': int(src_pos.shape[0]), 'n_target': int(tgt_pos.shape[0]),
           'pred_0': bool(p0), 'identity_abs_delta': ident,
           'cluster8_mean_delta': a_mean, 'random_mean_delta': a_rand_mean,
           'pred_a': bool(pa), 'pred_b': bool(pb),
           'same_class_mean_delta': c_mean, 'pred_c': bool(pc),
           'null_mean_delta': null_mean, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
