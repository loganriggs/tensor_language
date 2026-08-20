"""NEWLINE CLUSTER PATCHING -- promote a newly-discovered cluster to a
causally-confirmed circuit, with the validated patching method (592).

600 found, in mlp0's units ranked 300-600 (below the usual cutoff), a
clean 81-unit cluster that fires positive before newline tokens and
negative before sentence-final period/quote -- a NEWLINE-vs-sentence-
end discriminator, structurally like the article cluster (cluster 8)
but for line breaks. It was found by clustering only; this gives it
the same causal test that confirmed cluster 8 (592): activation
patching, replace not delete.

Method (mirrors 592 exactly, retargeted to newline): margin =
P(newline: '\n','\n\n') - P(sentence-end: '.','"'). SOURCE = real
positions where the model favors a newline (margin > 0.05); TARGET =
real positions where it favors a sentence-end (margin < -0.05). Patch
the newline cluster's 81 units from a real newline-favoring source
into a real sentence-end-favoring target (everything else that
target's own forward pass computed left exact) and measure the margin
shift toward newline.

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY + IDENTITY: reclustering mlp0 ranks 300-600
      reproduces 600's sizes exactly and the newline cluster is the
      largest (81 units); patching a target's own activation into
      itself changes nothing (< 1e-4). Both VOID on failure;
  (a) CORRECT SIGN: patching a newline-favoring source into a
      sentence-end target moves the margin toward newline (delta > 0)
      -- the primary bar (592's lesson: sign, not a fixed magnitude);
  (b) SPECIFICITY vs RANDOM: the newline cluster's patch effect is
      >= 3x a size-matched random-unit patch;
  (c) SPECIFICITY vs SOURCE CLASS: a same-class (sentence-end into
      sentence-end) control patch produces a much smaller shift than
      the real cross-class patch;
  NULL: the same newline-favoring-source patch applied at digit-
      target positions produces a shift under 30% of the real-target
      effect -- the effect is newline-specific."""
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
OUT = PT + 'newline_cluster_patching_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
NL1, NL2, PER, QUO = 198, 628, 13, 1  # '\n', '\n\n', '.', '"'
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


def recover_newline_cluster(fresh):
    H, Dw = capture(fresh)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(H.shape[0], generator=g)[:NSAMP]
    Hs = H[perm]
    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    topk = order[TOPK:2 * TOPK].numpy()   # ranks 300-600 (matches 600)
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
    expect = [81, 54, 37, 31, 22, 12, 10, 8, 7, 6, 6, 5, 4, 4, 4, 3, 3, 1, 1, 1]
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    nl = max(by_cluster, key=lambda k: len(by_cluster[k]))
    return by_cluster[nl], sizes == expect, sizes


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    units, repro, sizes = recover_newline_cluster(fresh)
    print(f'(0a) reproduced sizes {sizes}: '
          f"{'HELD' if repro else 'FAILED'} | newline cluster n={len(units)}",
          flush=True)
    if not repro:
        json.dump({'void': 'reclustering mismatch', 'sizes': sizes},
                   open(OUT, 'w'), indent=1)
        return

    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight
    b = mlp.Down_bias
    H_full = Dw.shape[1]

    @torch.no_grad()
    def forward_h_and_margin(idx):
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
        margin = p[..., NL1] + p[..., NL2] - p[..., PER] - p[..., QUO]
        return h, margin

    idx_all = fresh[:, :256].to(DEV)
    nxt = fresh[:, 1:257]
    N_docs = fresh.shape[0]
    h_all, margin_all = forward_h_and_margin(idx_all)
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
    print(f'source (newline-favoring): {src_pos.shape[0]}, target '
          f'(sentence-end-favoring): {tgt_pos.shape[0]}', flush=True)
    if min(len(src_pos), len(tgt_pos)) < 30:
        json.dump({'void': 'too few positions',
                   'n_src': int(src_pos.shape[0]),
                   'n_tgt': int(tgt_pos.shape[0])}, open(OUT, 'w'), indent=1)
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
        mg = p[..., NL1] + p[..., NL2] - p[..., PER] - p[..., QUO]
        return float(mg[0, tq])

    ident_deltas = []
    for s_rq, t_rq in pairs[:10]:
        base = float(margin_all[t_rq])
        own_h = h_all[t_rq[0], t_rq[1]].to(DEV)
        ident_deltas.append(patched_margin(t_rq, units, own_h) - base)
    ident = float(np.mean(np.abs(ident_deltas)))
    p0 = ident < 1e-4
    print(f'(0b) identity mean |delta| {ident:.6f}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'identity failed', 'ident': ident},
                   open(OUT, 'w'), indent=1)
        return

    rng2 = np.random.default_rng(13)
    rand_units = rng2.choice(list(range(H_full)), size=len(units),
                             replace=False).tolist()
    a_deltas, a_rand_deltas = [], []
    for s_rq, t_rq in pairs:
        base = float(margin_all[t_rq])
        src_h = h_all[s_rq[0], s_rq[1]].to(DEV)
        a_deltas.append(patched_margin(t_rq, units, src_h) - base)
        a_rand_deltas.append(patched_margin(t_rq, rand_units, src_h) - base)
    a_mean = float(np.mean(a_deltas))
    a_rand_mean = float(np.mean(a_rand_deltas))
    pa = a_mean > 0
    pb = (a_mean >= 3.0 * a_rand_mean if a_rand_mean > 0
          else a_mean > a_rand_mean)
    print(f'(a) newline-cluster patch mean delta {a_mean:+.5f} (sign): '
          f"{'HELD' if pa else 'FAILED'}", flush=True)
    print(f'(b) random-unit patch mean delta {a_rand_mean:+.5f} '
          f'(ratio {a_mean/max(a_rand_mean,1e-9):.2f}): '
          f"{'HELD' if pb else 'FAILED'}", flush=True)

    c_deltas = []
    for t1_rq, t2_rq in same_class_pairs:
        base = float(margin_all[t2_rq])
        src_h = h_all[t1_rq[0], t1_rq[1]].to(DEV)
        c_deltas.append(patched_margin(t2_rq, units, src_h) - base)
    c_mean = float(np.mean(c_deltas))
    pc = a_mean >= 3.0 * abs(c_mean) if c_mean != 0 else a_mean > 0
    print(f'(c) same-class patch mean delta {c_mean:+.5f} vs cross-class '
          f'{a_mean:+.5f}: {"HELD" if pc else "FAILED"}', flush=True)

    dig_pos = digit_mask.nonzero()
    null_deltas = []
    if len(dig_pos) >= 10:
        for _ in range(min(NPAIRS, len(dig_pos))):
            dr = tuple(dig_pos[g.integers(len(dig_pos))].tolist())
            sr = tuple(src_pos[g.integers(len(src_pos))].tolist())
            base = float(margin_all[dr])
            src_h = h_all[sr[0], sr[1]].to(DEV)
            null_deltas.append(patched_margin(dr, units, src_h) - base)
    null_mean = float(np.mean(null_deltas)) if null_deltas else None
    null_ok = (null_mean is not None and abs(null_mean) < 0.3 * abs(a_mean))
    print(f'NULL: digit-position delta {null_mean} vs real {a_mean:+.5f}: '
          f"{'ok' if null_ok else 'CHECK'}", flush=True)

    out = {'repro': bool(repro), 'n_units': len(units),
           'n_source': int(src_pos.shape[0]), 'n_target': int(tgt_pos.shape[0]),
           'pred_0': bool(p0), 'identity_abs_delta': ident,
           'cluster_mean_delta': a_mean, 'random_mean_delta': a_rand_mean,
           'pred_a': bool(pa), 'pred_b': bool(pb),
           'same_class_mean_delta': c_mean, 'pred_c': bool(pc),
           'null_mean_delta': null_mean, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
