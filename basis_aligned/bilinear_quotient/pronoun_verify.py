"""PRONOUN VERIFY -- causally test the pronoun cluster with a properly
POPULATED target class, after 605's number-word test was underpowered
(spelled numbers too rare, only 77 targets). Pronouns are common, so
this test is well-powered and completes the correct/reversed tally
with a fourth cluster.

600 found, in mlp0 ranks 300-600, a 54-unit cluster firing positive
before pronouns (he/they/you/us) and negative before the demonstrative
"This"/"this". 604 placed it mid-spectrum (attn0-fold correlation
0.795, mostly current-token). This gives it (1) the validated 592
patching causal test -- does it push toward pronouns -- a fourth data
point on the correct/reversed pattern (3 correct article-family, 2
reversed newline/aux so far), and (2) a current-token trigger census.

Margin = P(pronoun: he/they/you/we/she/it/I) - P(demonstrative:
This/this/that). SOURCE = pronoun-favoring positions, TARGET =
demonstrative-favoring positions. Both classes are common, so unlike
605 the margin has real positions to move.

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY + IDENTITY: reclustering mlp0 ranks 300-600
      reproduces 600's sizes; the pronoun cluster is the 2nd largest
      (54 units); self-patch changes nothing -- VOIDS on failure;
  (a) CAUSAL SIGN (the tally): patching a pronoun-favoring source into
      a demonstrative target moves the margin toward pronouns
      (delta > 0 = correct-signed) or away (< 0 = reversed). No
      pre-set pass -- the sign is the data point;
  (b) SPECIFICITY: the cluster's patch effect exceeds a size-matched
      random-unit patch (|ratio| >= 3) -- real and specific whichever
      sign it has. This is the bar 605 failed for lack of power;
  (c) CURRENT-TOKEN TRIGGERS (no bar): at pronoun-target positions,
      report which CURRENT tokens have the highest cluster firing;
  NULL: the same source patch at an unrelated class (article-target
      positions) produces a shift under 30% of the pronoun-target
      effect -- pronoun-specific."""
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
OUT = PT + 'pronoun_verify_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
NPAIRS = 80
NUM_TOK = [339, 484, 345, 356, 673, 340, 314]  # he they you we she it I (pronouns)
CTRL_TOK = [770, 428, 326]  # This this that (demonstratives)
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


def recover_number_cluster(fresh):
    H, Dw = capture(fresh)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(H.shape[0], generator=g)[:NSAMP]
    Hs = H[perm]
    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    topk = order[TOPK:2 * TOPK].numpy()
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
    ranked = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
    return ranked[1][1], sizes == expect, sizes  # 2nd largest = pronoun (54)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    units, repro, sizes = recover_number_cluster(fresh)
    print(f'(0a) reproduced {sizes}: {"HELD" if repro else "FAILED"} | '
          f'number cluster n={len(units)}', flush=True)
    if not repro or len(units) != 54:
        json.dump({'void': 'reclustering mismatch', 'sizes': sizes,
                   'n': len(units)}, open(OUT, 'w'), indent=1)
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
        margin = sum(p[..., t] for t in NUM_TOK) - sum(p[..., t] for t in CTRL_TOK)
        return h, margin

    idx_all = fresh[:, :256].to(DEV)
    cur = fresh[:, :256]
    nxt = fresh[:, 1:257]
    N_docs = fresh.shape[0]
    h_all, margin_all = forward_h_and_margin(idx_all)
    h_all = h_all.reshape(N_docs, T, H_full)
    margin_all = margin_all.reshape(N_docs, T)

    art_mask = torch.zeros(N_docs, T, dtype=torch.bool)
    for t in (TOK_A, TOK_AN, TOK_THE, TOK_THE2):
        art_mask |= (nxt == t)

    src_pos = (margin_all > 0.02).nonzero()
    tgt_pos = (margin_all < -0.02).nonzero()
    print(f'source (number-favoring): {src_pos.shape[0]}, target: '
          f'{tgt_pos.shape[0]}', flush=True)
    if min(len(src_pos), len(tgt_pos)) < 30:
        json.dump({'void': 'too few positions',
                   'n_src': int(src_pos.shape[0]),
                   'n_tgt': int(tgt_pos.shape[0])}, open(OUT, 'w'), indent=1)
        return

    g = np.random.default_rng(9)
    pairs = [(tuple(src_pos[g.integers(len(src_pos))].tolist()),
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
        mg = sum(p[..., t] for t in NUM_TOK) - sum(p[..., t] for t in CTRL_TOK)
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
    sign = 'correct-signed (toward numbers)' if a_mean > 0 else 'REVERSED'
    pb = abs(a_mean) >= 3 * abs(a_rand_mean) if a_rand_mean != 0 else abs(a_mean) > abs(a_rand_mean)
    print(f'(a) number-cluster patch mean delta {a_mean:+.5f} -> {sign}',
          flush=True)
    print(f'(b) random-unit patch {a_rand_mean:+.5f} '
          f'(|ratio| {abs(a_mean)/max(abs(a_rand_mean),1e-9):.2f}): '
          f"{'HELD' if pb else 'FAILED'}", flush=True)

    # (c) current-token triggers: at number-target positions, which
    # CURRENT tokens have highest cluster firing
    num_tgt = torch.zeros(N_docs, T, dtype=torch.bool)
    for t in NUM_TOK:
        num_tgt |= (nxt == t)
    signed = h_all[:, :, units].sum(-1)  # per-position firing
    tok_fire = {}
    idxs = num_tgt.reshape(-1).nonzero().squeeze(1)
    curf = cur.reshape(-1)
    sfl = signed.reshape(-1)
    for gi in idxs.tolist():
        t = int(curf[gi])
        tok_fire.setdefault(t, []).append(float(sfl[gi]))
    tok_mean = {t: (float(np.mean(v)), len(v)) for t, v in tok_fire.items()
                if len(v) >= 4}
    top_trig = sorted(tok_mean, key=lambda t: -tok_mean[t][0])[:12]
    print('(c) top current-token triggers at number positions:', flush=True)
    for t in top_trig:
        print(f'    {cl.d1(t)!r}: {tok_mean[t][0]:+.1f} (n={tok_mean[t][1]})',
              flush=True)

    # NULL: same source patch at article-target positions
    art_pos = art_mask.nonzero()
    null_deltas = []
    if len(art_pos) >= 10:
        for _ in range(min(NPAIRS, len(art_pos))):
            ar = tuple(art_pos[g.integers(len(art_pos))].tolist())
            sr = tuple(src_pos[g.integers(len(src_pos))].tolist())
            base = float(margin_all[ar])
            src_h = h_all[sr[0], sr[1]].to(DEV)
            null_deltas.append(patched_margin(ar, units, src_h) - base)
    null_mean = float(np.mean(null_deltas)) if null_deltas else None
    null_ok = (null_mean is not None and abs(null_mean) < 0.3 * abs(a_mean))
    print(f'NULL: article-position delta {null_mean} vs number {a_mean:+.5f}: '
          f"{'ok' if null_ok else 'CHECK'}", flush=True)

    out = {'repro': bool(repro), 'n_units': len(units),
           'n_source': int(src_pos.shape[0]), 'n_target': int(tgt_pos.shape[0]),
           'pred_0': bool(p0), 'identity_abs_delta': ident,
           'cluster_mean_delta': a_mean, 'random_mean_delta': a_rand_mean,
           'causal_sign': 'correct' if a_mean > 0 else 'reversed',
           'pred_b': bool(pb),
           'top_triggers': [(cl.d1(t), tok_mean[t][0], tok_mean[t][1])
                            for t in top_trig],
           'null_mean_delta': null_mean, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
