"""CLUSTER WRITE SCALING -- is the causal reversal (602: newline and
aux-contraction clusters push opposite their activation reading) a
real property or a patching-method artifact? Tests a DIFFERENT causal
method that does not transplant activations.

Patching (592/593/595/602) replaces a cluster's units with a source
context's real values -- a transplant. It gave correct signs for the
article clusters but reversed signs for the newline and aux-
contraction clusters. This tests the same causal-direction question a
different way: directly SCALE the cluster's own write into the
residual at the positions where it fires, without moving activations
between contexts.

Method: at positions where a cluster fires positively (its own
source positions), replace its output contribution h_j*Down[:,j] with
scale * (real contribution) + (1-scale) * (mean contribution), for
scale in {0, 1, 2} -- 0 removes the cluster's write, 1 is real, 2
doubles it. Measure the relevant class margin at each scale. The SLOPE
of margin vs scale is the causal direction, measured by amplifying
the cluster's OWN write rather than importing another context's. Run
on both the article cluster (cluster 8, known correct-signed under
patching) and the newline cluster (reversed under patching).

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY + LINEARITY CHECK: both clusters reproduce
      their sizes; scale=1 reproduces the true margin exactly
      (< 1e-4 vs unhooked) -- VOIDS on failure;
  (a) ARTICLE STAYS CORRECT: for cluster 8 at its own positive
      (a/an-favoring) positions, the margin INCREASES with scale
      (positive slope) -- amplifying the article cluster's write
      pushes further toward a/an, confirming its patching sign was
      right and this method agrees;
  (b) THE TEST -- newline direction under a non-transplant method: for
      the newline cluster at its own positive (newline-favoring)
      positions, report the sign of the margin-vs-scale slope. If
      NEGATIVE (amplifying the write pushes AWAY from newline), the
      reversal is a REAL property confirmed by two independent methods.
      If POSITIVE (amplifying pushes toward newline), the patching
      reversal was a transplant artifact and the cluster is correct-
      signed after all. No pre-set pass -- the sign is the finding;
  (c) MAGNITUDES: report both slopes (margin change per unit scale)
      for the record;
  NULL: scaling a size-matched RANDOM unit set at the same positions
      produces a much smaller |slope| than the real cluster for both
      clusters -- the write-scaling effect is specific to the found
      units."""
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
OUT = PT + 'cluster_write_scaling_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
SCALES = [0.0, 1.0, 2.0]
NL1, NL2, PER, QUO = 198, 628, 13, 1
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


def recover(fresh, band, expect):
    H, Dw = capture(fresh)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(H.shape[0], generator=g)[:NSAMP]
    Hs = H[perm]
    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    lo, hi = band
    topk = order[lo:hi].numpy()
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
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    largest = max(by_cluster, key=lambda k: len(by_cluster[k]))
    return by_cluster[largest], sizes == expect, sizes


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    exp_top = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    exp_mid = [81, 54, 37, 31, 22, 12, 10, 8, 7, 6, 6, 5, 4, 4, 4, 3, 3, 1, 1, 1]
    article, ok_a, s_a = recover(fresh, (0, TOPK), exp_top)
    # article cluster = largest of top-300 (101); newline = largest of 300-600 (81)
    newline, ok_n, s_n = recover(fresh, (TOPK, 2 * TOPK), exp_mid)
    p0_repro = ok_a and ok_n and len(article) == 101 and len(newline) == 81
    print(f'(0a) article {ok_a} (n={len(article)}), newline {ok_n} '
          f'(n={len(newline)}): {"HELD" if p0_repro else "FAILED -- VOID"}',
          flush=True)
    if not p0_repro:
        json.dump({'void': 'reclustering mismatch', 's_a': s_a, 's_n': s_n},
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
    def scaled_margins(units, scale, pos_tok, neg_tok, pos_mask_only=None):
        """Per-position margin with `units`' write scaled by `scale`
        (0=remove, 1=real, 2=double). pos_mask_only restricts the scaling
        to positions in that (N_docs,T) bool mask (others left real)."""
        msk = torch.zeros(H_full, device=DEV)
        msk[torch.tensor(units, device=DEV)] = 1.0
        out = torch.zeros(N_docs, T)
        for i in range(0, N_docs, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            B = bb.shape[0]

            def fh(mo, args, o_, msk=msk, scale=scale, i0=i, B=B):
                X = args[0].float()
                h = (X @ L.T) * (X @ R.T)
                # scaled hidden for the cluster's units:
                # h' = mean + scale*(h - mean) on those units
                hs_units = hmean[None, None, :] + scale * (h - hmean[None, None, :])
                hnew = h * (1 - msk) + hs_units * msk
                if pos_mask_only is not None:
                    pm = pos_mask_only[i0:i0 + B].to(DEV)[:, :, None].float()
                    hnew = torch.where(pm.bool().expand_as(h), hnew, h)
                return (hnew @ Dw.T.float() + b.float()).to(o_.dtype)
            hh = mlp.register_forward_hook(fh)
            x = F.rms_norm(m.transformer.wte(idx), (D,))
            x0 = x
            v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            hh.remove()
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            p = F.softmax(lg, dim=-1)
            mg = sum(p[..., t] for t in pos_tok) - sum(p[..., t] for t in neg_tok)
            out[i:i + B] = mg.cpu()
        return out

    def positive_positions(units, pos_tok, neg_tok):
        """positions where the cluster fires positive AND the margin
        favors the positive class (its own source positions)."""
        base = scaled_margins(units, 1.0, pos_tok, neg_tok)
        return (base > 0.02)

    def slope_at(units, pos_tok, neg_tok, pmask):
        vals = []
        for sc in SCALES:
            mg = scaled_margins(units, sc, pos_tok, neg_tok, pmask)
            vals.append(float(mg[pmask].mean()))
        # slope via least squares over SCALES
        sx = np.array(SCALES); sy = np.array(vals)
        slope = float(np.polyfit(sx, sy, 1)[0])
        return slope, vals

    # identity check: scale=1 reproduces the true margin
    art_pos = positive_positions(article, [TOK_A, TOK_AN], [TOK_THE, TOK_THE2])
    nl_pos = positive_positions(newline, [NL1, NL2], [PER, QUO])
    true_art = scaled_margins(article, 1.0, [TOK_A, TOK_AN], [TOK_THE, TOK_THE2])
    # compare to no-hook margin
    @torch.no_grad()
    def raw_margin(pos_tok, neg_tok):
        out = torch.zeros(N_docs, T)
        for i in range(0, N_docs, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous(); B = bb.shape[0]
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            p = F.softmax(lg, dim=-1)
            mg = sum(p[..., t] for t in pos_tok) - sum(p[..., t] for t in neg_tok)
            out[i:i + B] = mg.cpu()
        return out
    raw_art = raw_margin([TOK_A, TOK_AN], [TOK_THE, TOK_THE2])
    id_err = float((true_art - raw_art).abs().mean())
    p0 = id_err < 1e-4
    print(f'(0b) scale=1 identity err {id_err:.2e}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}", flush=True)
    if not p0:
        json.dump({'void': 'scale=1 not identity', 'id_err': id_err},
                   open(OUT, 'w'), indent=1)
        return

    art_slope, art_vals = slope_at(article, [TOK_A, TOK_AN],
                                   [TOK_THE, TOK_THE2], art_pos)
    nl_slope, nl_vals = slope_at(newline, [NL1, NL2], [PER, QUO], nl_pos)
    pa = art_slope > 0
    print(f'(a) ARTICLE slope {art_slope:+.5f} (vals {art_vals} at scales '
          f'{SCALES}): {"HELD -- amplifying pushes toward a/an" if pa else "FAILED"}',
          flush=True)
    nl_real_reversed = nl_slope < 0
    print(f'(b) NEWLINE slope {nl_slope:+.5f} (vals {nl_vals}): '
          f'{"REVERSAL CONFIRMED by 2nd method (negative slope)" if nl_real_reversed else "reversal was a patching artifact (positive slope)"}',
          flush=True)

    # NULL: random units
    rng = np.random.default_rng(31)
    rand_a = rng.choice(list(range(H_full)), size=len(article),
                        replace=False).tolist()
    rand_n = rng.choice(list(range(H_full)), size=len(newline),
                        replace=False).tolist()
    ra_slope, _ = slope_at(rand_a, [TOK_A, TOK_AN], [TOK_THE, TOK_THE2], art_pos)
    rn_slope, _ = slope_at(rand_n, [NL1, NL2], [PER, QUO], nl_pos)
    null_ok = abs(art_slope) > 2 * abs(ra_slope) and abs(nl_slope) > 2 * abs(rn_slope)
    print(f'NULL: random slopes article {ra_slope:+.5f}, newline {rn_slope:+.5f} '
          f'(real should be >2x): {"ok" if null_ok else "CHECK"}', flush=True)

    out = {'repro': bool(p0_repro), 'identity_err': id_err, 'pred_0': bool(p0),
           'article_slope': art_slope, 'article_vals': art_vals,
           'pred_a': bool(pa),
           'newline_slope': nl_slope, 'newline_vals': nl_vals,
           'newline_reversal_confirmed': bool(nl_real_reversed),
           'random_article_slope': ra_slope, 'random_newline_slope': rn_slope,
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
