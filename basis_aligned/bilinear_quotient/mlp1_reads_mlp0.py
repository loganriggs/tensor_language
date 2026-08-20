"""MLP1 READS MLP0 -- is the article redundancy SERIAL (mlp1 reads
mlp0's write) or PARALLEL (mlp1 independently recomputes from the same
input)? Directly tests the user's "components fold into later layers"
idea.

595/597 established mlp1 has a real, causally-confirmed article
cluster that fires on the SAME positions as mlp0 cluster 8 (energy
correlation 0.47). Two mechanisms produce that co-firing:
  SERIAL: mlp0 cluster 8 writes the article signal into the residual
    stream; mlp1's cluster READS that write and continues it. Then
    ablating mlp0 cluster 8's write should DEGRADE mlp1's firing.
  PARALLEL: both clusters independently read the same upstream input
    (token embedding + attn0 bigram) and separately compute the same
    decision. Then ablating mlp0 cluster 8 leaves mlp1's own input
    (which still contains the embedding + attn0) intact, so mlp1
    fires unchanged.
This distinguishes them: mean-fill mlp0 cluster 8's 101 units (remove
its residual write, everything else exact) and measure whether mlp1's
article cluster still fires at article positions.

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY: both clusters reproduce exactly -- VOIDS on
      failure;
  (a) SERIAL-vs-PARALLEL (the finding): report the fractional change
      in mlp1-article-cluster firing energy at article positions when
      mlp0 cluster 8 is ablated. A drop >= 20% indicates SERIAL
      (mlp1 reads mlp0's write); a change < 5% indicates PARALLEL
      (independent recomputation); in between is partial. No hard
      pass/fail -- the fraction is the result;
  (b) SPECIFICITY: ablating mlp0 cluster 8 changes mlp1's article
      firing MORE than ablating a size-matched RANDOM mlp0 cluster
      does -- any serial dependence is on cluster 8 specifically, not
      a generic consequence of perturbing mlp0;
  (c) TARGET SPECIFICITY: mlp1's UNRELATED cluster (the tokenization-
      artifact detector) firing is affected LESS by the mlp0 cluster 8
      ablation than mlp1's article cluster is -- the dependence (if
      any) is article-specific;
  NULL / machinery check: the mlp0 cluster 8 ablation must actually
      reach the output -- verify it shifts the whole-model a/an-vs-the
      probability margin at article positions (a known-nonzero effect,
      592). If the margin doesn't move, the ablation isn't propagating
      and (a) would be measuring nothing."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_reads_mlp0_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383


@torch.no_grad()
def capture_h(fresh, LJ):
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


def recover(fresh, LJ, expect):
    H, Dw = capture_h(fresh, LJ)
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
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    ranked = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
    return ranked, sizes == expect


@torch.no_grad()
def firing_energies(fresh, mlp1_units_list, mlp0_ablate=None, mlp0_hmean=None):
    """Return per-position firing energy for each unit-set in
    mlp1_units_list (measured at mlp1), plus mlp0-cluster8's own energy.
    mlp0_ablate: list of mlp0 unit ids to mean-fill (or None)."""
    mlp0 = m.transformer.h[0].mlp
    mlp1 = m.transformer.h[1].mlp
    L1 = mlp1.Left.weight.float()
    R1 = mlp1.Right.weight.float()
    L0 = mlp0.Left.weight.float()
    R0 = mlp0.Right.weight.float()
    Dw0 = mlp0.Down.weight
    b0 = mlp0.Down_bias
    H0full = Dw0.shape[1]
    cap = {}
    hk1 = mlp1.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X1', a_[0]))
    cap0 = {}
    hk0pre = mlp0.register_forward_pre_hook(
        lambda mo_, a_: cap0.__setitem__('X0', a_[0]))
    abl_hook = None
    if mlp0_ablate is not None:
        msk = torch.zeros(H0full, device=DEV)
        msk[torch.tensor(mlp0_ablate, device=DEV)] = 1.0

        def fh0(mo, args, o_):
            X = args[0].float()
            h = (X @ L0.T) * (X @ R0.T)
            h = h * (1 - msk) + mlp0_hmean[None, None, :] * msk
            return (h @ Dw0.T.float() + b0.float()).to(o_.dtype)
        abl_hook = mlp0.register_forward_hook(fh0)

    e_mlp1 = [[] for _ in mlp1_units_list]
    e_mlp0c8 = []
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        X1 = cap['X1'].float()
        h1 = (X1 @ L1.T) * (X1 @ R1.T)
        for j, units in enumerate(mlp1_units_list):
            e_mlp1[j].append((h1[:, :, units] ** 2).sum(-1).reshape(-1).cpu())
        X0 = cap0['X0'].float()
        h0 = (X0 @ L0.T) * (X0 @ R0.T)
        e_mlp0c8.append(h0[:, :, MLP0_C8].pow(2).sum(-1).reshape(-1).cpu())
    hk1.remove()
    hk0pre.remove()
    if abl_hook is not None:
        abl_hook.remove()
    return ([torch.cat(e) for e in e_mlp1], torch.cat(e_mlp0c8))


MLP0_C8 = None


@torch.no_grad()
def article_margin(fresh, mlp0_ablate=None, mlp0_hmean=None):
    """Whole-model a/an-vs-the probability margin per position, optionally
    with mlp0 units mean-filled. Verifies an ablation reaches the output
    (known-nonzero for cluster 8, 592)."""
    mlp0 = m.transformer.h[0].mlp
    L0 = mlp0.Left.weight.float()
    R0 = mlp0.Right.weight.float()
    Dw0 = mlp0.Down.weight
    b0 = mlp0.Down_bias
    H0full = Dw0.shape[1]
    abl_hook = None
    if mlp0_ablate is not None:
        msk = torch.zeros(H0full, device=DEV)
        msk[torch.tensor(mlp0_ablate, device=DEV)] = 1.0

        def fh0(mo, args, o_):
            X = args[0].float()
            h = (X @ L0.T) * (X @ R0.T)
            h = h * (1 - msk) + mlp0_hmean[None, None, :] * msk
            return (h @ Dw0.T.float() + b0.float()).to(o_.dtype)
        abl_hook = mlp0.register_forward_hook(fh0)
    out = []
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        mg = (p[..., TOK_A] + p[..., TOK_AN] - p[..., TOK_THE] - p[..., TOK_THE2])
        out.append(mg.reshape(-1).cpu())
    if abl_hook is not None:
        abl_hook.remove()
    return torch.cat(out)


def main():
    global MLP0_C8
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    exp0 = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    exp1 = [86, 46, 42, 37, 12, 11, 11, 11, 10, 9, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1]
    r0, ok0 = recover(fresh, 0, exp0)
    r1, ok1 = recover(fresh, 1, exp1)
    p0 = ok0 and ok1
    print(f'(0) mlp0 {ok0}, mlp1 {ok1}: {"HELD" if p0 else "FAILED -- VOID"}',
          flush=True)
    if not p0:
        json.dump({'void': 'reclustering mismatch'}, open(OUT, 'w'), indent=1)
        return
    MLP0_C8 = r0[0][1]
    mlp1_article = r1[1][1]
    mlp1_artifact = r1[0][1]
    rng = np.random.default_rng(21)
    H0 = m.transformer.h[0].mlp.Down.weight.shape[1]
    mlp0_random = rng.choice(list(range(H0)), size=len(MLP0_C8),
                             replace=False).tolist()

    # mlp0 hidden mean for mean-fill
    H0cap, _ = capture_h(fresh, 0)
    mlp0_hmean = H0cap.mean(0).to(DEV)

    nxt = fresh[:, 1:257].reshape(-1)
    art = ((nxt == TOK_A) | (nxt == TOK_AN) |
           (nxt == TOK_THE) | (nxt == TOK_THE2)).numpy()
    amask = torch.tensor(art)

    units_list = [mlp1_article, mlp1_artifact]
    # baseline
    (base_e, base_c8) = firing_energies(fresh, units_list)
    # ablate mlp0 cluster 8
    (abl_e, abl_c8) = firing_energies(fresh, units_list,
                                      mlp0_ablate=MLP0_C8,
                                      mlp0_hmean=mlp0_hmean)
    # ablate random mlp0 cluster
    (rnd_e, _) = firing_energies(fresh, units_list,
                                 mlp0_ablate=mlp0_random,
                                 mlp0_hmean=mlp0_hmean)

    def art_mean(e):
        return float(e[amask].mean())

    b_art = art_mean(base_e[0])
    a_art = art_mean(abl_e[0])
    r_art = art_mean(rnd_e[0])
    b_artifact = art_mean(base_e[1])
    a_artifact = art_mean(abl_e[1])

    frac_c8 = (a_art - b_art) / b_art
    frac_rand = (r_art - b_art) / b_art
    frac_artifact = (a_artifact - b_artifact) / b_artifact
    verdict = ('SERIAL (mlp1 reads mlp0)' if abs(frac_c8) >= 0.20
               else 'PARALLEL (independent)' if abs(frac_c8) < 0.05
               else 'PARTIAL')
    print(f'(a) mlp1-article firing change under mlp0-c8 ablation: '
          f'{frac_c8:+.1%} -> {verdict}', flush=True)
    pb = abs(frac_c8) > abs(frac_rand)
    print(f'(b) vs random mlp0 cluster ablation {frac_rand:+.1%}: '
          f"{'HELD' if pb else 'FAILED'}", flush=True)
    pc = abs(frac_c8) > abs(frac_artifact)
    print(f'(c) mlp1-artifact firing change {frac_artifact:+.1%} (should be '
          f"smaller than article's): {'HELD' if pc else 'FAILED'}", flush=True)

    # NULL / machinery check: the mlp0-c8 ablation must actually reach the
    # output -- verify it shifts the whole-model article margin (592).
    amask_np = amask.numpy()
    base_mg = article_margin(fresh).numpy()[amask_np]
    abl_mg = article_margin(fresh, mlp0_ablate=MLP0_C8,
                            mlp0_hmean=mlp0_hmean).numpy()[amask_np]
    margin_shift = float(np.abs(abl_mg - base_mg).mean())
    null_ok = margin_shift > 1e-4
    print(f'NULL (mlp0-c8 ablation shifts article margin by {margin_shift:.5f} '
          f"-> ablation reaches output): {'ok' if null_ok else 'CHECK'}",
          flush=True)

    out = {'pred_0': bool(p0),
           'mlp1_article_base': b_art, 'mlp1_article_ablated': a_art,
           'mlp1_article_random_ablated': r_art,
           'frac_change_c8': frac_c8, 'frac_change_random': frac_rand,
           'frac_change_artifact': frac_artifact, 'verdict': verdict,
           'pred_b': bool(pb), 'pred_c': bool(pc),
           'margin_shift_under_ablation': margin_shift, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
