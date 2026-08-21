"""CLUSTER ABLATION COVARIANCE (the user's actual proposed method, never
run here -- prior clustering was on output DIRECTIONS, not ablation
damage). Procedure:
  1. A-SVD of mlp1.Down -> top-M output components (A[:,k], B[k,:]).
  2. For T random small subsets S (|S|=s) of the M components, ABLATE S
     (remove those rank-1 terms) and record, PER DATAPOINT (token):
       - MSE damage = ||removed MLP-output contribution||^2  (no forward,
         cheap -- rank-s: (G@B[S].T)@A[:,S].T);
       - CE damage = per-token CE(ablated Down) - CE(full)  (forward).
  3. This gives a loss-by-datapoint matrix L (T x N). COVARIANCE across
     datapoints (how tokens co-vary in damage across trials) is the
     clustering signal. Cluster datapoints by their loss-response vectors.
  4. CONVERGENCE: as T grows, does the datapoint-similarity (covariance)
     matrix stabilize? Metric = split-half Frobenius correlation of the
     similarity matrix vs T. Plot it. Run enough datapoints (N) and trials
     (T=pairs) to plateau.
  5. Compare the ablation-covariance clustering to (a) the CE version, (b)
     the earlier output-direction k-means (704) via ARI -- does the user's
     method find the same structure?

REGISTERED PREDICTIONS:
  (0) SANITY: MSE-damage and CE-damage per datapoint are positively
      correlated (removing a token's output raises its CE);
  (a) CONVERGENCE: the split-half similarity-matrix correlation rises with
      T and plateaus >=0.9 by T<=1500 -- report the curve and the plateau T;
  (b) STRUCTURE: the ablation-covariance clustering agrees with the output-
      direction clustering above chance (ARI >= 0.15) -- the two methods
      find related token structure; report the ARI (and the clusters);
  NULL: shuffling each datapoint's loss vector independently destroys the
      covariance block structure (split-half corr ~ 0)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; HID = 4608; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cluster_ablation_covariance_results.json'
NFIT = 128          # 32k tokens for the A-SVD fit
NDATA = 8           # 2048 datapoints (tokens) for the loss matrix
M = 64              # top A-SVD components considered
S = 4               # subset size ablated per trial
T_MSE = 1500        # MSE trials (cheap)
T_CE = 120          # CE trials (forward)
KCL = 8


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


def kmeans(Xn, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = Xn[a == j].mean(0)
    return a


def ari(a, b):
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    cont = {}
    for i in range(n): cont[(a[i], b[i])] = cont.get((a[i], b[i]), 0) + 1
    from collections import Counter as C2
    ca, cb = C2(a.tolist()), C2(b.tolist())
    sc = sum(v*(v-1)/2 for v in cont.values()); sa = sum(v*(v-1)/2 for v in ca.values())
    sb = sum(v*(v-1)/2 for v in cb.values()); tot = n*(n-1)/2; exp = sa*sb/tot
    return (sc - exp)/(0.5*(sa+sb) - exp + 1e-12)


@torch.no_grad()
def capture(rows, n, both=False):
    cap = []; toks = []
    h = (m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
            lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID))))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0), torch.cat(toks, 0)


@torch.no_grad()
def per_tok_ce(rows, n, Wsub=None):
    mod = m.transformer.h[LAYER].mlp.Down; orig = mod.weight.data
    if Wsub is not None: mod.weight.data = Wsub.to(orig.dtype)
    out = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        out.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    if Wsub is not None: mod.weight.data = orig
    return torch.cat(out).numpy()


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    allrows = cl.fineweb_rows(NFIT + NDATA)
    fit, dat = allrows[:NFIT], allrows[NFIT:NFIT+NDATA]
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().to(DEV)
    Xfit, _ = capture(fit, NFIT)
    A, B = asvd_fast(W, Xfit)
    A = A[:, :M].contiguous(); B = B[:M, :].contiguous()   # top-M components

    G, toks = capture(dat, NDATA); tk = toks.numpy(); N = G.shape[0]
    GB = G @ B.T                                            # (N, M): datapoint x component coeff
    print(f'N datapoints {N}, M components {M}, S={S}', flush=True)

    rng = np.random.default_rng(0)
    # MSE damage matrix (T_MSE x N) -- cheap, rank-S removed contribution
    Lmse = np.zeros((T_MSE, N), dtype=np.float32)
    subsets = []
    for t in range(T_MSE):
        Ssub = rng.choice(M, size=S, replace=False); subsets.append(Ssub)
        removed = (GB[:, Ssub]) @ A[:, Ssub].T             # (N, D)
        Lmse[t] = (removed.float() ** 2).sum(1).cpu().numpy()
    print(f'MSE damage matrix done ({time.time()-t0:.0f}s)', flush=True)

    # CE damage matrix (T_CE x N) -- forward per trial
    ce_full = per_tok_ce(dat, NDATA)
    Lce = np.zeros((T_CE, N), dtype=np.float32)
    for t in range(T_CE):
        Ssub = subsets[t]; keep = np.setdiff1d(np.arange(M), Ssub)
        Wsub = A[:, keep] @ B[keep, :]
        Lce[t] = per_tok_ce(dat, NDATA, Wsub) - ce_full
    print(f'CE damage matrix done ({time.time()-t0:.0f}s)', flush=True)

    # (0) sanity: MSE vs CE damage per datapoint (mean over shared trials)
    corr_mse_ce = float(np.corrcoef(Lmse[:T_CE].mean(0), Lce.mean(0))[0, 1])

    # convergence: split-half similarity-matrix correlation vs T (MSE)
    def simmat(L):
        Lc = L - L.mean(0, keepdims=True)
        Ln = Lc / (np.linalg.norm(Lc, axis=0, keepdims=True) + 1e-9)
        return Ln.T @ Ln                                   # (N,N) datapoint correlation
    Tgrid = [50, 100, 200, 400, 700, 1000, 1500]
    conv = []
    for T in Tgrid:
        idx = rng.permutation(T_MSE)[:T]
        h = T // 2
        SA = simmat(Lmse[idx[:h]]); SB = simmat(Lmse[idx[h:2*h]])
        iu = np.triu_indices(N, 1)
        conv.append(round(float(np.corrcoef(SA[iu], SB[iu])[0, 1]), 4))
    print(f'convergence (split-half sim corr) vs T={Tgrid}: {conv}', flush=True)

    # null: shuffle each datapoint's loss vector independently
    Lsh = Lmse.copy()
    for i in range(N): Lsh[:, i] = Lsh[rng.permutation(T_MSE), i]
    h = T_MSE // 2
    SA = simmat(Lsh[:h]); SB = simmat(Lsh[h:2*h]); iu = np.triu_indices(N, 1)
    null_conv = float(np.corrcoef(SA[iu], SB[iu])[0, 1])

    # cluster datapoints by loss-response (normalized columns), MSE & CE
    def cluster_by_loss(L):
        Lc = L - L.mean(0, keepdims=True)
        Ln = torch.tensor((Lc / (np.linalg.norm(Lc, axis=0, keepdims=True) + 1e-9)).T)  # (N,T)
        return kmeans(Ln, KCL).numpy()
    cl_mse = cluster_by_loss(Lmse); cl_ce = cluster_by_loss(Lce)
    # output-direction clustering (704 method) for comparison
    On = (G @ W.T); On = (On / On.norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu()
    cl_dir = kmeans(On, KCL).numpy()
    ari_mse_ce = ari(cl_mse, cl_ce); ari_mse_dir = ari(cl_mse, cl_dir)

    print(f'\n(0) MSE-CE damage corr {corr_mse_ce:.3f}', flush=True)
    print(f'ARI(MSE-cluster, CE-cluster) {ari_mse_ce:.3f}', flush=True)
    print(f'ARI(MSE-cluster, output-dir cluster) {ari_mse_dir:.3f}', flush=True)
    print(f'NULL shuffled split-half corr {null_conv:.3f}', flush=True)
    for j in range(KCL):
        ex = [d1(t) for t, _ in Counter(tk[cl_mse == j].tolist()).most_common(5)]
        print(f'  MSE-cluster {j}: n={int((cl_mse==j).sum()):4d}  {ex}', flush=True)

    # convergence + covariance plots
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sys.path.insert(0, '/workspace/tensor_language')
    from palette import INK, SECONDARY, MUTED, GRID, SURFACE, BLUES
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)); fig.patch.set_facecolor(SURFACE)
    ax1.set_facecolor(SURFACE)
    ax1.plot(Tgrid, conv, 'o-', color='#3987e5', lw=2, label='ablation-covariance (real)')
    ax1.axhline(null_conv, color='#e34948', ls='--', lw=1.3, label=f'shuffled null ({null_conv:.2f})')
    ax1.axhline(0.9, color=MUTED, ls=':', lw=1)
    ax1.set_xlabel('number of ablation trials T'); ax1.set_ylabel('split-half similarity-matrix correlation')
    ax1.set_title('Convergence of the datapoint covariance', color=INK, loc='left', fontsize=12)
    ax1.set_ylim(-0.05, 1.02); ax1.legend(fontsize=9, frameon=False); ax1.grid(True, color=GRID, lw=0.6)
    for s in ['top', 'right']: ax1.spines[s].set_visible(False)
    # covariance heatmap ordered by MSE-cluster (subsample 512)
    order = np.argsort(cl_mse)[::max(1, N//512)]
    Scov = simmat(Lmse)[np.ix_(order, order)]
    ax2.set_facecolor(SURFACE)
    im = ax2.imshow(Scov, cmap=BLUES, vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('datapoint ablation-covariance,\nordered by cluster', color=INK, loc='left', fontsize=12)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)
    fig.tight_layout(); fig.savefig(PT + 'cluster_ablation_covariance.png', dpi=150, facecolor=SURFACE)
    print('wrote cluster_ablation_covariance.png', flush=True)

    p0 = corr_mse_ce > 0.2
    pa = max(conv) >= 0.9
    pb = ari_mse_dir >= 0.15
    null_ok = null_conv < 0.2
    out = {'mse_ce_corr': round(corr_mse_ce, 4), 'convergence_T': Tgrid, 'convergence': conv,
           'null_convergence': round(null_conv, 4), 'ari_mse_ce': round(ari_mse_ce, 4),
           'ari_mse_dir': round(ari_mse_dir, 4), 'N': int(N), 'M': M, 'S': S,
           'T_mse': T_MSE, 'T_ce': T_CE, 'pred_0': bool(p0), 'pred_a_converge': bool(pa),
           'pred_b_agree': bool(pb), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) converges>=0.9: {pa}; (b) agrees w/ output-dir (ARI>=0.15): {pb}; '
          f'(0) mse-ce corr>0.2: {p0}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
