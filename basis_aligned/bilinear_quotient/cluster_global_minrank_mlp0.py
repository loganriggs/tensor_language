"""CLUSTER GLOBAL MINRANK MLP0 -- POSITIVE CONTROL on a genuinely LOW-rank
layer (mlp0, r80=8) vs mlp1's high-rank (720). If the shared-basis method
is valid, mlp0's clusters should need FEW distinct global components (r90
small, LESS Jaccard overlap, and the random-set NULL should FAIL -- which
components matters when few are needed). This confirms mlp1's high-rank is
real, not a method artifact.
ORIG DOC:
CLUSTER GLOBAL MINRANK (user's follow-up): with the ablation-covariance
clusters, find each cluster's MINIMAL RANK to run without loss, expressed in
the GLOBAL A-SVD basis (shared vocabulary across clusters), so we can say
"cluster A needs global components {..}, cluster B needs {..}" with real
overlap.

Steps:
  1. Global A-SVD of mlp1.Down -> M shared components (the shared language).
  2. Cluster datapoints by CE-ablation-covariance (loss-coherent; MSE and
     CE decouple per 719, and "without LOSS" = CE). Verify convergence.
  3. Component x cluster CE-importance: ablate each single global component
     k, measure per-cluster CE increase -> importance[k, c].
  4. Per cluster c: keep its top-r most-important global components, measure
     that cluster's CE recovery; minimal rank r90 = smallest r recovering
     >=90% of cluster c's own CE benefit ("without loss"). Report r90 + the
     component SET per cluster.
  5. SHARED LANGUAGE: pairwise Jaccard overlap of the component sets across
     clusters; the component x cluster importance heatmap.

REGISTERED PREDICTIONS:
  (0) SANITY: clustering converges (split-half cov corr >=0.8 at T used);
  (a) PER-CLUSTER MINRANK: report r90 per cluster. From 709 the fair per-
      cluster rank was high (~128) -- register the expectation that in the
      SHARED basis most clusters still need MANY global components (r90 not
      tiny), but SOME (e.g. punctuation) are low; report the spread;
  (b) SHARED LANGUAGE: the per-cluster component sets OVERLAP (mean pairwise
      Jaccard in (0.1, 0.9)) -- partial sharing, not identical, not disjoint;
  NULL: a RANDOM set of the same size recovers far less per-cluster CE than
      the importance-ranked set."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; HID = 4608; LAYER = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cluster_global_minrank_mlp0_results.json'
NFIT = 128; NDATA = 8; M = 96; S = 4; T_CE = 200; KCL = 8
RANKS = [1, 2, 4, 8, 16, 32, 48, 64, 96]


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


@torch.no_grad()
def capture(rows, n):
    cap = []; toks = []
    h = m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID)))
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
    if Wsub is not None: mod.weight.data = (torch.zeros_like(orig) if Wsub == 'ablate' else Wsub.to(orig.dtype))
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
    A, B = asvd_fast(W, Xfit); A = A[:, :M].contiguous(); B = B[:M, :].contiguous()

    G, toks = capture(dat, NDATA); tk = toks.numpy(); N = G.shape[0]
    GB = G @ B.T
    ce_full = per_tok_ce(dat, NDATA)

    # CE-damage matrix (T_CE x N) for clustering
    rng = np.random.default_rng(0)
    Lce = np.zeros((T_CE, N), dtype=np.float32)
    for t in range(T_CE):
        Ssub = rng.choice(M, size=S, replace=False); keep = np.setdiff1d(np.arange(M), Ssub)
        Lce[t] = per_tok_ce(dat, NDATA, A[:, keep] @ B[keep, :]) - ce_full
    print(f'CE-damage matrix done ({time.time()-t0:.0f}s)', flush=True)

    # convergence check
    def simmat(L):
        Lc = L - L.mean(0, keepdims=True); Ln = Lc/(np.linalg.norm(Lc, axis=0, keepdims=True)+1e-9)
        return Ln.T @ Ln
    h = T_CE//2; iu = np.triu_indices(N, 1)
    conv = float(np.corrcoef(simmat(Lce[:h])[iu], simmat(Lce[h:2*h])[iu])[0, 1])
    print(f'CE clustering convergence (split-half sim corr, T={T_CE}): {conv:.3f}', flush=True)

    # cluster by CE-damage covariance
    Lc = Lce - Lce.mean(0, keepdims=True); Ln = torch.tensor((Lc/(np.linalg.norm(Lc, axis=0, keepdims=True)+1e-9)).T)
    assign = kmeans(Ln, KCL).numpy()

    # component x cluster CE-importance (single-component ablation)
    imp = np.zeros((M, KCL), dtype=np.float32)
    cl_full = {c: ce_full[assign == c].mean() for c in range(KCL)}
    cl_abl = {c: per_tok_ce(dat, NDATA, 'ablate')[assign == c].mean() for c in range(KCL)}  # full mlp ablate
    for k in range(M):
        keep = np.setdiff1d(np.arange(M), [k])
        ce_k = per_tok_ce(dat, NDATA, A[:, keep] @ B[keep, :])
        for c in range(KCL):
            imp[k, c] = ce_k[assign == c].mean() - cl_full[c]
    print(f'importance matrix done ({time.time()-t0:.0f}s)', flush=True)

    # per-cluster minimal global rank for 90% CE recovery, greedy by importance
    minrank = {}; sets = {}
    for c in range(KCL):
        ce_full_c = cl_full[c]; ben_c = cl_abl[c] - ce_full_c
        order = np.argsort(-imp[:, c])
        r90 = M
        for r in RANKS:
            keep = order[:r]
            ce_r = per_tok_ce(dat, NDATA, A[:, keep] @ B[keep, :])[assign == c].mean()
            rec = (cl_abl[c] - ce_r) / max(ben_c, 1e-6)
            if rec >= 0.90:
                r90 = r; break
        minrank[c] = int(r90); sets[c] = order[:r90].tolist()
    # shared-language: pairwise Jaccard of component sets
    def jac(a, b): a, b = set(a), set(b); return len(a & b)/max(len(a | b), 1)
    jacs = [jac(sets[i], sets[j]) for i in range(KCL) for j in range(i+1, KCL)]
    mean_jac = float(np.mean(jacs))

    # NULL: random same-size set per cluster
    null_rec = []
    for c in range(KCL):
        r = minrank[c]; keep = rng.choice(M, size=min(r, M), replace=False)
        ce_r = per_tok_ce(dat, NDATA, A[:, keep] @ B[keep, :])[assign == c].mean()
        ben_c = cl_abl[c] - cl_full[c]
        null_rec.append((cl_abl[c] - ce_r)/max(ben_c, 1e-6))
    mean_null = float(np.mean(null_rec))

    print(f'\nper-cluster minimal GLOBAL rank (90% CE), shared M={M} basis:', flush=True)
    for c in range(KCL):
        ex = [d1(t) for t, _ in Counter(tk[assign == c].tolist()).most_common(4)]
        print(f'  cluster {c}: r90={minrank[c]:3d}  n={int((assign==c).sum()):4d}  comps={sorted(sets[c])[:8]}{"..." if minrank[c]>8 else ""}  {ex}', flush=True)
    print(f'\nmean pairwise Jaccard of component sets (shared language): {mean_jac:.3f}', flush=True)
    print(f'NULL random-set mean recovery {mean_null:.3f} (vs importance-set >=0.90)', flush=True)

    p0 = conv >= 0.8
    ranks = list(minrank.values())
    pb = 0.1 < mean_jac < 0.9
    null_ok = mean_null < 0.7
    print(f'\n(0) converges: {p0}; per-cluster r90 spread {min(ranks)}-{max(ranks)}; '
          f'(b) shared language (Jaccard {mean_jac:.2f} in .1-.9): {pb}; NULL: {null_ok}', flush=True)

    out = {'convergence': round(conv, 4), 'minrank': minrank,
           'component_sets': {c: sorted(sets[c]) for c in sets}, 'mean_jaccard': round(mean_jac, 4),
           'null_recovery': round(mean_null, 4), 'importance': imp.round(4).tolist(),
           'M': M, 'K': KCL, 'pred_0': bool(p0), 'pred_b_shared': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
