"""CROSS-LAYER ABLATION CLUSTER (user: cross-layer clustering via random-pair
ablation of A-SVD components). Extends 719-721 (single-layer A-SVD ablation-
covariance clustering) ACROSS layers. Pool the data-conditioned A-SVD
output components of SEVERAL layers (front to back); ablate random subsets
that SPAN layers; record per-datapoint CE damage; covariance across
datapoints -> cluster. The clusters now reflect CROSS-LAYER circuit
dependencies (tokens damaged by the same cross-layer component combinations
group together). Also report which cross-layer components each cluster
depends on, and convergence.

Foundation is A-SVD throughout: each layer's components are A[:, :M] from
asvd_fast(Down.weight, captured gate) -- the data-conditioned decomposition.

REGISTERED PREDICTIONS:
  (0) SANITY: the datapoint covariance CONVERGES (split-half corr >= 0.8);
  (a) CROSS-LAYER STRUCTURE: the clustering assigns tokens by cross-layer
      damage, and at least some clusters depend on components from MULTIPLE
      DIFFERENT layers (not just one) -- report per-cluster the layer-mix of
      its top-damaging components;
  (b) report clusters (token content) + their dominant (layer,component)s +
      convergence;
  NULL: shuffling each datapoint's damage vector destroys the covariance
      (split-half corr ~ 0)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_layer_ablation_cluster_results.json'
LAYERS = [0, 1, 2, 16, 17]      # front-to-back span
M = 24                           # A-SVD components per layer
NFIT = 96; NDATA = 8             # ~2048 datapoints
S = 4                            # ablation subset size (spans layers)
T_CE = 220; KCL = 8
PROJ = {}                        # layer -> orthonormal (D, r) to project out of that layer's mlp output


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


def mk_hook(layer):
    def hook(mo, i_, o_):
        P = PROJ.get(layer)
        if P is None: return o_
        of = o_.float(); return (of - (of @ P) @ P.T).to(o_.dtype)
    return hook


@torch.no_grad()
def capture_gate(rows, n, layer):
    cap = []
    h = m.transformer.h[layer].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def per_tok_ce(rows, n):
    ce = []; toks = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    return torch.cat(ce).numpy(), (torch.cat(toks).numpy() if toks else None)


def simmat(L):
    Lc = L - L.mean(0, keepdims=True); Ln = Lc/(np.linalg.norm(Lc, axis=0, keepdims=True)+1e-9)
    return Ln.T @ Ln


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    allrows = cl.fineweb_rows(NFIT + NDATA)
    fit, dat = allrows[:NFIT], allrows[NFIT:NFIT+NDATA]

    # A-SVD components per layer (data-conditioned)
    Acomp = {}
    for L in LAYERS:
        g = capture_gate(fit, NFIT, L).to(DEV)
        A, _ = asvd_fast(m.transformer.h[L].mlp.Down.weight.data.float().to(DEV), g)
        Acomp[L] = (A[:, :M] / A[:, :M].norm(dim=0, keepdim=True))     # (D, M) unit dirs
        del g
    pool = [(L, k) for L in LAYERS for k in range(M)]      # cross-layer component pool
    print(f'pool: {len(pool)} components across {len(LAYERS)} layers', flush=True)

    hooks = [m.transformer.h[L].mlp.register_forward_hook(mk_hook(L)) for L in LAYERS]
    for L in LAYERS: PROJ[L] = None
    base, toks = per_tok_ce(dat, NDATA); N = len(base)

    rng = np.random.default_rng(0)
    Lce = np.zeros((T_CE, N), dtype=np.float32); subsets = []
    for t in range(T_CE):
        sel = [pool[i] for i in rng.choice(len(pool), size=S, replace=False)]; subsets.append(sel)
        for L in LAYERS:
            idxs = [k for (Ls, k) in sel if Ls == L]
            PROJ[L] = Acomp[L][:, idxs] if idxs else None
        ce, _ = per_tok_ce(dat, NDATA); Lce[t] = ce - base
        for L in LAYERS: PROJ[L] = None
    print(f'CE-damage matrix done ({time.time()-t0:.0f}s)', flush=True)

    # convergence
    h = T_CE//2; iu = np.triu_indices(N, 1)
    conv = float(np.corrcoef(simmat(Lce[:h])[iu], simmat(Lce[h:2*h])[iu])[0,1])
    Lsh = Lce.copy()
    for i in range(N): Lsh[:, i] = Lsh[rng.permutation(T_CE), i]
    null_conv = float(np.corrcoef(simmat(Lsh[:h])[iu], simmat(Lsh[h:2*h])[iu])[0,1])
    print(f'convergence split-half corr {conv:.3f} (null {null_conv:.3f})', flush=True)

    # cluster datapoints by cross-layer damage vector
    Lc = Lce - Lce.mean(0, keepdims=True); Ln = torch.tensor((Lc/(np.linalg.norm(Lc,axis=0,keepdims=True)+1e-9)).T)
    assign = kmeans(Ln, KCL).numpy()

    # per-cluster: which (layer,component) trials damage it most -> layer mix
    # attribute each trial's damage to its components; per cluster, top components by mean damage
    comp_dmg = {c: np.zeros(len(pool)) for c in range(KCL)}
    comp_cnt = np.zeros(len(pool))
    pool_idx = {p:i for i,p in enumerate(pool)}
    for t in range(T_CE):
        for p in subsets[t]: comp_cnt[pool_idx[p]] += 1
        for c in range(KCL):
            dmg = Lce[t, assign==c].mean() if (assign==c).any() else 0.0
            for p in subsets[t]: comp_dmg[c][pool_idx[p]] += dmg
    clusters = []
    for c in range(KCL):
        avg = comp_dmg[c]/np.maximum(comp_cnt,1)
        top = np.argsort(-avg)[:6]
        layer_mix = Counter(pool[i][0] for i in top)
        ex = [d1(t) for t,_ in Counter(toks[assign==c].tolist()).most_common(5)]
        clusters.append({'c': c, 'n': int((assign==c).sum()), 'top_components': [list(pool[i]) for i in top],
                         'layer_mix': dict(layer_mix), 'tokens': ex})
        print(f'cluster {c}: n={int((assign==c).sum()):4d} layer-mix {dict(layer_mix)}  {ex}', flush=True)
    for hh in hooks: hh.remove()

    multi = sum(1 for cl_ in clusters if len(cl_['layer_mix']) >= 2)
    p0 = conv >= 0.8; pa = multi >= 1; null_ok = null_conv < 0.2
    print(f'\n(0) converges {p0}; (a) {multi} clusters span >=2 layers: {pa}; NULL {null_ok}', flush=True)
    out = {'layers': LAYERS, 'M': M, 'convergence': round(conv,3), 'null_convergence': round(null_conv,3),
           'clusters': clusters, 'n_multilayer_clusters': multi, 'pred_0': bool(p0),
           'pred_a_crosslayer': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
