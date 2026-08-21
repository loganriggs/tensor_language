"""CROSS-LAYER COMPONENT GROUPS (clean refinement of 735 + the user's cross-
layer COMPOSITION idea). Instead of random subsets (crude additive
attribution, 735), ablate each A-SVD component SINGLY across 5 layers ->
exact (n_components x n_datapoints) CE-damage matrix. Then cluster the
COMPONENTS by which datapoints they co-damage: components (from possibly
DIFFERENT layers) that damage the SAME tokens are COMPOSING into a cross-
layer circuit. Groups that span >=2 layers are cross-layer circuits.

Foundation = data-conditioned A-SVD per layer.

REGISTERED PREDICTIONS:
  (0) SANITY: single-component damages are non-trivial and the damage matrix
      has structure (not all components identical);
  (a) CROSS-LAYER CIRCUITS: clustering the 120 components by their datapoint
      co-damage yields groups, and at least one group SPANS >=2 layers
      (components from different layers that co-damage the same tokens =
      a composed cross-layer circuit); report the groups, their layer span,
      and the tokens they co-damage;
  (b) report per-group: member (layer,comp), layer span, top co-damaged tokens;
  NULL: components' damage profiles are NOT all mutually correlated -- a
      random pair of components has low profile correlation (so the groups
      are real structure, not everything-damages-everything)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_layer_component_groups_results.json'
LAYERS = [0, 1, 2, 16, 17]
M = 24
NFIT = 96; NDATA = 12       # ~3072 datapoints
NGROUP = 10
PROJ = {}


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


def kmeans(Xn, k, iters=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(Xn, C).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = Xn[a == j].mean(0)
    return a


def mk_hook(layer):
    def hook(mo, i_, o_):
        d = PROJ.get(layer)
        if d is None: return o_
        of = o_.float(); return (of - (of @ d)[..., None] * d).to(o_.dtype)
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
def per_tok_ce(rows, n, want_tok=False):
    ce = []; toks = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        if want_tok: toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    c = torch.cat(ce).numpy()
    return (c, torch.cat(toks).numpy()) if want_tok else c


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    allrows = cl.fineweb_rows(NFIT + NDATA)
    fit, dat = allrows[:NFIT], allrows[NFIT:NFIT+NDATA]
    Acomp = {}
    for L in LAYERS:
        g = capture_gate(fit, NFIT, L).to(DEV)
        A, _ = asvd_fast(m.transformer.h[L].mlp.Down.weight.data.float().to(DEV), g)
        Acomp[L] = (A[:, :M] / A[:, :M].norm(dim=0, keepdim=True)); del g
    pool = [(L, k) for L in LAYERS for k in range(M)]; P = len(pool)

    hooks = [m.transformer.h[L].mlp.register_forward_hook(mk_hook(L)) for L in LAYERS]
    for L in LAYERS: PROJ[L] = None
    base, toks = per_tok_ce(dat, NDATA, want_tok=True); N = len(base)
    DM = np.zeros((P, N), dtype=np.float32)
    for pi, (L, k) in enumerate(pool):
        PROJ[L] = Acomp[L][:, k]
        DM[pi] = per_tok_ce(dat, NDATA) - base
        PROJ[L] = None
    for hh in hooks: hh.remove()
    print(f'{P} single-component damage vectors, N={N} ({time.time()-t0:.0f}s)', flush=True)

    # cluster components by datapoint co-damage (normalized profile)
    DMc = DM - DM.mean(1, keepdims=True)
    DMn = DMc / (np.linalg.norm(DMc, axis=1, keepdims=True) + 1e-9)
    grp = kmeans(torch.tensor(DMn), NGROUP).numpy()

    # null: random pair profile correlation
    rng = np.random.default_rng(0)
    pairs = [(rng.integers(P), rng.integers(P)) for _ in range(500)]
    corrs = [float(np.dot(DMn[a], DMn[b])) for a, b in pairs if a != b]
    null_mean_abs = float(np.mean(np.abs(corrs)))

    groups = []
    for gi in range(NGROUP):
        members = [pool[i] for i in range(P) if grp[i] == gi]
        if not members: continue
        layers_in = sorted(set(L for L, _ in members))
        # tokens this group co-damages: mean damage over group members, top tokens
        gdmg = DM[[i for i in range(P) if grp[i] == gi]].mean(0)
        toptok = [d1(t) for t in np.argsort(-gdmg)[:6]]
        # avg pairwise profile corr within group (cohesion)
        idxs = [i for i in range(P) if grp[i] == gi]
        coh = float(np.mean([np.dot(DMn[a], DMn[b]) for a in idxs for b in idxs if a < b])) if len(idxs) > 1 else 1.0
        groups.append({'group': gi, 'n': len(members), 'layers': layers_in,
                       'members': [list(mp) for mp in members[:8]], 'top_damaged_tokens': toptok,
                       'cohesion': round(coh, 3)})
        print(f'group {gi}: {len(members)} comps, layers {layers_in}, cohesion {coh:.2f}, '
              f'damages {toptok}', flush=True)

    xlayer = [g for g in groups if len(g['layers']) >= 2]
    print(f'\nrandom-pair mean |profile corr| {null_mean_abs:.3f}', flush=True)
    print(f'{len(xlayer)}/{len(groups)} groups span >=2 layers (cross-layer circuits)', flush=True)
    p0 = float(np.abs(DM).mean()) > 0.001
    pa = len(xlayer) >= 1
    null_ok = null_mean_abs < 0.5
    out = {'layers': LAYERS, 'M': M, 'n_components': P, 'groups': groups,
           'n_crosslayer_groups': len(xlayer), 'null_pair_abscorr': round(null_mean_abs,3),
           'pred_0': bool(p0), 'pred_a_crosslayer': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'\n(a) cross-layer component groups exist: {pa}; NULL pairs not all-correlated: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
