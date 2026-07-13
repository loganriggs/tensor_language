"""E5: DEPTH as the hierarchy-discovery device (data-free).

The point of multiple layers is not to *impose* hierarchy with a mask — it is to *compute* it.
A 1-layer bilinear map is degree-2 in x: flat features, no composition. A 2-layer stack is degree-4, but only
the degree-4 maps that FACTOR THROUGH A BOTTLENECK z. That bottleneck IS the hierarchy: layer 1 builds
mid-level features, layer 2 composes them. So "does this computation have hierarchical structure, and how wide
is it?" is answerable with no data at all:

  SWEEP THE BOTTLENECK WIDTH dz'. The smallest dz' at which tensor-sim stays 1.0 = the number of mid-level
  features the target actually needs. Below it fidelity MUST break (a control that can fail).

Unlike a block mask this presupposes nothing about which coordinates group together — the grouping is
DISCOVERED (we then check the recovered layer-1 features against the planted groups).

Ground truth (a genuinely compositional target):
  x in R^12 split into 4 disjoint groups of 3 -> each group is squeezed into ONE mid-level feature z_j
  (a quadratic form on that group alone) -> layer 2 mixes the 4 z's DENSELY into y.
  True bottleneck = 4. Nothing tells the transcoder that; it must find it.

Also computed exactly, data-free: the fidelity CEILING of a FLAT (1-layer, degree-2) transcoder on this
degree-4 target, at any rank — the quantitative statement that depth is NECESSARY.
(⟨deep|Λ|flat⟩ needs E[q q q] = a product of THREE quadratic forms; the generic set-partition/cumulant
expansion in tensor_sim_deep handles any n, and is MC-verified in __main__.)
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim_deep import collapse, expect_prod_quadratic, deep_forward

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NG, GSZ = 12, 4, 3          # x in R^12 = 4 groups x 3 coords;  true bottleneck = NG = 4
R1, R2, K = 8, 6, 3            # gt layer-1 hidden, layer-2 hidden, outputs
R1H, R2H = 16, 12              # transcoder hidden ranks (overcomplete at both layers)
STEPS, SEEDS = 1500, 3
sym = lambda M: 0.5 * (M + M.transpose(-1, -2))


# ---------- degree-4 (deep) and degree-2 (flat) Gaussian inner products ----------
def quads(Df, Lf, Rf):
    """Flat model as one symmetric matrix per output: y_k = xᵀ A_k x."""
    return torch.einsum("kh,hij->kij", Df, sym(Lf.unsqueeze(2) * Rf.unsqueeze(1)))


def _pairs(X, Y):
    n, m = X.shape[0], Y.shape[0]
    d = X.shape[-1]
    return (X.unsqueeze(1).expand(-1, m, -1, -1).reshape(-1, d, d),
            Y.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, d, d))


def deep_inner(a, b, S):
    Q, P = collapse(a[0], a[1], a[2], a[4], a[5])
    Qh, Ph = collapse(b[0], b[1], b[2], b[4], b[5])
    Qe, Qhe = _pairs(Q, Qh); Pe, Phe = _pairs(P, Ph)
    e = expect_prod_quadratic([Qe, Pe, Qhe, Phe], S)
    return ((a[3].T @ b[3]).reshape(-1) * e).sum()


def deep_flat_inner(a, f, S):
    """⟨y_deep · y_flat⟩ — diagonal in the output index k; needs E[q q q] (3 quadratic forms)."""
    Q, P = collapse(a[0], a[1], a[2], a[4], a[5])          # (r2,d,d)
    A = quads(*f)                                          # (K,d,d)
    Qe, Ae = _pairs(Q, A); Pe, _ = _pairs(P, A)
    e = expect_prod_quadratic([Qe, Pe, Ae], S)             # (r2*K,)
    return (a[3].T.reshape(-1) * e).sum()                  # weight (g,k) = D2[k,g]


def flat_inner(f1, f2, S):
    return expect_prod_quadratic([quads(*f1), quads(*f2)], S).sum()


def fid_deep(a, b, S, aa):   return (aa - 2 * deep_inner(a, b, S) + deep_inner(b, b, S)) / aa
def fid_flat(a, f, S, aa):   return (aa - 2 * deep_flat_inner(a, f, S) + flat_inner(f, f, S)) / aa


# ---------- the compositional ground truth ----------
def group_of():  return torch.arange(D, device=DEV) // GSZ          # coord -> group (0..NG-1)


def make_gt(seed):
    """Each mid-level feature z_j is a quadratic form on group j ALONE; layer 2 mixes the z's densely."""
    g = torch.Generator().manual_seed(seed)
    grp = (torch.arange(D) // GSZ)
    ug = torch.arange(R1) % NG                                       # layer-1 unit -> which group it reads
    L1 = torch.zeros(R1, D); R1_ = torch.zeros(R1, D)
    for p in range(R1):
        idx = torch.nonzero(grp == ug[p]).squeeze(1)
        L1[p, idx] = torch.randn(len(idx), generator=g)
        R1_[p, idx] = torch.randn(len(idx), generator=g)
    D1 = torch.randn(NG, R1, generator=g) * (torch.arange(NG)[:, None] == ug[None, :]).float()  # z_j <- group j
    L2, R2_ = torch.randn(R2, NG, generator=g), torch.randn(R2, NG, generator=g)                # DENSE mixing
    D2 = torch.randn(K, R2, generator=g)
    return tuple(t.to(DEV) for t in (D1, L1, R1_, D2, L2, R2_))


def purity(L, R):
    """Group purity of the RECOVERED layer-1 features: fraction of each feature's input mass in one group.
    gt = 1.000 by construction; chance (dense random) ~ 1/NG scaled by group size."""
    m = (L.abs() + R.abs())                                          # (r1', D)
    grp = group_of()
    per = torch.stack([m[:, grp == j].sum(1) for j in range(NG)], 1)  # (r1', NG)
    w = m.sum(1)                                                     # weight features by their total mass
    return float(((per.max(1).values / m.sum(1).clamp_min(1e-9)) * w).sum() / w.sum())


def fit_deep(seed, dz):
    a = make_gt(seed)
    g = torch.Generator(device=DEV).manual_seed(seed + 11)
    rnd = lambda *s: (torch.randn(*s, generator=g, device=DEV) / s[-1] ** .5).requires_grad_()
    b = [rnd(dz, R1H), rnd(R1H, D), rnd(R1H, D), rnd(K, R2H), rnd(R2H, dz), rnd(R2H, dz)]
    I = torch.eye(D, device=DEV)
    aa = deep_inner(a, a, I).detach()
    opt = torch.optim.Adam(b, 3e-3)
    for _ in range(STEPS):
        loss = fid_deep(a, tuple(b), I, aa) + 1e-3 * (b[1].abs().mean() + b[2].abs().mean())
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        return dict(tsim=1 - float(fid_deep(a, tuple(b), I, aa)), pur=purity(b[1], b[2]))


def fit_flat(seed, r):
    a = make_gt(seed)
    g = torch.Generator(device=DEV).manual_seed(seed + 11)
    rnd = lambda *s: (torch.randn(*s, generator=g, device=DEV) / s[-1] ** .5).requires_grad_()
    f = [rnd(K, r), rnd(r, D), rnd(r, D)]
    I = torch.eye(D, device=DEV)
    aa = deep_inner(a, a, I).detach()
    opt = torch.optim.Adam(f, 3e-3)
    for _ in range(STEPS):
        loss = fid_flat(a, tuple(f), I, aa)
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        return 1 - float(fid_flat(a, tuple(f), I, aa))


if __name__ == "__main__":
    print("E5  DEPTH AS THE HIERARCHY DETECTOR — data-free (L_fid only, Λ = N(0,I))\n")
    print(f"  gt: x{D} = {NG} disjoint groups of {GSZ}; each group -> ONE mid-level feature; layer 2 mixes them")
    print(f"      densely -> y{K}.  TRUE bottleneck = {NG}.  Transcoder is told NOTHING about the groups.")
    print(f"  transcoder: r1'={R1H}, r2'={R2H}, bottleneck dz' swept.  {SEEDS} seeds.\n")

    print("  (1) BOTTLENECK SWEEP — the smallest dz' holding tsim=1 reveals the hierarchy's width")
    hdr = f"  {'dz′':>4s} {'tensor-sim':>14s} {'group purity of recovered layer-1 feats':>40s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for dz in [1, 2, 3, 4, 5, 6, 8]:
        rs = [fit_deep(s, dz) for s in range(SEEDS)]
        ts, td = np.mean([r["tsim"] for r in rs]), np.std([r["tsim"] for r in rs])
        pu = np.mean([r["pur"] for r in rs])
        tag = "  <-- TRUE bottleneck" if dz == NG else ("  (starved)" if dz < NG else "  (slack)")
        print(f"  {dz:4d} {ts:8.3f}±{td:.3f} {pu:35.3f}{tag}")

    print(f"\n  (2) DEPTH IS NECESSARY — exact fidelity ceiling of a FLAT (1-layer, degree-2) transcoder")
    print(f"  {'rank':>5s} {'tensor-sim':>14s}")
    for r in [4, 8, 16, 32, 64]:
        rs = [fit_flat(s, r) for s in range(SEEDS)]
        print(f"  {r:5d} {np.mean(rs):8.3f}±{np.std(rs):.3f}")
    print("\n  A degree-2 model cannot represent a degree-4 target at ANY rank: the ceiling is a property of")
    print("  the function class, not of optimisation. Depth buys the composition, and L_fid MEASURES it.")
    print("DONE")
