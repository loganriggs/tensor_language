"""E4: hierarchy ACROSS LAYERS — a data-free fit of a TWO-layer bilinear transcoder (degree-4 metric).

Uses the verified degree-4 tensor-sim (`tensor_sim_deep.py`): stacking two bilinear layers gives a degree-4
polynomial in x, y_k = Σ_g D2_kg (xᵀQ_g x)(xᵀP_g x), and ⟨y|Λ|ŷ⟩ is closed-form via the set-partition/cumulant
expansion of E[∏ of 4 quadratic forms].  So the whole 2-layer stack can be fit with NO DATA.

Ground truth is a genuine CROSS-LAYER TREE:  x-coords partitioned into blocks; layer-1 unit p reads only
block(p); z-coord j belongs to block(j); D1 routes unit p only to z-coords of block(p); layer-2 unit g reads
only z-coords of block(g).  Information stays inside its block for the whole depth — hierarchy across layers.

Arms (all fit the SAME gt):
  MSE(subspace data)          — the baseline that E1/FINDING 3 shows goes blind off-distribution
  deep L_fid (dense)          — DATA-FREE, no structure
  deep L_fid + L1             — DATA-FREE, soft sparsity (E2's winner)
  deep L_fid + cross-layer hierarchy (hard masks in BOTH layers)
plus the diagnostic control: the same hard cross-layer prior against a NON-hierarchical gt (must lose fidelity).

Recovery metric is gauge-correct: z admits a GL(dz) gauge (D1 -> S D1, L2/R2 -> L2/R2 S^-1) which leaves y
unchanged, so layer-1 factors are only identifiable as the SET of rank-1 quadratic forms {sym(l_p r_pᵀ)}.
We MMCS those (permutation+scale invariant), never D1.
"""
import sys, itertools, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim_deep import collapse, expect_prod_quadratic, deep_forward

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, R1, DZ, R2, K = 8, 6, 4, 4, 3          # x:8 -> l1 hidden:6 -> z:4 -> l2 hidden:4 -> y:3
R1H, R2H = 12, 8                          # transcoder: 2x overcomplete at both layers
NBLK, KSP, STEPS, SEEDS = 2, 2, 1200, 3


# ---------- batched degree-4 inner product (vectorised over the (g,g') pairs) ----------
def deep_inner(a, b, Sigma):
    """a, b = (D1,L1,R1,D2,L2,R2).  ⟨y|Λ|ŷ⟩ = E[y·ŷ], x~N(0,Sigma).  Closed form, data-free."""
    Q, P = collapse(a[0], a[1], a[2], a[4], a[5])          # (r2, d, d)
    Qh, Ph = collapse(b[0], b[1], b[2], b[4], b[5])        # (r2', d, d)
    r2, r2h = Q.shape[0], Qh.shape[0]
    ex = lambda M, n: M.unsqueeze(1).expand(-1, n, -1, -1).reshape(-1, D, D)
    exh = lambda M, n: M.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, D, D)
    e = expect_prod_quadratic([ex(Q, r2h), ex(P, r2h), exh(Qh, r2), exh(Ph, r2)], Sigma)   # (r2*r2',)
    W = a[3].T @ b[3]                                       # (r2, r2')
    return (W.reshape(-1) * e).sum()


def deep_fid(a, b, Sigma, aa=None):
    if aa is None:
        aa = deep_inner(a, a, Sigma)
    return (aa - 2 * deep_inner(a, b, Sigma) + deep_inner(b, b, Sigma)) / aa


# ---------- ground truth ----------
def blocks_x(): return torch.arange(D, device=DEV) % NBLK
def blocks_z(): return torch.arange(DZ, device=DEV) % NBLK


def make_gt(seed, kind="tree"):
    """kind='tree': cross-layer hierarchy (block-confined at every layer). 'random': respects no partition."""
    g = torch.Generator().manual_seed(seed)
    cx, cz = blocks_x().cpu(), blocks_z().cpu()
    ub1 = torch.randint(0, NBLK, (R1,), generator=g)             # layer-1 unit -> block
    ub2 = torch.arange(R2) % NBLK                                # layer-2 unit -> block
    L1 = torch.zeros(R1, D); R1_ = torch.zeros(R1, D)
    for p in range(R1):
        pool = torch.nonzero(cx == ub1[p]).squeeze(1) if kind == "tree" else torch.arange(D)
        for M in (L1, R1_):
            idx = pool[torch.randperm(len(pool), generator=g)[:KSP]]
            M[p, idx] = torch.randn(KSP, generator=g)
    D1 = torch.randn(DZ, R1, generator=g)
    L2 = torch.randn(R2, DZ, generator=g); R2_ = torch.randn(R2, DZ, generator=g)
    if kind == "tree":
        D1 = D1 * (cz[:, None] == ub1[None, :]).float()          # unit p writes only to its block's z-coords
        m2 = (cz[None, :] == ub2[:, None]).float()               # layer-2 unit g reads only its block's z
        L2, R2_ = L2 * m2, R2_ * m2
    D2 = torch.randn(K, R2, generator=g)
    return tuple(t.to(DEV) for t in (D1, L1, R1_, D2, L2, R2_))


def feat_mmcs(a, b):
    """MMCS over layer-1 features as symmetric rank-1 forms sym(l r^T) — the GL(dz)-gauge-invariant object."""
    f = lambda L, R: torch.nn.functional.normalize(
        (0.5 * (L.unsqueeze(2) * R.unsqueeze(1) + R.unsqueeze(2) * L.unsqueeze(1))).reshape(L.shape[0], -1), dim=1)
    C = (f(a[1], a[2]) @ f(b[1], b[2]).T).abs()
    return float(C.max(1).values.mean())


def masks(r1, r2):
    g = torch.Generator(device=DEV).manual_seed(0)
    cx, cz = blocks_x(), blocks_z()
    u1 = torch.randint(0, NBLK, (r1,), generator=g, device=DEV)
    u2 = torch.arange(r2, device=DEV) % NBLK
    m1 = (cx[None, :] == u1[:, None]).float()                    # (r1, D)  layer-1 factor mask
    mD1 = (cz[:, None] == u1[None, :]).float()                   # (DZ, r1) routing mask
    m2 = (cz[None, :] == u2[:, None]).float()                    # (r2, DZ) layer-2 factor mask
    return m1, mD1, m2


def fit(seed, arm, gt_kind="tree"):
    a = make_gt(seed, gt_kind)
    g = torch.Generator(device=DEV).manual_seed(seed + 11)
    rnd = lambda *s: (torch.randn(*s, generator=g, device=DEV) / s[-1] ** .5).requires_grad_()
    D1, L1, Rr1, D2, L2, Rr2 = rnd(DZ, R1H), rnd(R1H, D), rnd(R1H, D), rnd(K, R2H), rnd(R2H, DZ), rnd(R2H, DZ)
    ps = [D1, L1, Rr1, D2, L2, Rr2]
    I = torch.eye(D, device=DEV)
    m1, mD1, m2 = masks(R1H, R2H)
    hier = arm.startswith("hier")

    def eff():
        if hier:
            return (D1 * mD1, L1 * m1, Rr1 * m1, D2, L2 * m2, Rr2 * m2)
        return tuple(ps)

    mm0 = feat_mmcs(a, [t.detach() for t in eff()])
    aa = deep_inner(a, a, I).detach()
    if arm == "MSE":                                             # data on a 4-dim subspace of R^8
        B = torch.linalg.qr(torch.randn(D, 4, generator=g, device=DEV))[0]
        x = torch.randn(4096, 4, generator=g, device=DEV) @ B.T
        y = deep_forward(*a, x).detach()
    opt = torch.optim.Adam(ps, 3e-3)
    for _ in range(STEPS):
        b = eff()
        if arm == "MSE":
            loss = ((deep_forward(*b, x) - y) ** 2).sum(1).mean() / (y ** 2).sum(1).mean()
        else:
            loss = deep_fid(a, b, I, aa=aa)
            if arm in ("L1", "hier+L1"):
                loss = loss + 3e-3 * sum(t.abs().mean() for t in (L1, Rr1, L2, Rr2))
        loss.backward(); opt.step(); opt.zero_grad()

    with torch.no_grad():
        b = eff()
        tsim = 1 - float(deep_fid(a, b, I, aa=aa))               # TRUE (full-support) global fidelity
        xo = torch.randn(4096, D, generator=g, device=DEV)       # off-distribution probe (all directions)
        yo, yh = deep_forward(*a, xo), deep_forward(*b, xo)
        ood = float(((yh - yo) ** 2).sum(1).mean() / (yo ** 2).sum(1).mean())
        return dict(tsim=tsim, ood=ood, mm=feat_mmcs(a, b), mm0=mm0)


if __name__ == "__main__":
    print("E4  TWO-LAYER (degree-4) transcoder — hierarchy ACROSS LAYERS, fit DATA-FREE\n")
    print(f"  gt: cross-layer tree, x{D} -> h{R1} -> z{DZ} -> h{R2} -> y{K}, {NBLK} blocks confined at every layer")
    print(f"  transcoder: r1'={R1H}, r2'={R2H} (2x overcomplete). {SEEDS} seeds. Λ = N(0,I) full-support.\n")
    hdr = f"  {'arm':22s} {'TRUE tsim':>11s} {'MSE(OOD)':>10s} {'layer-1 feat recov':>19s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    ctrl = []
    for arm, lab in [("MSE", "MSE (subspace data)"), ("dense", "deep L_fid (dense)"),
                     ("L1", "deep L_fid + L1"), ("hier", "deep L_fid + X-LAYER hier"),
                     ("hier+L1", "deep L_fid + hier + L1")]:
        rs = [fit(s, arm) for s in range(SEEDS)]
        f = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
        ts, td = f("tsim"); od, _ = f("ood"); mm, md = f("mm"); ctrl.append(f("mm0")[0])
        print(f"  {lab:22s} {ts:6.3f}±{td:.3f} {od:10.3f} {mm:13.3f}±{md:.3f}")
    print(f"\n  CONTROL random-init feat recovery = {np.mean(ctrl):.3f} (chance)")

    print("\n  DIAGNOSTIC: does the hard cross-layer prior betray a MIS-SPECIFIED structure?")
    print(f"  {'gt':28s} {'hard X-layer hier: TRUE tsim':>30s}")
    for gk, lab in [("tree", "tree (truly hierarchical)"), ("random", "random (respects no blocks)")]:
        rs = [fit(s, "hier", gt_kind=gk) for s in range(SEEDS)]
        ts = np.mean([r["tsim"] for r in rs]); td = np.std([r["tsim"] for r in rs])
        print(f"  {lab:28s} {ts:24.3f}±{td:.3f}")
    print("DONE")
