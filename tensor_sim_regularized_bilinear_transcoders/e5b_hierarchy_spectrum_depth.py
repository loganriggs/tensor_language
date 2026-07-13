"""E5b: why the bottleneck sweep had no elbow — hierarchy WIDTH IS A SPECTRUM, not an integer.

E5(1) proposed: "the smallest dz' holding tsim=1 = the number of mid-level features." It FAILED — tsim was
already 0.954 at dz'=1. Diagnosis: L_fid is a RELATIVE-norm error, and E5's planted mid-level features had
wildly unequal scales (random D1 rows), so one z_j dominated E‖y‖² and a single squared quadratic form already
captured most of it. So the sweep does not read an integer — it reads a SPECTRUM, weighted by how much each
mid-level feature contributes to the output. Sharpness of the curve = how equally the mid-level features matter.

Test (a falsifiable prediction, two controls that can fail):
  BALANCED gt   — rescale D1 so every z_j has E[z_j²]=1: all NG mid-level features matter equally.
                  PREDICT: a genuine elbow appears at dz' = NG (tsim breaks below it, saturates at/above it).
  SKEWED gt     — geometric scale ladder across the z_j (one dominates): PREDICT no elbow, a graded curve.
  1-FEATURE gt  — a target that TRULY needs only one mid-level feature. PREDICT tsim=1 at dz'=1 (a control:
                  the sweep must report a SMALL width when the width really is small, else it measures nothing).

Also: E5's recovered layer-1 features had group purity 0.44 ≈ chance — the transcoder found the right function
but not the planted groups (E1/E2's CP non-uniqueness). E2's lesson says sparsity should break it, so we sweep
the L1 strength at dz'=NG against an explicit chance control.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from e5_hierarchy_via_depth import (D, NG, GSZ, R1, R2, K, R1H, R2H, SEEDS, DEV, sym,
                                    deep_inner, fid_deep, purity, group_of)
from tensor_sim_deep import expect_prod_quadratic

torch.set_default_dtype(torch.float32)
STEPS = 1500


def make_gt(seed, scales):
    """scales: per-mid-level-feature output weight (len NG). Each z_j is a quadratic form on group j alone."""
    g = torch.Generator().manual_seed(seed)
    grp = torch.arange(D) // GSZ
    ug = torch.arange(R1) % NG
    L1 = torch.zeros(R1, D); R1_ = torch.zeros(R1, D)
    for p in range(R1):
        idx = torch.nonzero(grp == ug[p]).squeeze(1)
        L1[p, idx] = torch.randn(len(idx), generator=g)
        R1_[p, idx] = torch.randn(len(idx), generator=g)
    D1 = torch.randn(NG, R1, generator=g) * (torch.arange(NG)[:, None] == ug[None, :]).float()
    L2, R2_ = torch.randn(R2, NG, generator=g), torch.randn(R2, NG, generator=g)
    D2 = torch.randn(K, R2, generator=g)
    a = tuple(t.to(DEV) for t in (D1, L1, R1_, D2, L2, R2_))
    # normalise each mid-level feature to E[z_j²]=1 under x~N(0,I), then apply the requested scale ladder
    I = torch.eye(D, device=DEV)
    M = torch.einsum("jp,pab->jab", a[0], sym(a[1].unsqueeze(2) * a[2].unsqueeze(1)))     # z_j = xᵀ M_j x
    ez2 = expect_prod_quadratic([M, M], I)                                                # (NG,)
    s = torch.as_tensor(scales, dtype=torch.float32, device=DEV) / ez2.sqrt()
    return (a[0] * s[:, None], a[1], a[2], a[3], a[4], a[5])


def fit(seed, dz, scales, lam=1e-3):
    a = make_gt(seed, scales)
    g = torch.Generator(device=DEV).manual_seed(seed + 11)
    rnd = lambda *s: (torch.randn(*s, generator=g, device=DEV) / s[-1] ** .5).requires_grad_()
    b = [rnd(dz, R1H), rnd(R1H, D), rnd(R1H, D), rnd(K, R2H), rnd(R2H, dz), rnd(R2H, dz)]
    I = torch.eye(D, device=DEV)
    aa = deep_inner(a, a, I).detach()
    pur0 = purity(b[1].detach(), b[2].detach())                       # CHANCE control (random init)
    opt = torch.optim.Adam(b, 3e-3)
    for _ in range(STEPS):
        loss = fid_deep(a, tuple(b), I, aa) + lam * (b[1].abs().mean() + b[2].abs().mean())
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        return dict(tsim=1 - float(fid_deep(a, tuple(b), I, aa)),
                    pur=purity(b[1], b[2]), pur0=pur0)


BAL = [1.0] * NG                                   # all NG mid-level features matter equally
SKEW = [1.0, 0.5, 0.25, 0.125]                     # geometric ladder: one dominates
ONE = [1.0, 0.0, 0.0, 0.0]                         # target truly needs ONE mid-level feature

if __name__ == "__main__":
    print("E5b  HIERARCHY WIDTH IS A SPECTRUM — the tsim(dz′) curve IS the mid-level-feature scree plot\n")
    print(f"  gt: x{D} = {NG} groups of {GSZ}; group j -> mid-level feature z_j; layer 2 mixes z densely -> y{K}.")
    print(f"  Data-free (L_fid, Λ=N(0,I)). {SEEDS} seeds. Only the SCALE LADDER over the z_j changes.\n")
    hdr = f"  {'dz′':>4s} {'BALANCED (all 4 equal)':>24s} {'SKEWED (1,.5,.25,.125)':>24s} {'ONE-FEATURE gt':>18s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for dz in [1, 2, 3, 4, 5, 6]:
        row = []
        for sc in (BAL, SKEW, ONE):
            rs = [fit(s, dz, sc) for s in range(SEEDS)]
            row.append((np.mean([r["tsim"] for r in rs]), np.std([r["tsim"] for r in rs])))
        tag = "  <-- true width (balanced)" if dz == NG else ""
        print(f"  {dz:4d} {row[0][0]:16.3f}±{row[0][1]:.3f} {row[1][0]:17.3f}±{row[1][1]:.3f}"
              f" {row[2][0]:11.3f}±{row[2][1]:.3f}{tag}")
    print("\n  Read the CURVE, not a single number: where it saturates = the EFFECTIVE number of mid-level")
    print("  features. A sharp knee => a few equally-important sub-features. A graded ramp => a skewed")
    print("  hierarchy. Saturation at 1 => the computation is not hierarchical at all (one mid-level feature).")

    print(f"\n  L1 STRENGTH vs GROUP RECOVERY at dz′={NG} (balanced gt) — does sparsity find the planted groups?")
    print(f"  {'λ_L1':>8s} {'tensor-sim':>13s} {'group purity':>14s}")
    ch = []
    for lam in [0.0, 1e-3, 3e-3, 1e-2, 3e-2]:
        rs = [fit(s, NG, BAL, lam=lam) for s in range(SEEDS)]
        ts = np.mean([r["tsim"] for r in rs]); pu = np.mean([r["pur"] for r in rs])
        ch.append(np.mean([r["pur0"] for r in rs]))
        print(f"  {lam:8.3g} {ts:9.3f} {pu:14.3f}")
    print(f"\n  CONTROL random-init group purity = {np.mean(ch):.3f} (chance).  gt purity = 1.000.")
    print("DONE")
