"""E3a: hierarchy as a SPECTRUM (not one hard point), tested against MATCHED and MISMATCHED ground truth.
E3b: the metric-temperature knob  Sigma_t = (1-t)*Sigma_data + t*I   (FINDING 3's tradeoff).

E2 found the hard block prior FAILED (tsim 0.478) — but that was one extreme AND the ground truth did not
respect blocks. The honest question is not "does hierarchy help" but "does it help WHEN THE LAYER IS
HIERARCHICAL". So we plant two ground truths:
    gt=random  — 3-sparse factors, support anywhere (does NOT respect any block partition)
    gt=block   — each hidden unit reads coords from ONE block (genuinely hierarchical layer)
and sweep hierarchy GRANULARITY (n_blocks = 1 [dense] .. 16 [maximally fine]), plus a SOFT/graded variant
(off-block weights penalized with strength s, not zeroed: s=0 dense, s->inf hard block).

All fits are DATA-FREE (L_fid only, full-support Lambda).  5 seeds.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import fid_loss_mean, tensor_inner_mean, lifted_moments, forward
from e1_synthetic_recovery import rank1_mmcs, D_IN, R_GT, K_OUT, K_SP, DEV

torch.set_default_dtype(torch.float32)
STEPS = 5000
N_BLK_GT = 4                                   # the gt block partition (for gt=block)


def coord_block(n_blocks, d_in=D_IN):
    return torch.arange(d_in, device=DEV) % n_blocks


def make_gt(seed, kind):
    """kind='random': sparse support anywhere.  kind='block': each unit reads ONE block (hierarchical layer)."""
    g = torch.Generator().manual_seed(seed); d = D_IN + 1
    L = torch.zeros(R_GT, d); R = torch.zeros(R_GT, d)
    cb = (torch.arange(D_IN) % N_BLK_GT)
    for h in range(R_GT):
        if kind == "random":
            iL = torch.randperm(D_IN, generator=g)[:K_SP]
            iR = torch.randperm(D_IN, generator=g)[:K_SP]
        else:
            blk = int(torch.randint(0, N_BLK_GT, (1,), generator=g))
            pool = torch.nonzero(cb == blk).squeeze(1)
            iL = pool[torch.randperm(len(pool), generator=g)[:K_SP]]
            iR = pool[torch.randperm(len(pool), generator=g)[:K_SP]]
        L[h, iL + 1] = torch.randn(len(iL), generator=g)
        R[h, iR + 1] = torch.randn(len(iR), generator=g)
    D = torch.randn(K_OUT, R_GT, generator=g)
    p = torch.randperm(R_GT, generator=g)                       # true CP gauge (perm+rescale)
    a = torch.rand(R_GT, generator=g) + .5; b = torch.rand(R_GT, generator=g) + .5
    L, R, D = (a[:, None] * L)[p], (b[:, None] * R)[p], (D / (a * b)[None, :])[:, p]
    return D.to(DEV), L.to(DEV), R.to(DEV)


def full_support(d):
    S = torch.eye(d, device=DEV); S[0, 0] = 0.0
    m = torch.zeros(d, device=DEV); m[0] = 1.0
    return S, m


def fit(seed, gt_kind, n_blocks, soft=None, r_tc=32, lam_l1=3e-3, Sig=None, mu=None):
    """n_blocks=1 -> dense. soft=None -> HARD mask; soft=s -> off-block weights L1-penalized with strength s."""
    Dg, Lg, Rg = make_gt(seed, gt_kind); d = D_IN + 1
    g = torch.Generator(device=DEV).manual_seed(seed + 5)
    if Sig is None:
        Sig, mu = full_support(d)
    aa = tensor_inner_mean(Dg, Lg, Rg, Dg, Lg, Rg, Sig, mu)
    Lt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Rt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Dt = (torch.randn(K_OUT, r_tc, generator=g, device=DEV) / r_tc ** .5).requires_grad_()
    # block assignment for the transcoder's hidden units
    cb = coord_block(n_blocks)
    ub = torch.randint(0, n_blocks, (r_tc,), generator=g, device=DEV)
    inblk = (cb[None, :] == ub[:, None]).float()                        # (r_tc, D_IN)
    hard = torch.cat([torch.ones(r_tc, 1, device=DEV), inblk], 1)
    offblk = torch.cat([torch.zeros(r_tc, 1, device=DEV), 1 - inblk], 1)
    opt = torch.optim.Adam([Lt, Rt, Dt], 3e-3)
    for _ in range(STEPS):
        if n_blocks == 1 or soft is not None:
            Le, Re = Lt, Rt                                             # dense params; structure via penalty
        else:
            Le, Re = Lt * hard, Rt * hard                               # HARD structural mask
        loss = fid_loss_mean(Dg, Lg, Rg, Dt, Le, Re, Sig, mu, aa=aa)
        loss = loss + lam_l1 * (Lt[:, 1:].abs().mean() + Rt[:, 1:].abs().mean())   # base L1 (E2's winner)
        if soft is not None and n_blocks > 1:                           # graded hierarchy: extra off-block L1
            loss = loss + soft * ((Lt * offblk).abs().mean() + (Rt * offblk).abs().mean())
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        if n_blocks == 1 or soft is not None:
            Le, Re = Lt, Rt
        else:
            Le, Re = Lt * hard, Rt * hard
        lf = float(fid_loss_mean(Dg, Lg, Rg, Dt, Le, Re, Sig, mu, aa=aa))     # under the TRAINING metric
        Si_, mi_ = full_support(d)
        lf_true = float(fid_loss_mean(Dg, Lg, Rg, Dt, Le, Re, Si_, mi_))      # TRUE (full-support) fidelity
        mm = rank1_mmcs(Dg, Lg, Rg, Dt, Le, Re)
        xo = torch.randn(8000, D_IN, generator=g, device=DEV)                 # OFF-distribution probe
        Xo = torch.cat([torch.ones(8000, 1, device=DEV), xo], 1)
        yo = forward(Dg, Lg, Rg, Xo)
        yh = ((Xo @ Le.T) * (Xo @ Re.T)) @ Dt.T
        mse_ood = float(((yh - yo) ** 2).sum(1).mean() / (yo ** 2).sum(1).mean())
    return dict(tsim_train=1 - lf, tsim_true=1 - lf_true, mmcs=mm, mse_ood=mse_ood)


if __name__ == "__main__":
    print("E3a  HIERARCHY SPECTRUM — does hierarchy help WHEN THE LAYER IS HIERARCHICAL?")
    print("     (data-free: L_fid only, full-support Λ; base L1 prior; rank 32; 5 seeds)\n")
    for gt_kind, desc in [("random", "gt = random 3-sparse support (does NOT respect blocks)"),
                          ("block", f"gt = BLOCK-structured ({N_BLK_GT} blocks; a genuinely hierarchical layer)")]:
        print(f"  --- {desc} ---")
        print(f"  {'n_blocks':>9s} {'HARD: tsim':>12s} {'HARD: recov':>13s} {'SOFT(s=.03): tsim':>18s} {'SOFT: recov':>12s}")
        for nb in [1, 2, 4, 8, 16]:
            hs = [fit(s, gt_kind, nb) for s in range(5)]
            ss = [fit(s, gt_kind, nb, soft=0.03) for s in range(5)]
            ht, hm = np.mean([h["tsim_true"] for h in hs]), np.mean([h["mmcs"] for h in hs])
            st, sm = np.mean([x["tsim_true"] for x in ss]), np.mean([x["mmcs"] for x in ss])
            tag = " (=dense)" if nb == 1 else (f" (=gt: {N_BLK_GT})" if nb == N_BLK_GT and gt_kind == "block" else "")
            print(f"  {nb:9d} {ht:12.3f} {hm:13.3f} {st:18.3f} {sm:12.3f}{tag}")
        print()

    print("E3b  METRIC TEMPERATURE  Sigma_t = (1-t)*Sigma_data + t*I   (FINDING 3's knob)")
    print("     Data on a 6-dim subspace (10 directions never probed). Train under Sigma_t;")
    print("     score under the FULL-SUPPORT metric (the TRUE global fidelity) + an OOD probe.\n")
    d = D_IN + 1
    Si, mi = full_support(d)
    print(f"  {'t':>5s} {'TRUE tsim':>11s} {'MSE(OOD)':>13s} {'gt-recov':>9s}   what t means")
    for t in [0.0, 0.01, 0.05, 0.2, 0.5, 1.0]:
        res = []
        for s in range(5):
            g = torch.Generator(device=DEV).manual_seed(s + 5)
            B = torch.linalg.qr(torch.randn(D_IN, 6, generator=g, device=DEV))[0]
            x = torch.randn(20000, 6, generator=g, device=DEV) @ B.T
            Sd, md = lifted_moments(x); Sd, md = Sd.to(DEV), md.to(DEV)
            St = (1 - t) * Sd + t * Si
            mt = (1 - t) * md + t * mi
            res.append(fit(s, "random", 1, Sig=St, mu=mt))
        tt = np.mean([r["tsim_true"] for r in res]); ttd = np.std([r["tsim_true"] for r in res])
        mo = np.mean([r["mse_ood"] for r in res]); mm = np.mean([r["mmcs"] for r in res])
        lab = "data-matched (BLIND)" if t == 0 else ("full-support (SAFE)" if t == 1 else "tempered")
        print(f"  {t:5.2f} {tt:6.3f}±{ttd:.3f} {mo:8.3f} {mm:12.3f}   {lab}")
    print("\n  TRUE tsim / MSE(OOD) are scored on ALL input directions. t=0 is the handoff's recipe.")
    print("DONE")
