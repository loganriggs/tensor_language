"""E2: structural-sparsity priors, fit with L_fid ALONE — fully DATA-FREE weight-space optimization.

Motivation (E1/FINDING 3): with a full-support Λ, L_fid alone reaches tensor-sim 1.000 using NO DATA — but it
lands on a DENSE, non-ground-truth factorization (gt-recovery 0.43) because an overcomplete CP decomposition is
non-unique. Hypothesis: a STRUCTURAL sparsity prior breaks that non-uniqueness, so pure weight-space fitting
should recover the planted sparse factors. If so: you can reverse-engineer a bilinear layer's sparse structure
with no data at all.

Priors on the transcoder factors (L',R',D'), all trained on L_fid ONLY (data-free, full-support Λ):
  dense       — no structure (E1 baseline)
  topk-rows   — hard: keep top-k entries per factor row after each step (learned support; projected gradient)
  L1          — soft row sparsity (L1 penalty on L',R')
  block       — HIERARCHICAL: input coords partitioned into blocks; each hidden unit reads only its block
                (fixed structured support; stays strictly inside the tensor-network class)
  symmetric   — l'_h = r'_h ("squared readout"): each feature is a squared linear form (eigendecomposable)

Metrics: tensor-sim (full-support = TRUE global fidelity), gt-recovery (MMCS to planted rank-1 terms),
measured factor sparsity. CONTROL: random-init recovery = chance. 5 seeds.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import fid_loss_mean, tensor_inner_mean
from e1_synthetic_recovery import make_gt, rank1_mmcs, D_IN, R_GT, K_OUT, K_SP, DEV

torch.set_default_dtype(torch.float32)
STEPS = 6000


def full_support(d):
    """Λ = lifted N(0,I): protects EVERY input direction (FINDING 3 — do NOT use a data-matched Σ)."""
    S = torch.eye(d, device=DEV); S[0, 0] = 0.0
    m = torch.zeros(d, device=DEV); m[0] = 1.0
    return S, m


def proj_topk_rows(M, k):
    """Hard-project each row to its top-k entries by magnitude (constant coord col 0 always kept free)."""
    body = M[:, 1:]
    thr = body.abs().topk(k, dim=1).values[:, -1:]
    M[:, 1:] = body * (body.abs() >= thr)
    return M


def block_mask(r, d, n_blocks, gen):
    """HIERARCHICAL prior: partition the d-1 input coords into n_blocks; each hidden unit reads one block."""
    body = d - 1
    coord_block = torch.arange(body, device=DEV) % n_blocks          # coord -> block
    unit_block = torch.randint(0, n_blocks, (r,), generator=gen, device=DEV)
    m = (coord_block[None, :] == unit_block[:, None]).float()        # (r, body)
    return torch.cat([torch.ones(r, 1, device=DEV), m], 1)           # constant coord always readable


def run(seed, prior, r_tc=32, lam_l1=3e-3):
    Dg, Lg, Rg = make_gt(seed)
    d = D_IN + 1
    g = torch.Generator(device=DEV).manual_seed(seed + 7)
    Sig, mu = full_support(d)
    aa = tensor_inner_mean(Dg, Lg, Rg, Dg, Lg, Rg, Sig, mu)

    Lt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Rt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Dt = (torch.randn(K_OUT, r_tc, generator=g, device=DEV) / r_tc ** .5).requires_grad_()
    params = [Lt, Dt] if prior == "symmetric" else [Lt, Rt, Dt]
    mask = block_mask(r_tc, d, 4, g) if prior == "block" else None
    mmcs0 = rank1_mmcs(Dg, Lg, Rg, Dt.detach(), Lt.detach(),
                       (Lt if prior == "symmetric" else Rt).detach())
    opt = torch.optim.Adam(params, 3e-3)

    for step in range(STEPS):
        Le = Lt * mask if mask is not None else Lt
        Re = Le if prior == "symmetric" else (Rt * mask if mask is not None else Rt)
        loss = fid_loss_mean(Dg, Lg, Rg, Dt, Le, Re, Sig, mu, aa=aa)     # DATA-FREE
        if prior == "L1":
            loss = loss + lam_l1 * (Lt[:, 1:].abs().mean() + Rt[:, 1:].abs().mean())
        loss.backward(); opt.step(); opt.zero_grad()
        if prior == "topk-rows":                                          # projected gradient
            with torch.no_grad():
                proj_topk_rows(Lt.data, K_SP); proj_topk_rows(Rt.data, K_SP)

    with torch.no_grad():
        Le = Lt * mask if mask is not None else Lt
        Re = Le if prior == "symmetric" else (Rt * mask if mask is not None else Rt)
        lf = float(fid_loss_mean(Dg, Lg, Rg, Dt, Le, Re, Sig, mu, aa=aa))
        mm = rank1_mmcs(Dg, Lg, Rg, Dt, Le, Re)
        # measured sparsity: frac of row mass in top-K_SP coords (gt = 1.000 by construction)
        def rowsp(M):
            v = M.abs()[:, 1:]
            return float((v.topk(K_SP, 1).values.sum(1) / v.sum(1).clamp_min(1e-9)).mean())
        fsp = 0.5 * (rowsp(Le) + rowsp(Re))
        # effective L0 per row (participation ratio of |row|)
        v = Le.abs()[:, 1:]
        l0 = float(((v.sum(1) ** 2) / (v ** 2).sum(1).clamp_min(1e-12)).mean())
    return dict(tsim=1 - lf, mmcs=mm, fsp=fsp, l0=l0, mmcs_rand=mmcs0)


if __name__ == "__main__":
    print(f"E2 structural priors — trained on L_fid ONLY (DATA-FREE, full-support Λ).")
    print(f"gt: rank {R_GT}, {K_SP}-sparse factors, d_in {D_IN}. transcoder rank 32. 5 seeds.\n")
    hdr = f"{'prior':12s} {'tensor-sim':>13s} {'gt-recovery':>14s} {'factor-sp':>10s} {'eff-L0/row':>11s}"
    print(hdr); print("-" * len(hdr))
    ctrl = []
    for prior in ["dense", "topk-rows", "L1", "block", "symmetric"]:
        rs = [run(s, prior) for s in range(5)]
        f = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
        ts, tsd = f("tsim"); mm, mmd = f("mmcs"); fs, _ = f("fsp"); l0, _ = f("l0")
        ctrl.append(f("mmcs_rand")[0])
        print(f"{prior:12s} {ts:+7.3f}±{tsd:.3f} {mm:8.3f}±{mmd:.3f} {fs:10.3f} {l0:11.1f}")
    print(f"\n  CONTROL random-init gt-recovery = {np.mean(ctrl):.3f}  (chance)")
    print(f"  gt has factor-sp 1.000 and eff-L0/row = {K_SP}.0 by construction.")

    print(f"\n  rank sweep for the sparse prior (can sim=1 coexist with sparsity?)")
    print(f"  {'r_tc':>5s} {'tensor-sim':>13s} {'gt-recovery':>14s}")
    for r_tc in [8, 16, 32, 64]:
        rs = [run(s, "topk-rows", r_tc=r_tc) for s in range(5)]
        ts = np.mean([r["tsim"] for r in rs]); tsd = np.std([r["tsim"] for r in rs])
        mm = np.mean([r["mmcs"] for r in rs]); mmd = np.std([r["mmcs"] for r in rs])
        print(f"  {r_tc:5d} {ts:+7.3f}±{tsd:.3f} {mm:8.3f}±{mmd:.3f}")
    print("DONE")
