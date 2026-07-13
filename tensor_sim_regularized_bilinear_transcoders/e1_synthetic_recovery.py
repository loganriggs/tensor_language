"""E1: synthetic recovery — can a sparse ground-truth CP structure be recovered, and by which loss?

Ground truth: rank-r bilinear layer with SPARSE CP factors (each hidden unit reads k_sp input coords).
Transcoder: overcomplete rank r' >> r, RANDOM init. It never sees (D,L,R) — it accesses the original layer
only through (i) forward passes (MSE) and (ii) the closed-form data-free L_fid. That is the "hiding"
(we also apply a random perm+rescale gauge to the presented layer; by FINDING 2 that's the true CP gauge).

Arms:
  (a) MSE + BatchTopK only        — data-dependent; sparse but only on-distribution
  (b) L_fid only                  — DATA-FREE weight-space; globally faithful but overcomplete => non-unique
  (c) MSE + BatchTopK + lambda*L_fid

Hypothesis (handoff): (c) recovers the ground-truth factors; neither (a) nor (b) alone does.
L_fid is computed on the DENSE transcoder tensor (the mask is not part of the multilinear model).

Metrics (held-out): tensor-sim (1 - L_fid, and cosine), relative MSE with TopK, ground-truth factor
recovery (MMCS over rank-1 terms, L<->R orientation allowed), learned-factor sparsity, and the
handoff's open question: is the DENSE hidden code still sparse without the mask?
CONTROL: a random (untrained) transcoder gives the chance level of the recovery metric.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import fid_loss_mean, cosine_sim, lifted_moments, forward, tensor_inner_mean

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_IN, R_GT, K_OUT, K_SP = 16, 8, 12, 3          # input dim, gt rank, out dim, nonzeros per factor row
R_TC, KTOP = 32, 4                               # transcoder rank (overcomplete), BatchTopK avg-k
N_TR, N_TE, STEPS = 20000, 8000, 4000


def make_gt(seed):
    """Sparse ground-truth CP layer on lifted inputs (d = D_IN+1)."""
    g = torch.Generator().manual_seed(seed); d = D_IN + 1
    L = torch.zeros(R_GT, d); R = torch.zeros(R_GT, d)
    for h in range(R_GT):                        # each hidden unit reads K_SP coords (of the non-constant part)
        iL = torch.randperm(D_IN, generator=g)[:K_SP] + 1
        iR = torch.randperm(D_IN, generator=g)[:K_SP] + 1
        L[h, iL] = torch.randn(K_SP, generator=g); R[h, iR] = torch.randn(K_SP, generator=g)
    D = torch.randn(K_OUT, R_GT, generator=g)
    # apply the TRUE CP gauge (perm + rescale) to the presented layer — must not change anything
    p = torch.randperm(R_GT, generator=g)
    a = torch.rand(R_GT, generator=g) + .5; b = torch.rand(R_GT, generator=g) + .5
    L, R, D = (a[:, None] * L)[p], (b[:, None] * R)[p], (D / (a * b)[None, :])[:, p]
    return D.to(DEV), L.to(DEV), R.to(DEV)


def rank1_mmcs(D1, L1, R1, D2, L2, R2):
    """Mean over gt terms of max cosine to any transcoder term. Rank-1 term cos = (d.d')(l.l')(r.r'),
    allowing the L<->R orientation swap (an exact gauge of the function; FINDING 2)."""
    n = lambda M: M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
    d1, d2 = n(D1.T), n(D2.T)                    # (r, K)
    l1, r1, l2, r2 = n(L1), n(R1), n(L2), n(R2)
    dd = (d1 @ d2.T).abs()
    direct = (l1 @ l2.T).abs() * (r1 @ r2.T).abs()
    swap = (l1 @ r2.T).abs() * (r1 @ l2.T).abs()
    sim = dd * torch.maximum(direct, swap)
    return float(sim.max(1).values.mean())


def batch_topk(h, k):
    nk = h.shape[0] * k; flat = h.abs().reshape(-1)
    thr = flat.kthvalue(flat.numel() - nk).values
    return h * (h.abs() > thr)


def make_basis(g, mode, d_in=D_IN, sub=6):
    """The data subspace, built ONCE per run (train and test must share it — a fresh basis per call
    would silently make the 'in-distribution' test set off-distribution)."""
    if mode == "iso":
        return None
    return torch.linalg.qr(torch.randn(d_in, sub, generator=g, device=DEV))[0]   # (d_in, sub) orthonormal


def sample_x(n, g, B, d_in=D_IN):
    """iso (B=None): x~N(0,I) probes every direction.  subspace: x lies in span(B), so the data NEVER
    probes the orthogonal directions — the regime where off-distribution mechanisms (backdoors) hide."""
    if B is None:
        return torch.randn(n, d_in, generator=g, device=DEV)
    return torch.randn(n, B.shape[1], generator=g, device=DEV) @ B.T


def run(seed, arm, lam=1.0, mode="iso", metric="data"):
    """metric: which Lambda the fidelity loss uses.
       'data'  = (Sigma,mu) of the observed inputs  (the handoff recipe)
       'iso'   = full-support reference N(0,I) lifted  -> protects UNPROBED directions."""
    Dg, Lg, Rg = make_gt(seed)
    d = D_IN + 1
    g = torch.Generator(device=DEV).manual_seed(seed + 99)
    B = make_basis(g, mode)
    xtr = sample_x(N_TR, g, B)
    xte = sample_x(N_TE, g, B)
    xood = torch.randn(N_TE, D_IN, generator=g, device=DEV)       # OFF-distribution probe (full space)
    lift = lambda x: torch.cat([torch.ones(x.shape[0], 1, device=DEV), x], 1)
    Xtr, Xte, Xood = lift(xtr), lift(xte), lift(xood)
    Sig_d, mu_d = lifted_moments(xtr)                                 # data-matched (non-central!)
    Sig_d, mu_d = Sig_d.to(DEV), mu_d.to(DEV)
    # full-support reference metric: x ~ N(0,I) lifted  (constant coord has zero variance, mean 1)
    Sig_i = torch.eye(d, device=DEV); Sig_i[0, 0] = 0.0
    mu_i = torch.zeros(d, device=DEV); mu_i[0] = 1.0
    Sig, mu = (Sig_d, mu_d) if metric == "data" else (Sig_i, mu_i)    # metric used for the LOSS
    ytr, yte = forward(Dg, Lg, Rg, Xtr), forward(Dg, Lg, Rg, Xte)
    yood = forward(Dg, Lg, Rg, Xood)
    aa = tensor_inner_mean(Dg, Lg, Rg, Dg, Lg, Rg, Sig, mu)          # precompute <A|L|A>

    Lt = (torch.randn(R_TC, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Rt = (torch.randn(R_TC, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Dt = (torch.randn(K_OUT, R_TC, generator=g, device=DEV) / R_TC ** .5).requires_grad_()
    mmcs0 = rank1_mmcs(Dg, Lg, Rg, Dt.detach(), Lt.detach(), Rt.detach())   # CONTROL: random init = chance
    opt = torch.optim.Adam([Lt, Rt, Dt], 3e-3)
    for step in range(STEPS):
        bi = torch.randint(0, N_TR, (2048,), generator=g, device=DEV)
        loss = 0.0
        if arm in ("mse", "both"):
            h = (Xtr[bi] @ Lt.T) * (Xtr[bi] @ Rt.T)
            yh = batch_topk(h, KTOP) @ Dt.T
            loss = loss + ((yh - ytr[bi]) ** 2).sum(1).mean() / (ytr[bi] ** 2).sum(1).mean()
        if arm in ("fid", "both"):
            lf = fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Sig, mu, aa=aa)      # DATA-FREE, dense Ahat
            loss = loss + (lam * lf if arm == "both" else lf)
        loss.backward(); opt.step(); opt.zero_grad()

    with torch.no_grad():
        lf = float(fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Sig, mu, aa=aa))          # in the training metric
        lf_iso = float(fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Sig_i, mu_i))         # FULL-SUPPORT fidelity
        cs = float(cosine_sim(Dg, Lg, Rg, Dt, Lt, Rt, None))                # eval-only (scale-invariant)
        hte = (Xte @ Lt.T) * (Xte @ Rt.T)
        yh_topk = batch_topk(hte, KTOP) @ Dt.T
        mse_topk = float(((yh_topk - yte) ** 2).sum(1).mean() / (yte ** 2).sum(1).mean())
        yh_dense = hte @ Dt.T
        mse_dense = float(((yh_dense - yte) ** 2).sum(1).mean() / (yte ** 2).sum(1).mean())
        hood = (Xood @ Lt.T) * (Xood @ Rt.T)
        mse_ood = float(((hood @ Dt.T - yood) ** 2).sum(1).mean() / (yood ** 2).sum(1).mean())
        # is the DENSE code sparse without the mask? (handoff open question) — participation ratio
        a = hte.abs(); pr = float(((a.sum(1) ** 2) / (a ** 2).sum(1).clamp_min(1e-12)).mean())
        mmcs = rank1_mmcs(Dg, Lg, Rg, Dt, Lt, Rt)
        # learned factor sparsity: fraction of mass in top-K_SP coords per row
        def rowsp(M):
            v = M.abs()[:, 1:]                                              # ignore constant coord
            top = v.topk(K_SP, dim=1).values.sum(1); return float((top / v.sum(1).clamp_min(1e-9)).mean())
        fsp = 0.5 * (rowsp(Lt) + rowsp(Rt))
    return dict(L_fid=lf, tsim=1 - lf, tsim_iso=1 - lf_iso, cos=cs, mse_topk=mse_topk, mse_dense=mse_dense, mse_ood=mse_ood,
                dense_PR=pr, mmcs=mmcs, mmcs_rand=mmcs0, factor_sp=fsp)


if __name__ == "__main__":
    print(f"E1. gt rank {R_GT}, {K_SP}-sparse factors, d_in {D_IN}. transcoder rank {R_TC} "
          f"(x{R_TC//R_GT}), BatchTopK k={KTOP}. 5 seeds.\n")
    for mode, desc in [("iso", "x ~ N(0,I) — data probes EVERY direction"),
                       ("subspace", "x on a 6-dim SUBSPACE — 10 directions NEVER probed (backdoor regime)")]:
        print(f"=== data: {desc} ===")
        hdr = (f"{'arm':22s} {'tsim(train-metric)':>18s} {'tsim(FULL-SUPPORT)':>19s} "
               f"{'MSE(in)':>8s} {'MSE(OOD)':>13s} {'gt-recov':>9s}")
        print(hdr); print("-" * len(hdr))
        arms = [("mse", "data", "MSE+TopK"), ("fid", "data", "L_fid(data-matched)"),
                ("fid", "iso", "L_fid(full-support)"), ("both", "data", "MSE+fid(data-matched)"),
                ("both", "iso", "MSE+fid(full-support)")]
        for arm, met, name in arms:
            if mode == "iso" and met == "iso" and arm != "fid":
                continue                                   # iso data: the two metrics coincide; skip dupes
            rs = [run(s_, arm, mode=mode, metric=met) for s_ in range(5)]
            f = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
            ts, _ = f("tsim"); ti, tid = f("tsim_iso"); mi, _ = f("mse_dense")
            mo, mod = f("mse_ood"); mm, mmd = f("mmcs")
            print(f"{name:22s} {ts:18.3f} {ti:13.3f}±{tid:.3f} {mi:8.3f} {mo:7.3f}±{mod:.3f} {mm:5.3f}±{mmd:.3f}")
        print(f"  CONTROL random-init gt-recovery = {run(0,'mse',mode=mode)['mmcs_rand']:.3f} (chance)\n")
    print("tsim(FULL-SUPPORT) = true global fidelity (all input directions). MSE(OOD) = relative MSE on")
    print("full-space inputs. KEY: a data-matched metric on subspace data is BLIND to unprobed directions.")
    print("DONE")
