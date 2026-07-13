"""E6: the Pareto frontier — sparsity (TopK k) x fidelity weight (lambda) -> L0 vs tensor-sim vs MSE.

The paper's story, stated as a falsifiable claim: an MSE-only transcoder can look GREAT on MSE while its
measured tensor-similarity is LOW. Test it on the honest regime (data on a 6-dim subspace of R^16 — FINDING 3
showed an isotropic toy hides the entire effect), with the metric's Lambda full-support (FINDING 3/6).

CONFOUND, stated up front and then measured away:
  L_fid is a closed form for a CP TENSOR. A BatchTopK transcoder is NOT a CP tensor — the gate makes it
  piecewise. So the closed-form "tensor-sim" scores the transcoder's UNDERLYING DENSE tensor, not the deployed
  gated model. Reporting only that would be cheating in the fidelity term's favour. So we ALSO report
  `gated global sim` = 1 - E||y-yhat_gated||^2/E||y||^2 by MONTE CARLO over the FULL space x~N(0,I), with the
  TopK gate applied — the honest global fidelity of the model you would actually deploy. Both are reported.

5 seeds. Controls: random-init recovery (chance), and the k=32(=rank) column where TopK is a no-op.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import fid_loss_mean, tensor_inner_mean, forward
from e1_synthetic_recovery import (make_gt, rank1_mmcs, batch_topk, make_basis, sample_x,
                                   D_IN, R_GT, K_OUT, K_SP, DEV)

torch.set_default_dtype(torch.float32)
R_TC, N_TR, N_TE, STEPS, SEEDS = 32, 20000, 8000, 4000, 5
KS = [1, 2, 4, 8, 32]                       # 32 = rank => gate is a no-op (control)
LAMS = [0.0, 0.1, 1.0, 10.0]                # 0.0 = MSE-only baseline
lift = lambda x: torch.cat([torch.ones(x.shape[0], 1, device=DEV), x], 1)


def gate(h, k):
    """BatchTopK, with the k>=rank case (gate is a no-op) handled — batch_topk's kthvalue underflows there."""
    return h if k >= h.shape[1] else batch_topk(h, k)


def run(seed, ktop, lam, fid_only=False):
    Dg, Lg, Rg = make_gt(seed)
    d = D_IN + 1
    g = torch.Generator(device=DEV).manual_seed(seed + 99)
    B = make_basis(g, "subspace")                                  # data on a 6-dim subspace (the real regime)
    Xtr, Xte = lift(sample_x(N_TR, g, B)), lift(sample_x(N_TE, g, B))
    Xood = lift(torch.randn(N_TE, D_IN, generator=g, device=DEV))  # full-space probe
    Sig = torch.eye(d, device=DEV); Sig[0, 0] = 0.0                # FULL-SUPPORT Lambda (FINDING 3)
    mu = torch.zeros(d, device=DEV); mu[0] = 1.0
    ytr, yte, yood = forward(Dg, Lg, Rg, Xtr), forward(Dg, Lg, Rg, Xte), forward(Dg, Lg, Rg, Xood)
    aa = tensor_inner_mean(Dg, Lg, Rg, Dg, Lg, Rg, Sig, mu)

    Lt = (torch.randn(R_TC, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Rt = (torch.randn(R_TC, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Dt = (torch.randn(K_OUT, R_TC, generator=g, device=DEV) / R_TC ** .5).requires_grad_()
    mm0 = rank1_mmcs(Dg, Lg, Rg, Dt.detach(), Lt.detach(), Rt.detach())
    opt = torch.optim.Adam([Lt, Rt, Dt], 3e-3)
    for _ in range(STEPS):
        loss = 0.0
        if not fid_only:
            bi = torch.randint(0, N_TR, (2048,), generator=g, device=DEV)
            h = (Xtr[bi] @ Lt.T) * (Xtr[bi] @ Rt.T)
            yh = gate(h, ktop) @ Dt.T
            loss = loss + ((yh - ytr[bi]) ** 2).sum(1).mean() / (ytr[bi] ** 2).sum(1).mean()
        if fid_only or lam > 0:
            lf = fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Sig, mu, aa=aa)     # DATA-FREE, on the dense tensor
            loss = loss + (lf if fid_only else lam * lf)
        loss.backward(); opt.step(); opt.zero_grad()

    with torch.no_grad():
        tsim = 1 - float(fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Sig, mu, aa=aa))   # dense tensor, closed form
        fwd = lambda X: gate((X @ Lt.T) * (X @ Rt.T), ktop) @ Dt.T
        rel = lambda yh, y: float(((yh - y) ** 2).sum(1).mean() / (y ** 2).sum(1).mean())
        mse_in = rel(fwd(Xte), yte)                                # held-out, in-distribution, GATED
        gsim = 1 - rel(fwd(Xood), yood)                            # HONEST global sim of the GATED model (MC)
        return dict(tsim=tsim, gsim=gsim, mse_in=mse_in, mm=rank1_mmcs(Dg, Lg, Rg, Dt, Lt, Rt), mm0=mm0)


if __name__ == "__main__":
    print("E6  PARETO: TopK k  x  fidelity weight lambda.  Data on a 6-dim subspace of R^16; Lambda full-support.")
    print(f"    transcoder rank {R_TC}, gt rank {R_GT}. {SEEDS} seeds. lambda=0 is the MSE-only baseline.\n")
    print("    tensor-sim  = closed-form, on the transcoder's DENSE tensor")
    print("    gated-sim   = MONTE-CARLO global fidelity of the DEPLOYED (TopK-gated) model  <- the honest one")
    print("    MSE(in)     = held-out relative MSE, in-distribution, gated\n")
    ctrl, rows = [], []
    for ktop in KS:
        tag = " (gate is a no-op)" if ktop >= R_TC else ""
        print(f"  --- TopK k = {ktop}{tag} ---")
        hdr = f"  {'lambda':>8s} {'MSE(in)':>9s} {'tensor-sim':>13s} {'gated-sim':>13s} {'gt-recov':>13s}"
        print(hdr)
        for lam in LAMS:
            rs = [run(s, ktop, lam) for s in range(SEEDS)]
            f = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
            mi, _ = f("mse_in"); ts, td = f("tsim"); gs, gd = f("gsim"); mm, md = f("mm")
            ctrl.append(f("mm0")[0])
            note = "  <- MSE-only" if lam == 0 else ""
            print(f"  {lam:8.1f} {mi:9.3f} {ts:8.3f}±{td:.3f} {gs:8.3f}±{gd:.3f} {mm:8.3f}±{md:.3f}{note}")
            rows.append((ktop, lam, mi, ts, gs, mm))
        print()
    print("  --- L_fid ONLY (no data at all), scored under each gate ---")
    print(f"  {'k':>8s} {'MSE(in)':>9s} {'tensor-sim':>13s} {'gated-sim':>13s} {'gt-recov':>13s}")
    for ktop in KS:
        rs = [run(s, ktop, 0.0, fid_only=True) for s in range(SEEDS)]
        f = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
        mi, _ = f("mse_in"); ts, td = f("tsim"); gs, gd = f("gsim"); mm, md = f("mm")
        print(f"  {ktop:8d} {mi:9.3f} {ts:8.3f}±{td:.3f} {gs:8.3f}±{gd:.3f} {mm:8.3f}±{md:.3f}")
    print(f"\n  CONTROL random-init gt-recovery = {np.mean(ctrl):.3f} (chance)")
    print("DONE")
