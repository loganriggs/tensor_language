"""E10: a TWO-LAYER transcoder fit to a REAL bilinear MLP, under the REAL (ridged) metric. Data-free.

Two things F5/FINDING 10 got wrong are fixed here:
  (1) the METRIC. F5 used Lambda = N(0,I). E8 measured the real input distribution: effective dim 23.5 of 1152,
      and ||mu||=33.0 vs mean||x||=33.9 — the data is a tight CONE AROUND ITS MEAN. So we use the ridged real
      metric Sigma_t = (1-t)Sigma_real + t*(tr/d)I (FINDING 6, t=0.05) with the real mean (non-central Wick,
      tensor_sim_deep_mean.py, MC-verified).
  (2) the ARCHITECTURE. A flat rank-r' transcoder can only be a shallower copy of the layer. A TWO-layer
      transcoder asks a different question: can the layer be re-expressed as a COMPOSITION — a few mid-level
      features, recombined? That is what "hierarchy across layers" means for a real model.

THE QUESTION: at a MATCHED PARAMETER BUDGET, does a 2-layer transcoder explain a real bilinear MLP better than
a flat one? If yes, the layer has compositional structure. If no, it does not — and that is a real answer too.

Setup. Work on the data manifold: the layer projected onto its top-96 input PCA directions (E8), lifted to
x~=(1,a), d=97. The target is then a flat CP layer (Lp,Rp,Dp). The 2-layer transcoder lifts z as well
(z~=(1,z)), so it can express degrees < 4 — without that it could not even represent a degree-2 target.

Key trick that makes the cross term cheap: <deep|Lambda|flat> = sum_k sum_g D2[k,g] E[q(Q_g)q(P_g)q(A_k)] is
LINEAR in A_k, so fold first: Abar_g = sum_k D2[k,g] A_k. Cost drops from K*r2' (=9216) triples to r2' (=8).
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language")
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim_deep_mean import expect_prod_quadratic_mean as EQ

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MET = "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders/real_metric.pt"
STEPS, SEEDS, RIDGE_T = 1200, 3, 0.05
sym = lambda M: 0.5 * (M + M.transpose(-1, -2))


def load():
    M = torch.load(MET, weights_only=False)
    Lp, Rp, Dp = M["Lp"].to(DEV), M["Rp"].to(DEV), M["Dp"].to(DEV)     # (r,d) (r,d) (K,r), d=97
    d = Lp.shape[1]
    Sig = M["Sig_lift"].to(DEV).clone()                                  # lifted: row/col 0 = 0
    blk = Sig[1:, 1:]
    Sig[1:, 1:] = (1 - RIDGE_T) * blk + RIDGE_T * (torch.diagonal(blk).sum() / blk.shape[0]) \
                  * torch.eye(blk.shape[0], device=DEV)                  # RIDGE (FINDING 6)
    mu = M["mu_lift"].to(DEV)                                            # e_0  (x~ = (1,a), E[a]=0)
    # target as one symmetric quadratic form per output: y_k = x~^T A_k x~
    A = torch.einsum("kh,hij->kij", Dp, sym(Lp.unsqueeze(2) * Rp.unsqueeze(1)))     # (K,d,d)
    return Lp, Rp, Dp, A, Sig, mu, d, M


def flat_quads(Df, Lf, Rf):
    return torch.einsum("kh,hij->kij", Df, sym(Lf.unsqueeze(2) * Rf.unsqueeze(1)))


def collapse_lifted(D1, L1, R1, L2c, L2z, d):
    """z~=(1,z) with z=D1((L1 x~)*(R1 x~)).  L2 z~ = L2c*1 + L2z·z  =  x~^T Q x~   (the constant is absorbed
    into the x~_0 coordinate, since x~_0 = 1)."""
    a = L2z @ D1                                                        # (r2, r1)
    Q = torch.einsum("gp,pij->gij", a, sym(L1.unsqueeze(2) * R1.unsqueeze(1)))      # (r2,d,d)
    e0 = torch.zeros(d, d, device=DEV); e0[0, 0] = 1.0
    return Q + L2c[:, None, None] * e0


def deep_parts(p, d):
    D1, L1, R1, D2, L2c, L2z, R2c, R2z = p
    Q = collapse_lifted(D1, L1, R1, L2c, L2z, d)
    P = collapse_lifted(D1, L1, R1, R2c, R2z, d)
    return Q, P, D2


def fit_deep(A, aa, Sig, mu, d, dz, r1, r2, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    K = A.shape[0]
    rnd = lambda *s: (torch.randn(*s, generator=g, device=DEV) / s[-1] ** .5).requires_grad_()
    p = [rnd(dz, r1), rnd(r1, d), rnd(r1, d), rnd(K, r2), rnd(r2), rnd(r2, dz), rnd(r2), rnd(r2, dz)]
    opt = torch.optim.Adam(p, 3e-3)
    for _ in range(STEPS):
        Q, P, D2 = deep_parts(p, d)
        Ab = torch.einsum("kg,kij->gij", D2, A)                          # fold the target into r2 slices
        ab = EQ([Q, P, Ab], Sig, mu).sum()
        W = D2.T @ D2                                                    # (r2,r2)
        n = Q.shape[0]
        ex = lambda M: M.unsqueeze(1).expand(-1, n, -1, -1).reshape(-1, d, d)
        exh = lambda M: M.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, d, d)
        bb = (W.reshape(-1) * EQ([ex(Q), ex(P), exh(Q), exh(P)], Sig, mu)).sum()
        loss = (aa - 2 * ab + bb) / aa
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        Q, P, D2 = deep_parts(p, d)
        Ab = torch.einsum("kg,kij->gij", D2, A)
        ab = EQ([Q, P, Ab], Sig, mu).sum()
        W = D2.T @ D2; n = Q.shape[0]
        ex = lambda M: M.unsqueeze(1).expand(-1, n, -1, -1).reshape(-1, d, d)
        exh = lambda M: M.unsqueeze(0).expand(n, -1, -1, -1).reshape(-1, d, d)
        bb = (W.reshape(-1) * EQ([ex(Q), ex(P), exh(Q), exh(P)], Sig, mu)).sum()
        return 1 - float((aa - 2 * ab + bb) / aa), sum(t.numel() for t in p)


def fit_flat(A, aa, Sig, mu, d, r, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    K = A.shape[0]
    rnd = lambda *s: (torch.randn(*s, generator=g, device=DEV) / s[-1] ** .5).requires_grad_()
    Df, Lf, Rf = rnd(K, r), rnd(r, d), rnd(r, d)
    opt = torch.optim.Adam([Df, Lf, Rf], 3e-3)
    for _ in range(STEPS):
        B = flat_quads(Df, Lf, Rf)
        ab = EQ([A, B], Sig, mu).sum(); bb = EQ([B, B], Sig, mu).sum()
        loss = (aa - 2 * ab + bb) / aa
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        B = flat_quads(Df, Lf, Rf)
        ab = EQ([A, B], Sig, mu).sum(); bb = EQ([B, B], Sig, mu).sum()
        return 1 - float((aa - 2 * ab + bb) / aa), Df.numel() + Lf.numel() + Rf.numel()


if __name__ == "__main__":
    Lp, Rp, Dp, A, Sig, mu, d, M = load()
    print("E10  TWO-LAYER transcoder on a REAL bilinear MLP, under the REAL (ridged) metric — DATA-FREE\n")
    print(f"  target: L8 bilinear MLP of a 500M bilinear GPT, projected to its top-{M['D_EFF']} input PCA dirs")
    print(f"          (retains {100*M['var_kept']:.0f}% of input variance and {M['out_kept']:.3f} of the layer's")
    print(f"           output on real inputs — the projected layer IS the target, stated as an approximation)")
    print(f"  metric: real Sigma, effective dim {M['eff_dim']:.1f} of 1152, ridged with t={RIDGE_T}; real mean.")
    print(f"          (F5 used N(0,I) — scoring the layer on ~1128 directions the model never visits.)\n")
    aa = EQ([A, A], Sig, mu).sum().detach()
    print(f"  ||A||^2_Lambda = {float(aa):.4e}\n")

    print("  FLAT (1-layer) transcoder — the F5 sweep, now under the CORRECT metric")
    print(f"  {'r′':>6s} {'params':>9s} {'tensor-sim':>14s}")
    flat = []
    for r in [8, 16, 32, 64, 128]:
        v = [fit_flat(A, aa, Sig, mu, d, r, s) for s in range(SEEDS)]
        ts = np.mean([x[0] for x in v]); sd = np.std([x[0] for x in v])
        flat.append((v[0][1], ts))
        print(f"  {r:6d} {v[0][1]:9d} {ts:9.3f}±{sd:.3f}")

    print("\n  DEEP (2-layer) transcoder — bottleneck sweep (r1′=64, r2′=16)")
    print(f"  {'dz′':>6s} {'params':>9s} {'tensor-sim':>14s}")
    deep = []
    for dz in [1, 2, 4, 8, 16, 32]:
        v = [fit_deep(A, aa, Sig, mu, d, dz, 64, 16, s) for s in range(SEEDS)]
        ts = np.mean([x[0] for x in v]); sd = np.std([x[0] for x in v])
        deep.append((v[0][1], ts))
        print(f"  {dz:6d} {v[0][1]:9d} {ts:9.3f}±{sd:.3f}")

    print("\n  MATCHED-PARAMETER COMPARISON (does depth buy anything on a REAL layer?)")
    print(f"  {'params':>9s} {'best FLAT':>11s} {'best DEEP':>11s}   verdict")
    for lo, hi in [(0, 30_000), (30_000, 80_000), (80_000, 200_000), (200_000, 10 ** 9)]:
        f = [t for p, t in flat if lo <= p < hi]
        dp = [t for p, t in deep if lo <= p < hi]
        if not f or not dp: continue
        bf, bd = max(f), max(dp)
        v = "DEEP wins" if bd > bf + .01 else ("FLAT wins" if bf > bd + .01 else "tie")
        print(f"  {lo:9d}+ {bf:11.3f} {bd:11.3f}   {v}")
    print("DONE")
