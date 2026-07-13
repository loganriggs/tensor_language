"""TIER 1 — recovery phase diagram with DEFINED ground truth (the only place an identifiability
claim can be tested, now that SAE-overlap is retired as an oracle).

Puzzle this resolves. Tier 0's recovery gate passes at min-cos 0.999. But refine_power.py showed
that from RANDOM init on a clean toy the same objective stalls at held-out 0.70 while the true
dictionary sits at 0.99 (cos-to-truth 0.52). Both cannot be the general case. Which knob breaks
recovery: overcompleteness K/d, sparsity k_true, noise, the E-step (OMP vs lasso), or dead atoms?

Ground truth: X = D_true C_true + noise, C_true nonneg k_true-sparse, Y = W X.
Recovery metric: for each TRUE atom, max |cos| to any learned atom (mean over true atoms).
  ~1.0 = recovered.  Chance level is reported alongside (random dicts of the same size).

Guards (standing rule — a control before any result is read):
  * init = D_true must STAY at cos 1.0 and not lose held-out R² (the fixed-point property that
    validates the M-step; refine_power.py established it).
  * chance-level recovery is measured, not assumed, so "0.52" is never read as "half recovered".

Run: python -m mechdecomp.tier1_recovery
"""

import torch
import torch.nn.functional as Fn

from .estep import solve_codes
from .mstep import resample_dead
from .refine_power import make_data, r2_of
from .release_d import omp_codes
from .mstep import rowspace_basis

DEV = "cuda"
torch.set_default_dtype(torch.float32)


def mstep_gs(W, D, C, Y, RS, Wpinv):
    """Validated Gauss-Seidel M-step (refine_power.py)."""
    WD = W @ D
    E = Y - WD @ C
    for j in range(D.shape[1]):
        act = C[j].abs() > 1e-10
        if act.sum() < 2:
            continue
        beta = C[j, act]
        Rj = E[:, act] + torch.outer(WD[:, j], beta)
        wd = (Rj @ beta) / (beta @ beta).clamp_min(1e-12)
        d = RS @ (RS.T @ (Wpinv @ wd))
        if d.norm() < 1e-8:
            continue
        D[:, j] = d
        wd = W @ d
        beta = (Rj.T @ wd) / (wd @ wd).clamp_min(1e-12)
        C[j, act] = beta
        WD[:, j] = wd
        E[:, act] = Rj - torch.outer(wd, beta)
    nrm = D.norm(dim=0).clamp_min(1e-12)
    return D / nrm, C * nrm[:, None]


def recovery(D, Dt):
    """mean over TRUE atoms of max |cos| to any learned atom."""
    M = (Fn.normalize(D, dim=0).T @ Fn.normalize(Dt, dim=0)).abs()   # (K_learn, K_true)
    return float(M.max(0).values.mean())


def chance_recovery(d_in, K_learn, K_true, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = Fn.normalize(torch.randn(d_in, K_learn, generator=g), dim=0).to(DEV)
    B = Fn.normalize(torch.randn(d_in, K_true, generator=g), dim=0).to(DEV)
    return recovery(A, B)


def run(K, ktrue, noise, estep="omp", rounds=20, resample=False, init="random", seed=0):
    Dt, W, Xtr, Xva = make_data_K(noise, K, ktrue, seed)
    Ytr, Yva = W @ Xtr, W @ Xva
    RS = rowspace_basis(W); Wpinv = torch.linalg.pinv(W)
    if init == "true":
        D = Dt.clone()
    else:
        g = torch.Generator(device="cpu").manual_seed(seed + 100)
        D = Fn.normalize(torch.randn(Dt.shape[0], K, generator=g), dim=0).to(DEV)
    for _ in range(rounds):
        WD = W @ D
        if estep == "omp":
            C, _ = omp_codes(WD, Ytr, ktrue)
        else:
            C = solve_codes(W, D, Xtr, lam=0.02)
            C = C * (D.T @ Xtr)          # effective coeffs, so both E-steps share the M-step
        if resample:
            D, _ = resample_dead(W, D, C, Xtr)
            WD = W @ D
        D, C = mstep_gs(W, D, C, Ytr, RS, Wpinv)
    WDv = W @ D
    Cv, _ = omp_codes(WDv, Yva, ktrue)
    return recovery(D, Dt), r2_of(WDv @ Cv, Yva)


def make_data_K(noise, K, ktrue, seed):
    import mechdecomp.refine_power as rp
    K_old, kt_old = rp.K, rp.KTRUE
    rp.K, rp.KTRUE = K, ktrue
    try:
        out = make_data(noise=noise, seed=seed)
    finally:
        rp.K, rp.KTRUE = K_old, kt_old
    return out


def main():
    d_in = 64
    print("GUARD: init = D_true must stay put (validates the M-step before any row is read)\n", flush=True)
    rec, r2 = run(K=256, ktrue=4, noise=0.05, init="true", rounds=6)
    assert rec > 0.99, f"GUARD FAIL: true-init recovery decayed to {rec:.3f}"
    print(f"  [guard ok] init=D_true  recovery {rec:.4f}  held-out R2 {r2:.4f}\n", flush=True)

    print("PHASE DIAGRAM (init = random, 20 rounds, OMP E-step)\n", flush=True)
    print("    K   K/d  ktrue  noise   recovery   chance   held-out R2", flush=True)
    for K in (64, 128, 256):
        for ktrue in (2, 4):
            ch = chance_recovery(d_in, K, K)
            rec, r2 = run(K=K, ktrue=ktrue, noise=0.05)
            print(f"  {K:3d}  {K/d_in:4.1f}   {ktrue:3d}   0.05    {rec:.4f}   {ch:.4f}    {r2:.4f}", flush=True)

    print("\nDOES THE E-STEP MATTER?  (K=128, ktrue=4)", flush=True)
    for es in ("omp", "lasso"):
        rec, r2 = run(K=128, ktrue=4, noise=0.05, estep=es)
        print(f"  E-step {es:6s}   recovery {rec:.4f}   held-out R2 {r2:.4f}", flush=True)

    print("\nDOES DEAD-ATOM RESAMPLING RESCUE IT?  (K=256, ktrue=4, OMP)", flush=True)
    for rs in (False, True):
        rec, r2 = run(K=256, ktrue=4, noise=0.05, resample=rs, rounds=20)
        print(f"  resample={str(rs):5s}   recovery {rec:.4f}   held-out R2 {r2:.4f}", flush=True)

    print("\nMORE ROUNDS?  (K=256, ktrue=4, OMP, random init)", flush=True)
    for R in (20, 60):
        rec, r2 = run(K=256, ktrue=4, noise=0.05, rounds=R)
        print(f"  rounds={R:3d}   recovery {rec:.4f}   held-out R2 {r2:.4f}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
