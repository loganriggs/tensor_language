"""NON-CENTRAL degree-n Gaussian metric: E[∏ xᵀA_i x] for x ~ N(mu, Sigma), mu ≠ 0.

Needed because the REAL bilinear MLP's input is x = rms_norm(resid), whose mean DOMINATES it (E8: ||mu||=33.0
vs mean||x||=33.9 — the data is a tight cone around its mean, not a ball). Using the zero-mean formula there
would be FINDING 1 all over again, in a new place.

Derivation. For x ~ N(m, Σ) and T = Σ_i t_i A_i (A_i symmetric):

    log E[exp(xᵀTx)] = -½ log det(I - 2ΣT)  +  mᵀ T (I - 2ΣT)^{-1} m
                     = ½ Σ_{k≥1} (2^k/k) tr((ΣT)^k)   +   Σ_{k≥0} 2^k · mᵀ T (ΣT)^k m

Reading off the coefficient of ∏_{i∈b} t_i gives the joint cumulant of a block b of size n:

    κ(b) = 2^{n-1} [  Σ over the (n-1)! CYCLIC orderings   tr(M_{σ1} ⋯ M_{σn})            (Σ-only term)
                    + Σ over all n! orderings              mᵀ A_{σ1} Σ A_{σ2} ⋯ Σ A_{σn} m ]   (MEAN term)

with M_i = A_i Σ.  Setting m=0 recovers tensor_sim_deep exactly.  Sanity: n=1 gives tr(AΣ) + mᵀAm = E[xᵀAx]. ✓

    E[∏_{i=1}^n q_i] = Σ over set partitions π  ∏_{blocks b ∈ π} κ(b)

Σ may be SINGULAR — which is exactly what we need: a lifted input x̃=(1,a) is a degenerate Gaussian whose
constant coordinate has zero variance and mean 1. Everything below is verified against Monte Carlo in __main__.
"""
from __future__ import annotations
import itertools, torch


def _cumulant(As, Ms, Sigma, mu):
    """κ for one block. As/Ms: lists of (..., d, d). Returns (...,)."""
    n = len(As)
    tr = lambda M: torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)
    # --- Sigma-only term: (n-1)! cyclic orderings (fix element 0) ---
    if n == 1:
        sig_term = tr(Ms[0])
    else:
        sig_term = 0.0
        first, rest = Ms[0], Ms[1:]
        for perm in itertools.permutations(range(n - 1)):
            P = first
            for j in perm:
                P = P @ rest[j]
            sig_term = sig_term + tr(P)
    # --- MEAN term: all n! orderings of  mᵀ A_{σ1} Σ A_{σ2} ⋯ Σ A_{σn} m ---
    mean_term = 0.0
    for perm in itertools.permutations(range(n)):
        P = As[perm[0]]
        for j in perm[1:]:
            P = P @ Sigma @ As[j]
        mean_term = mean_term + torch.einsum("i,...ij,j->...", mu, P, mu)
    return (2 ** (n - 1)) * (sig_term + mean_term)


def _set_partitions(n):
    if n == 0:
        yield []
        return
    for part in _set_partitions(n - 1):
        for i in range(len(part)):
            yield part[:i] + [part[i] + [n - 1]] + part[i + 1:]
        yield part + [[n - 1]]


def expect_prod_quadratic_mean(As, Sigma, mu):
    """E[ ∏_i xᵀ A_i x ]  for x ~ N(mu, Sigma).  As: list of symmetric (..., d, d). Sigma may be singular."""
    Ms = [A @ Sigma for A in As]
    total = 0.0
    for part in _set_partitions(len(As)):
        term = 1.0
        for block in part:
            term = term * _cumulant([As[i] for i in block], [Ms[i] for i in block], Sigma, mu)
        total = total + term
    return total


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(0)
    OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
    fails = []
    d = 5
    rnd = lambda *s: torch.randn(*s, generator=g)
    sym = lambda M: 0.5 * (M + M.T)

    def case(tag, Sig, mu, n):
        As = [sym(rnd(d, d)) for _ in range(n)]
        L = torch.linalg.cholesky(Sig + 1e-12 * torch.eye(d)) if Sig.diagonal().min() > 0 else None
        # sample from N(mu, Sig) even when Sig is singular: use eigen-sqrt
        ev, V = torch.linalg.eigh(Sig)
        S = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
        x = torch.randn(6_000_000, d, generator=g) @ S.T + mu
        mc = torch.ones(x.shape[0])
        for A in As:
            mc = mc * ((x @ A) * x).sum(1)
        mc = float(mc.mean())
        cf = float(expect_prod_quadratic_mean(As, Sig, mu))
        rel = abs(cf - mc) / max(abs(mc), 1e-12)
        ok = rel < 0.03
        print(f"  [{OK if ok else BAD}] n={n} {tag:22s} closed {cf:+12.3f} vs MC {mc:+12.3f} (rel {rel:.1e})")
        if not ok: fails.append(f"{tag}-n{n}")

    Spd = (lambda M: M @ M.T / d + 0.5 * torch.eye(d))(rnd(d, d))
    m0, m1 = torch.zeros(d), rnd(d) * 1.5
    # a LIFTED (degenerate) Gaussian: coord 0 is deterministically 1 -> zero variance, mean 1
    Sl = Spd.clone(); Sl[0, :] = 0; Sl[:, 0] = 0
    ml = torch.zeros(d); ml[0] = 1.0; ml[1:] = rnd(d - 1) * 0.7

    for n in (2, 3, 4):
        case("Sigma=SPD, mu=0", Spd, m0, n)                 # must reduce to the zero-mean formula
        case("Sigma=SPD, mu!=0", Spd, m1, n)                # the non-central case
        case("LIFTED (singular Σ)", Sl, ml, n)              # the case the real model actually needs

    # cross-check: mu=0 must agree with the already-verified zero-mean implementation
    from tensor_sim_deep import expect_prod_quadratic
    As = [sym(rnd(d, d)) for _ in range(4)]
    a = float(expect_prod_quadratic_mean(As, Spd, m0)); b = float(expect_prod_quadratic(As, Spd))
    ok = abs(a - b) / abs(b) < 1e-12
    print(f"  [{OK if ok else BAD}] mu=0 reduces to tensor_sim_deep exactly  {a:+.6f} vs {b:+.6f}")
    if not ok: fails.append("reduction")
    print("RESULT:", "ALL PASS" if not fails else f"FAILURES {fails}")
