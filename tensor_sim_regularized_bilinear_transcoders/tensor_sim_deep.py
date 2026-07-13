"""Degree-4 tensor-similarity: the Gaussian inner product for a TWO-LAYER bilinear network.

Stacking bilinear layers is where "hierarchy across layers" lives:
    z = D1((L1 x) ⊙ (R1 x))          degree 2 in x
    y = D2((L2 z) ⊙ (R2 z))          degree 2 in z  =>  degree 4 in x

Collapse layer 2 onto x.  (l2_g · z) = Σ_p α_gp (l1_p·x)(r1_p·x) = xᵀ Q_g x   with α = L2 D1,
Q_g = Σ_p α_gp l1_p r1_pᵀ  (only its symmetric part matters).  Likewise (r2_g·z) = xᵀ P_g x, β = R2 D1.
So the network is a sum of PRODUCTS OF TWO QUADRATIC FORMS:

    y_k = Σ_g D2_kg (xᵀ Q_g x)(xᵀ P_g x)

Hence ⟨y|Λ|ŷ⟩ = E[y·ŷ] needs the Gaussian expectation of a product of FOUR quadratic forms:

    ⟨y|Λ|ŷ⟩ = Σ_k Σ_{g,g'} D2_kg D̂2_kg' · E[ q(Q_g) q(P_g) q(Q̂_g') q(P̂_g') ]

E[∏_{i=1}^n q(A_i)] for x~N(0,Σ) is a sum over SET PARTITIONS of {1..n}; each block contributes a joint
cumulant.  With M_i := A_i Σ (A_i symmetric), from  -½ log det(I - 2T) = ½ Σ_m (2^m/m) tr(T^m):

    κ(block of size k) = 2^{k-1} · Σ over the (k-1)! cyclic orderings of  tr(M_{σ(1)} … M_{σ(k)})

    E[∏ q_i] = Σ_{set partitions π}  ∏_{blocks b ∈ π} κ(b)

For n=4 that's the 15 set partitions of {1,2,3,4}. Implemented generically below and VERIFIED against
Monte Carlo (see __main__) — this is exactly the kind of formula that is silently wrong if unchecked.
"""
from __future__ import annotations
import itertools, torch


def _cumulant(Ms):
    """κ for one block: 2^(k-1) * Σ over (k-1)! cyclic orderings of tr(prod)."""
    k = len(Ms)
    if k == 1:
        return torch.diagonal(Ms[0], dim1=-2, dim2=-1).sum(-1)
    tot = 0.0
    first, rest = Ms[0], Ms[1:]
    for perm in itertools.permutations(range(k - 1)):          # (k-1)! cyclic orderings (fix element 0)
        P = first
        for j in perm:
            P = P @ rest[j]
        tot = tot + torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)
    return (2 ** (k - 1)) * tot


def _set_partitions(n):
    """All set partitions of {0..n-1} (Bell(n); 15 for n=4)."""
    if n == 0:
        yield []
        return
    for part in _set_partitions(n - 1):
        for i in range(len(part)):
            yield part[:i] + [part[i] + [n - 1]] + part[i + 1:]
        yield part + [[n - 1]]


def expect_prod_quadratic(As, Sigma):
    """E[ ∏_i xᵀ A_i x ]  for x ~ N(0, Sigma).  As: list of (d,d) symmetric (batched leading dims OK)."""
    Ms = [A @ Sigma for A in As]
    total = 0.0
    for part in _set_partitions(len(As)):
        term = 1.0
        for block in part:
            term = term * _cumulant([Ms[i] for i in block])
        total = total + term
    return total


def collapse(D1, L1, R1, L2, R2):
    """Fold layer 2 onto x: returns Q,P of shape (r2, d, d), symmetric. y_k = Σ_g D2_kg (xᵀQ_g x)(xᵀP_g x)."""
    a = L2 @ D1                                     # (r2, r1)
    b = R2 @ D1                                     # (r2, r1)
    outer = L1.unsqueeze(2) * R1.unsqueeze(1)       # (r1, d, d)  = l1_p r1_p^T
    Q = torch.einsum("gp,pij->gij", a, outer)
    P = torch.einsum("gp,pij->gij", b, outer)
    sym = lambda M: 0.5 * (M + M.transpose(-1, -2))
    return sym(Q), sym(P)


def deep_inner(D1, L1, R1, D2, L2, R2, D1h, L1h, R1h, D2h, L2h, R2h, Sigma):
    """⟨y|Λ|ŷ⟩ = E[y·ŷ] for two 2-layer bilinear nets, x ~ N(0,Sigma). Closed form, data-free."""
    Q, P = collapse(D1, L1, R1, L2, R2)              # (r2,d,d)
    Qh, Ph = collapse(D1h, L1h, R1h, L2h, R2h)       # (r2',d,d)
    W = D2.T @ D2h                                   # (r2, r2')  = Σ_k D2_kg D2h_kg'
    r2, r2h = Q.shape[0], Qh.shape[0]
    tot = 0.0
    for g in range(r2):                              # small r2 in toys; O(r2 r2' d^3)
        As_g = [Q[g], P[g]]
        for gp in range(r2h):
            e = expect_prod_quadratic(As_g + [Qh[gp], Ph[gp]], Sigma)
            tot = tot + W[g, gp] * e
    return tot


def deep_forward(D1, L1, R1, D2, L2, R2, x):
    z = ((x @ L1.T) * (x @ R1.T)) @ D1.T
    return ((z @ L2.T) * (z @ R2.T)) @ D2.T


def deep_fid(D1, L1, R1, D2, L2, R2, D1h, L1h, R1h, D2h, L2h, R2h, Sigma, aa=None):
    """L_fid = ||A-Ahat||²_Λ / ||A||²_Λ = E||y-ŷ||²/E||y||²  for the 2-layer composite."""
    args_a = (D1, L1, R1, D2, L2, R2)
    args_b = (D1h, L1h, R1h, D2h, L2h, R2h)
    if aa is None:
        aa = deep_inner(*args_a, *args_a, Sigma)
    ab = deep_inner(*args_a, *args_b, Sigma)
    bb = deep_inner(*args_b, *args_b, Sigma)
    return (aa - 2 * ab + bb) / aa


if __name__ == "__main__":
    # ---- VERIFY against Monte Carlo (standing rule: never trust an unchecked moment formula) ----
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(0)
    d, r1, dz, r2, K = 5, 4, 3, 3, 2      # x:d -> layer1 hidden:r1 -> z:dz -> layer2 hidden:r2 -> y:K
    OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
    fails = []

    def rnd(*s): return torch.randn(*s, generator=g)
    # shapes: L1,R1 (r1,d)  D1 (dz,r1)  L2,R2 (r2,dz)  D2 (K,r2)
    D1, L1, R1, D2, L2, R2 = rnd(dz, r1), rnd(r1, d), rnd(r1, d), rnd(K, r2), rnd(r2, dz), rnd(r2, dz)
    r1h, r2h = 6, 5                        # transcoder: different (overcomplete) ranks
    D1h, L1h, R1h = rnd(dz, r1h), rnd(r1h, d), rnd(r1h, d)
    D2h, L2h, R2h = rnd(K, r2h), rnd(r2h, dz), rnd(r2h, dz)

    for tag, Sig in [("Σ=I", torch.eye(d)), ("Σ=SPD", (lambda M: M @ M.T / d + .5 * torch.eye(d))(rnd(d, d)))]:
        n = 4_000_000
        x = torch.randn(n, d, generator=g) @ torch.linalg.cholesky(Sig).T
        mc = float((deep_forward(D1, L1, R1, D2, L2, R2, x) *
                    deep_forward(D1h, L1h, R1h, D2h, L2h, R2h, x)).sum(1).mean())
        cf = float(deep_inner(D1, L1, R1, D2, L2, R2, D1h, L1h, R1h, D2h, L2h, R2h, Sig))
        rel = abs(cf - mc) / max(abs(mc), 1e-30)
        ok = rel < 0.03
        print(f"  [{OK if ok else BAD}] deg-4 ⟨y|Λ|ŷ⟩ closed==MC [{tag}]  closed {cf:+.2f} vs MC {mc:+.2f} (rel {rel:.1e})")
        if not ok: fails.append(tag)
        lf = float(deep_fid(D1, L1, R1, D2, L2, R2, D1h, L1h, R1h, D2h, L2h, R2h, Sig))
        y, yh = deep_forward(D1, L1, R1, D2, L2, R2, x), deep_forward(D1h, L1h, R1h, D2h, L2h, R2h, x)
        rel_mse = float(((y - yh) ** 2).sum(1).mean() / (y ** 2).sum(1).mean())
        ok2 = abs(lf - rel_mse) / rel_mse < 0.04
        print(f"  [{OK if ok2 else BAD}] deg-4 L_fid == E‖y-ŷ‖²/E‖y‖² [{tag}]  {lf:.4f} vs {rel_mse:.4f}")
        if not ok2: fails.append(tag + "-fid")
        lf0 = float(deep_fid(D1, L1, R1, D2, L2, R2, D1, L1, R1, D2, L2, R2, Sig))
        ok3 = abs(lf0) < 1e-10
        print(f"  [{OK if ok3 else BAD}] deg-4 L_fid(A,A)=0 [{tag}]  {lf0:.2e}")
        if not ok3: fails.append(tag + "-self")
    print("RESULT:", "ALL PASS" if not fails else f"FAILURES {fails}")
