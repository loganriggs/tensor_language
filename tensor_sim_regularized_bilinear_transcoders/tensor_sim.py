"""Closed-form Gaussian tensor inner product for bilinear layers (CP form), per the tensor-similarity paper.

A bilinear layer  y_k = Σ_ij A_kij x̃_i x̃_j   with the CP weight tensor
    A_kij = Σ_h D_kh L_hi R_hj      D∈R^{K×r}, L,R∈R^{r×(d+1)}
i.e. y = D( (L x̃) ⊙ (R x̃) ).

METRIC.  Define the *functional* inner product under Gaussian inputs x̃ ~ N(0, G):
    ⟨A|Λ|Â⟩ := E_x̃[ y(x̃) · ŷ(x̃) ]
Isserlis (Wick) for zero-mean Gaussians:  E[x̃_i x̃_j x̃_a x̃_b] = G_ij G_ab + G_ia G_jb + G_ib G_ja, so
    ⟨A|Λ|Â⟩ = Σ_k [ tr_G(A_k)·tr_G(Â_k) + 2 ⟨A_k^sym , Â_k^sym⟩_G ],   tr_G(M) := ⟨M,G⟩.
In CP factors this is four matmuls + two Hadamards (never build the K×d×d tensor):
    tr_G(A_k)   = Σ_h D_kh (l_h^T G r_h)            ->  t  = D @ rowdot(L G, R)
    E∥_{hg}     = (L G L'^T)_{hg} (R G R'^T)_{hg}
    E×_{hg}     = (L G R'^T)_{hg} (R G L'^T)_{hg}
    ⟨A|Λ|Â⟩     = t·t' + Σ_{hg} (D^T D')_{hg} (E∥ + E×)_{hg}
Cost O(r r' d). Fully differentiable, zero sampling variance, data-free per step.

CONSEQUENCE (the thing to test against):  since ⟨·|Λ|·⟩ IS E[y·ŷ],
    L_fid(A,Â) = ‖A-Â‖²_Λ / ‖A‖²_Λ = E‖y-ŷ‖² / E‖y‖²      (relative Gaussian MSE, in closed form)

GAUGE.  The CP form has a gauge group of *permutation × rescaling* of the hidden index h
(L_h→a L_h, R_h→b R_h, D_:,h→D_:,h/(ab)); these leave A itself invariant. In addition the metric
(not the tensor) is invariant to swapping L↔R, which transposes each slice: a quadratic form only
sees the symmetric part, so the layer's FUNCTION is unchanged. A general invertible U on the hidden
index is NOT a gauge of a CP tensor — see sanity_checks.py, which asserts it breaks.
"""
from __future__ import annotations
import torch


def _prep(G, L, R):
    """Return (L@G, R@G); G=None means identity."""
    if G is None:
        return L, R
    return L @ G, R @ G


def tensor_inner(D1, L1, R1, D2, L2, R2, G=None):
    """⟨A|Λ|Â⟩ = E_{x̃~N(0,G)}[ y·ŷ ], closed form from CP factors. G: (d+1,d+1) or None (=I)."""
    L1G, R1G = _prep(G, L1, R1)        # L1@G, R1@G  (G applied once, on side 1)
    L2G, _ = _prep(G, L2, R2)
    # traces:  tr_G(A_k) = Σ_h D_kh (l_h^T G r_h)
    t1 = D1 @ (L1G * R1).sum(1)                        # (K,)
    t2 = D2 @ (L2G * R2).sum(1)                        # (K,)
    # E∥_{hg} = (L1 G L2^T)(R1 G R2^T),  E×_{hg} = (L1 G R2^T)(R1 G L2^T)   [r × r']
    Epar = (L1G @ L2.T) * (R1G @ R2.T)
    Ecrs = (L1G @ R2.T) * (R1G @ L2.T)
    return t1 @ t2 + ((D1.T @ D2) * (Epar + Ecrs)).sum()


def tensor_inner_mean(D1, L1, R1, D2, L2, R2, Sigma, mu):
    """⟨A|Λ|Â⟩ = E_{x̃~N(mu,Sigma)}[y·ŷ] — NON-CENTRAL Wick. USE THIS FOR LIFTED INPUTS x̃=(1,x).

    WHY: the zero-mean Isserlis formula (tensor_inner) is WRONG for a lifted input, because x̃ is not
    zero-mean: its constant coordinate is deterministic. Plugging the *uncentered* second moment
    E[x̃x̃ᵀ] into the zero-mean formula predicts E[x̃₀⁴]=3·Σ₀₀²=3 when the truth is 1 — a silent,
    catastrophic bias (measured >1500% error). But x̃=(1,x) IS still a (degenerate) Gaussian with
    mean mu=(1,μ_x) and CENTERED covariance Sigma (zero row/col on the constant coord), so the
    non-central 4th-moment formula is exact:

      E[x_i x_j x_a x_b] = Σ_ij Σ_ab + Σ_ia Σ_jb + Σ_ib Σ_ja
                         + μ_iμ_j Σ_ab + μ_aμ_b Σ_ij
                         + μ_iμ_a Σ_jb + μ_iμ_b Σ_ja + μ_jμ_a Σ_ib + μ_jμ_b Σ_ia
                         + μ_iμ_jμ_aμ_b

    Contracting with the CP factors gives one r×r' kernel (same O(r r' d) cost as the centered form):
      a_h = l_hᵀΣr_h,  m_h = (l_h·μ)(r_h·μ),  and the four Σ-bilinears LL,RR,LR,RL.
    Reduces to tensor_inner() when mu=0.  Pass Sigma = CENTERED covariance, mu = mean.
    """
    LS, RS = L1 @ Sigma, R1 @ Sigma
    LL, RR = LS @ L2.T, RS @ R2.T                       # (r,r')
    LR, RL = LS @ R2.T, RS @ L2.T
    a1 = (LS * R1).sum(1)                               # (r,)   l_hᵀ Σ r_h
    a2 = ((L2 @ Sigma) * R2).sum(1)                     # (r',)
    l1m, r1m = L1 @ mu, R1 @ mu                         # (r,)
    l2m, r2m = L2 @ mu, R2 @ mu                         # (r',)
    m1, m2 = l1m * r1m, l2m * r2m
    K = (torch.outer(a1, a2) + LL * RR + LR * RL
         + torch.outer(m1, a2) + torch.outer(a1, m2)
         + torch.outer(l1m, l2m) * RR + torch.outer(l1m, r2m) * RL
         + torch.outer(r1m, l2m) * LR + torch.outer(r1m, r2m) * LL
         + torch.outer(m1, m2))
    return ((D1.T @ D2) * K).sum()


def fid_loss_mean(D, L, R, Dh, Lh, Rh, Sigma, mu, aa=None):
    """L_fid under the non-central (lifted-input) metric = E‖y-ŷ‖²/E‖y‖²."""
    if aa is None:
        aa = tensor_inner_mean(D, L, R, D, L, R, Sigma, mu)
    ab = tensor_inner_mean(D, L, R, Dh, Lh, Rh, Sigma, mu)
    bb = tensor_inner_mean(Dh, Lh, Rh, Dh, Lh, Rh, Sigma, mu)
    return (aa - 2 * ab + bb) / aa


def lifted_moments(x):
    """From raw inputs x (n,d_in) -> (Sigma, mu) of the lifted x̃=(1,x), ready for tensor_inner_mean.
    Sigma is the CENTERED covariance (its constant-coord row/col are 0); mu[0]=1."""
    xt = torch.cat([torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device), x], 1)
    mu = xt.mean(0)
    xc = xt - mu
    Sigma = (xc.T @ xc) / xt.shape[0]
    return Sigma, mu


def norm_sq(D, L, R, G=None):
    """‖A‖²_Λ = E‖y‖²."""
    return tensor_inner(D, L, R, D, L, R, G)


def fid_loss(D, L, R, Dh, Lh, Rh, G=None, aa=None):
    """L_fid = ‖A-Â‖²_Λ / ‖A‖²_Λ = E‖y-ŷ‖²/E‖y‖².  Pass precomputed aa=⟨A|Λ|A⟩ to save work."""
    if aa is None:
        aa = norm_sq(D, L, R, G)
    ab = tensor_inner(D, L, R, Dh, Lh, Rh, G)
    bb = norm_sq(Dh, Lh, Rh, G)
    return (aa - 2 * ab + bb) / aa


def cosine_sim(D, L, R, Dh, Lh, Rh, G=None):
    """Scale-invariant similarity (report as eval; do NOT train on this — see handoff)."""
    ab = tensor_inner(D, L, R, Dh, Lh, Rh, G)
    aa = norm_sq(D, L, R, G)
    bb = norm_sq(Dh, Lh, Rh, G)
    return ab / (aa.clamp_min(1e-30) * bb.clamp_min(1e-30)).sqrt()


# ---- reference (slow) implementations, used ONLY by the sanity checks ----

def build_tensor(D, L, R):
    """Explicit A_kij = Σ_h D_kh L_hi R_hj. O(K d²) memory — tiny cases only."""
    return torch.einsum("kh,hi,hj->kij", D, L, R)


def tensor_inner_bruteforce(A, Ah, G=None):
    """⟨A|Λ|Â⟩ by explicit Isserlis contraction on the full tensors (no CP structure used)."""
    d = A.shape[-1]
    if G is None:
        G = torch.eye(d, dtype=A.dtype, device=A.device)
    trA = torch.einsum("kij,ij->k", A, G)
    trB = torch.einsum("kij,ij->k", Ah, G)
    term1 = trA @ trB
    # Σ_k Σ_ijab A_kij Â_kab (G_ia G_jb + G_ib G_ja)
    t2 = torch.einsum("kij,kab,ia,jb->", A, Ah, G, G)
    t3 = torch.einsum("kij,kab,ib,ja->", A, Ah, G, G)
    return term1 + t2 + t3


def forward(D, L, R, x):
    """y = D((L x) ⊙ (R x)); x is (n, d+1) already lifted."""
    return ((x @ L.T) * (x @ R.T)) @ D.T
