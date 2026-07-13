"""DGP-C: two stacked bilinear layers, joint mechanism label (g1, g2).

x = [ c ; s1 ; s2 ; 1 ]        c in R^{d_c},  s1,s2 = eps * one_hot,  trailing constant 1
layer 1:  z = [ A_{g1} c ; s2 ; 1 ]        (experts on content; s2 and the constant copied forward)
layer 2:  y =   B_{g2} (A_{g1} c)

Copies are exact in a bilinear layer because of the constant coordinate: a hidden unit with
L = e_const and R = e_target computes 1 * target.

Predictions (writeup §5, DGP-C):
  layer-1 restricted Jacobian  -> recovers g1 only
  layer-2 restricted Jacobian  -> recovers g2 only
  end-to-end restricted Jacobian -> recovers the joint (g1, g2)

"Restricted" = Jacobian w.r.t. the content columns. Established in tick 2 that the full Jacobian of
ANY single bilinear layer is dominated by a gate-independent d/d(gate) term, so the restricted object
is the only one that can carry mechanism. Here the restriction is architectural (content vs control
streams), not a peek at the answer.
"""

import torch


def _orth(n, gen):
    return torch.linalg.qr(torch.randn(n, n, generator=gen))[0]


def build(k_g, d_c, eps, gen):
    """Return (D1,L1,R1), (D2,L2,R2), A, B and the index layout."""
    A = torch.stack([_orth(d_c, gen) for _ in range(k_g)])
    B = torch.stack([_orth(d_c, gen) for _ in range(k_g)])
    d_in = d_c + 2 * k_g + 1                      # [c | s1 | s2 | 1]
    i_s1, i_s2, i_b = d_c, d_c + k_g, d_c + 2 * k_g
    d_mid = d_c + k_g + 1                         # [Ac | s2 | 1]
    j_s2, j_b = d_c, d_c + k_g

    # ---- layer 1 ----
    Ls, Rs, Ds = [], [], []
    for g in range(k_g):
        U, S, Vh = torch.linalg.svd(A[g])
        for j in range(d_c):
            l = torch.zeros(d_in); l[i_s1 + g] = 1.0 / eps
            r = torch.zeros(d_in); r[:d_c] = Vh[j]
            d = torch.zeros(d_mid); d[:d_c] = U[:, j] * S[j]
            Ls.append(l); Rs.append(r); Ds.append(d)
    for t in range(k_g + 1):                       # copy s2 (k_g coords) and the constant
        src = i_s2 + t if t < k_g else i_b
        dst = j_s2 + t if t < k_g else j_b
        l = torch.zeros(d_in); l[i_b] = 1.0
        r = torch.zeros(d_in); r[src] = 1.0
        d = torch.zeros(d_mid); d[dst] = 1.0
        Ls.append(l); Rs.append(r); Ds.append(d)
    L1, R1, D1 = torch.stack(Ls), torch.stack(Rs), torch.stack(Ds, 1)

    # ---- layer 2 ----
    Ls, Rs, Ds = [], [], []
    for g in range(k_g):
        U, S, Vh = torch.linalg.svd(B[g])
        for j in range(d_c):
            l = torch.zeros(d_mid); l[j_s2 + g] = 1.0 / eps
            r = torch.zeros(d_mid); r[:d_c] = Vh[j]
            d = torch.zeros(d_c); d[:] = U[:, j] * S[j]
            Ls.append(l); Rs.append(r); Ds.append(d)
    L2, R2, D2 = torch.stack(Ls), torch.stack(Rs), torch.stack(Ds, 1)
    layout = dict(d_c=d_c, k_g=k_g, d_in=d_in, d_mid=d_mid, i_s1=i_s1, i_s2=i_s2, i_b=i_b)
    return (D1, L1, R1), (D2, L2, R2), A, B, layout


def sample(n, k_g, k_c, d_c, eps, gen):
    l1, l2, A, B, lay = build(k_g, d_c, eps, gen)
    centers = torch.randn(k_c, d_c, generator=gen) * 3.0
    lc = torch.randint(0, k_c, (n,), generator=gen)
    c = centers[lc] + torch.randn(n, d_c, generator=gen) * 0.5
    g1 = torch.randint(0, k_g, (n,), generator=gen)
    g2 = torch.randint(0, k_g, (n,), generator=gen)
    x = torch.zeros(n, lay["d_in"])
    x[:, :d_c] = c
    x[torch.arange(n), lay["i_s1"] + g1] = eps
    x[torch.arange(n), lay["i_s2"] + g2] = eps
    x[:, lay["i_b"]] = 1.0
    return x, c, g1, g2, lc, l1, l2, A, B, lay


def fwd(layer, x):
    D, L, R = layer
    return ((x @ L.T) * (x @ R.T)) @ D.T


def jac(layer, x):
    D, L, R = layer
    a, b = L @ x, R @ x
    return D @ (a[:, None] * R + b[:, None] * L)
