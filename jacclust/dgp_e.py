"""DGP-E: NO control stream. The gate must be inferred from the content itself.

This is the case that matters for real models: the whole input is `x`, there is no architectural
content/gate split, so no restricted Jacobian is available. Only the full end-to-end J exists.

    x = [ c ; 1 ]        c = a + v,   a in span{w_g} (gate part),  v in the orthogonal complement
                         mechanism label g = argmax_g (w_g . c)^2   (sign-invariant gate)
                         content label     = GMM component of v

    layer 1 (bilinear):  z = [ v ; (w_1.c)^2 ; ... ; (w_kg.c)^2 ; 1 ]
                         gate features are QUADRATIC -- a bilinear layer computes them natively
                         (unit with L=w_g, R=w_g); v and the constant are copied via the constant coord.

    layer 2 (bilinear):  y = sum_g  (w_g.c)^2 * (B_g v)      -- expert B_g weighted by its gate feature

Why this can escape the tick-3 impossibility theorem. For a SINGLE layer with a LINEAR gate readout
`p_g.c`, the gate-derivative term is `sum_g A_g c p_g^T`, summed over all experts, hence
gate-INdependent, and it dominates. Here the gate feature is quadratic, so the chain rule gives

    d y / d c  ⊃  sum_g B_g v * 2 (w_g.c) w_g^T

which is weighted by `(w_g.c)` -- i.e. it *does* depend on which gate is active. Depth + a nonlinear
gate should therefore restore mechanism information to the FULL Jacobian. That is the prediction.
"""

import torch


def _orth(n, gen):
    return torch.linalg.qr(torch.randn(n, n, generator=gen))[0]


def build(k_g, d_v, gen, gate_scale=1.0):
    d_c = k_g + d_v                       # c = [gate coords (k_g) ; content coords (d_v)] in an orth basis
    d_in = d_c + 1                        # trailing constant
    i_b = d_c
    W = torch.eye(d_c)[:k_g]              # w_g = e_g  (gate directions, wlog an orthonormal basis)
    B = torch.stack([_orth(d_v, gen) for _ in range(k_g)])

    # ---- layer 1: z = [ v ; (w_g.c)^2 ; 1 ] ----
    d_mid = d_v + k_g + 1
    j_gate, j_b = d_v, d_v + k_g
    Ls, Rs, Ds = [], [], []
    for t in range(d_v):                                   # copy v
        l = torch.zeros(d_in); l[i_b] = 1.0
        r = torch.zeros(d_in); r[k_g + t] = 1.0
        d = torch.zeros(d_mid); d[t] = 1.0
        Ls.append(l); Rs.append(r); Ds.append(d)
    for g in range(k_g):                                   # quadratic gate features
        l = torch.zeros(d_in); l[:d_c] = W[g] * gate_scale
        r = torch.zeros(d_in); r[:d_c] = W[g] * gate_scale
        d = torch.zeros(d_mid); d[j_gate + g] = 1.0
        Ls.append(l); Rs.append(r); Ds.append(d)
    l = torch.zeros(d_in); l[i_b] = 1.0                    # copy constant
    r = torch.zeros(d_in); r[i_b] = 1.0
    d = torch.zeros(d_mid); d[j_b] = 1.0
    Ls.append(l); Rs.append(r); Ds.append(d)
    l1 = (torch.stack(Ds, 1), torch.stack(Ls), torch.stack(Rs))

    # ---- layer 2: y = sum_g gate_g * (B_g v) ----
    Ls, Rs, Ds = [], [], []
    for g in range(k_g):
        U, S, Vh = torch.linalg.svd(B[g])
        for j in range(d_v):
            l = torch.zeros(d_mid); l[j_gate + g] = 1.0
            r = torch.zeros(d_mid); r[:d_v] = Vh[j]
            d = torch.zeros(d_v); d[:] = U[:, j] * S[j]
            Ls.append(l); Rs.append(r); Ds.append(d)
    l2 = (torch.stack(Ds, 1), torch.stack(Ls), torch.stack(Rs))
    lay = dict(d_c=d_c, d_v=d_v, k_g=k_g, d_in=d_in, d_mid=d_mid, i_b=i_b, j_gate=j_gate)
    return l1, l2, B, W, lay


def sample(n, k_g, k_c, d_v, gen, gate_amp=1.0, content_amp=3.0, gate_scale=1.0):
    l1, l2, B, W, lay = build(k_g, d_v, gen, gate_scale)
    d_c = lay["d_c"]
    # gate part: one dominant coordinate (so argmax is well defined) plus noise
    g = torch.randint(0, k_g, (n,), generator=gen)
    a = torch.randn(n, k_g, generator=gen) * 0.15 * gate_amp
    a[torch.arange(n), g] = gate_amp
    centers = torch.randn(k_c, d_v, generator=gen) * content_amp
    lc = torch.randint(0, k_c, (n,), generator=gen)
    v = centers[lc] + torch.randn(n, d_v, generator=gen) * 0.5
    x = torch.zeros(n, lay["d_in"])
    x[:, :k_g] = a
    x[:, k_g:d_c] = v
    x[:, lay["i_b"]] = 1.0
    return x, a, v, g, lc, l1, l2, B, W, lay


def fwd(layer, x):
    D, L, R = layer
    return ((x @ L.T) * (x @ R.T)) @ D.T


def jac(layer, x):
    D, L, R = layer
    a, b = L @ x, R @ x
    return D @ (a[:, None] * R + b[:, None] * L)
