"""DGP-A: gated linear experts. Mechanism label (gate g) and content label cross-cut by design.

x = [c ; s],  c ~ GMM(k_c) in R^{d_c},  s = eps * one_hot(g) in R^{k_g}
Hidden units indexed (g, j):
    L[(g,j), :] = e_{d_c+g}^T / eps        (reads the gate; 1/eps cancels the gate norm)
    R[(g,j), :] = [v_{g,j} ; 0]            (reads content)
    D[:, (g,j)] = u_{g,j}
=> y = D(Lx ⊙ Rx) = A_g c   with  A_g = sum_j u_{g,j} v_{g,j}^T.

Experts A_g are random ORTHOGONAL matrices: same output subspace, same norms, near-Frobenius-
orthogonal to each other. So output clustering cannot read the gate, and Jacobians of different
gates are near-orthogonal *in the content columns*.

WARNING (verified numerically in experiments/dgp_a.py): the exact Jacobian is
    J(x) = [ A_g   |   (1/eps) * F(c) ]
             ^content cols   ^gate cols
The gate-column block is O(1/eps) and is a function of c ALONE (independent of g). So at small eps
-- exactly the regime that makes input cosine blind to the gate -- J is dominated by a gate-
independent block, and Jacobian cosine recovers CONTENT, not mechanism. The writeup's P2 as stated
does not hold; the content-column-restricted Jacobian is the object that recovers the gate.
"""

import torch


def make_experts(k_g, d_c, gen, geometry="orthogonal"):
    """Return A_g of shape (k_g, d_c, d_c)."""
    if geometry == "orthogonal":
        return torch.stack([torch.linalg.qr(torch.randn(d_c, d_c, generator=gen))[0] for _ in range(k_g)])
    if geometry == "ring":
        # continuous one-parameter family: rotation by theta in the first plane
        A = []
        for t in range(k_g):
            th = 2 * torch.pi * t / k_g
            M = torch.eye(d_c)
            M[0, 0] = torch.cos(torch.tensor(th)); M[0, 1] = -torch.sin(torch.tensor(th))
            M[1, 0] = torch.sin(torch.tensor(th)); M[1, 1] = torch.cos(torch.tensor(th))
            A.append(M)
        return torch.stack(A)
    if geometry == "hierarchical":
        root = torch.linalg.qr(torch.randn(d_c, d_c, generator=gen))[0]
        out = []
        n_branch = 2
        branches = [torch.randn(d_c, d_c, generator=gen) for _ in range(n_branch)]
        for g in range(k_g):
            b = branches[g % n_branch]
            leaf = torch.randn(d_c, d_c, generator=gen)
            out.append(root + 0.6 * b / b.norm() * root.norm() + 0.2 * leaf / leaf.norm() * root.norm())
        return torch.stack(out)
    raise ValueError(geometry)


def build_layer(A, k_g, d_c, eps):
    """Hand-code (D, L, R) implementing y = A_{g(x)} c. Input dim d = d_c + k_g."""
    d = d_c + k_g
    Ds, Ls, Rs = [], [], []
    for g in range(k_g):
        U, S, Vh = torch.linalg.svd(A[g])
        for j in range(d_c):
            u = U[:, j] * S[j]
            v = Vh[j, :]
            l = torch.zeros(d); l[d_c + g] = 1.0 / eps
            r = torch.zeros(d); r[:d_c] = v
            Ds.append(u); Ls.append(l); Rs.append(r)
    return torch.stack(Ds, 1), torch.stack(Ls), torch.stack(Rs)


def sample(n, k_g, k_c, d_c, eps, gen, geometry="orthogonal", gate_content_corr=0.0):
    """Returns x, y, labels_gate, labels_content, (D,L,R), A."""
    A = make_experts(k_g, d_c, gen, geometry)
    D, L, R = build_layer(A, k_g, d_c, eps)
    centers = torch.randn(k_c, d_c, generator=gen) * 3.0
    lab_c = torch.randint(0, k_c, (n,), generator=gen)
    c = centers[lab_c] + torch.randn(n, d_c, generator=gen) * 0.5
    if gate_content_corr > 0:
        rnd = torch.randint(0, k_g, (n,), generator=gen)
        keep = torch.rand(n, generator=gen) < gate_content_corr
        lab_g = torch.where(keep, lab_c % k_g, rnd)
    else:
        lab_g = torch.randint(0, k_g, (n,), generator=gen)
    s = torch.zeros(n, k_g)
    s[torch.arange(n), lab_g] = eps
    x = torch.cat([c, s], 1)
    a = x @ L.T
    b = x @ R.T
    y = (a * b) @ D.T
    return x, y, lab_g, lab_c, (D, L, R), A
