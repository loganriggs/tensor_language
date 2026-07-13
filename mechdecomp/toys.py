"""Data-generating processes for Tiers 0-1 (spec §2)."""

import torch


def orthonormal_dgp(d_in=64, d_out=64, m_true=10, N=10_000, k_max=5, noise=0.01,
                    seed=0, structured_W=False, device="cpu"):
    """Tier 0 DGP: x_i = Σ_{j∈S_i} α_ij e_j, |S_i| ~ U{1..k_max}, α ~ U[0.5,1.5]."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(d_in, d_in, generator=g)).Q
    E = Q[:, :m_true]
    if structured_W:
        # distinct singular values per feature direction for identifiability tests
        U = torch.linalg.qr(torch.randn(d_out, d_out, generator=g)).Q
        s = torch.linspace(3.0, 0.3, m_true)
        W = U[:, :m_true] @ torch.diag(s) @ E.T + 0.05 * torch.randn(d_out, d_in, generator=g)
    else:
        W = torch.randn(d_out, d_in, generator=g) / d_in ** 0.5
    sizes = torch.randint(1, k_max + 1, (N,), generator=g)
    S = torch.zeros(N, m_true, dtype=torch.bool)
    X = torch.zeros(d_in, N)
    for i in range(N):
        idx = torch.randperm(m_true, generator=g)[:sizes[i]]
        S[i, idx] = True
        alpha = torch.rand(len(idx), generator=g) + 0.5
        X[:, i] = (E[:, idx] * alpha).sum(1)
    X = X + noise * torch.randn(d_in, N, generator=g)
    return W.to(device), X.to(device), E.to(device), S.to(device)


def correlated_pair_dgp(d_in=64, d_out=64, N=10_000, rho=1.0, anisotropic=False,
                        tied=False, noise=0.01, seed=0, device="cpu"):
    """Tier 1.1/1.2: features 0,1 co-occur with prob rho; W isotropic or strongly
    anisotropic on the pair. tied=True gives ONE shared coefficient (variation only
    along e0+e1 — the truly-merged case); tied=False varies coefficients
    independently (2-D pair-plane variation). 8 background features independent."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(d_in, d_in, generator=g)).Q
    E = Q[:, :10]
    if anisotropic:
        U = torch.linalg.qr(torch.randn(d_out, d_out, generator=g)).Q
        W = torch.randn(d_out, d_in, generator=g) / d_in ** 0.5
        # overwrite action on the pair: gain 4 vs 0.25, orthogonal output dirs
        W = W - W @ (E[:, :2] @ E[:, :2].T)
        W = W + 4.0 * U[:, 0:1] @ E[:, 0:1].T + 0.25 * U[:, 1:2] @ E[:, 1:2].T
    else:
        W = torch.randn(d_out, d_in, generator=g) / d_in ** 0.5
    S = torch.zeros(N, 10, dtype=torch.bool)
    X = torch.zeros(d_in, N)
    for i in range(N):
        pair = torch.rand(1, generator=g).item() < 0.4
        idx = []
        if pair:
            idx += [0, 1] if torch.rand(1, generator=g).item() < rho else \
                   ([0] if torch.rand(1, generator=g).item() < 0.5 else [1])
        nback = int(torch.randint(1, 4, (1,), generator=g))
        idx += (2 + torch.randperm(8, generator=g)[:nback]).tolist()
        idx = list(dict.fromkeys(idx))
        S[i, idx] = True
        alpha = torch.rand(len(idx), generator=g) + 0.5
        if tied and 0 in idx and 1 in idx:
            a = alpha[idx.index(0)]
            alpha[idx.index(1)] = a
        X[:, i] = (E[:, idx] * alpha).sum(1)
    X = X + noise * torch.randn(d_in, N, generator=g)
    return W.to(device), X.to(device), E.to(device), S.to(device)
