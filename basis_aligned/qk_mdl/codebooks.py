"""Codebooks 1, 2, 5 from the spec menu (§2), each returning
(dl_bits, achieved_distortion, meta) for a target distortion eps under the
frozen conventions in mdl_accounting.py. Distortion = relative Frobenius^2.

Implemented this tick: svd (baseline), bicluster (cross-associations flavor,
MDL model selection over k), toeplitz (diagonal profile, Fourier-truncated).
Pending (tick 3+): HODLR/tree (needs ordering machinery), sparse bilinear
dictionary + conjunction (needs the masked-projector solver).
"""

import math

import torch

from mdl_accounting import (dl_svd, dl_bicluster, dl_toeplitz_fourier,
                            dl_toeplitz_full)


def _fvu(Mhat, M):
    return float(((Mhat - M) ** 2).sum() / (M ** 2).sum())


@torch.no_grad()
def fit_svd(M, eps):
    """Smallest rank with FVU <= eps."""
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    tot = float((S ** 2).sum())
    tail = 1.0 - torch.cumsum(S ** 2, 0) / tot
    r = int((tail > eps).sum()) + 1
    r = min(r, len(S))
    Mhat = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]
    return dl_svd(r, *M.shape), _fvu(Mhat, M), {'rank': r}


@torch.no_grad()
def _kmeans_labels(X, k, iters=25, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    lab = torch.zeros(len(X), dtype=torch.long, device=X.device)
    for _ in range(iters):
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
        lab = d2.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab


@torch.no_grad()
def _bicluster_once(M, k_r, k_c, iters=30, seed=0, init='random'):
    if init == 'spectral':
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        r = min(k_r, len(S))
        rows = _kmeans_labels(U[:, :r] * S[:r], k_r, seed=seed)
        cols = _kmeans_labels(Vt[:r].T * S[:r], k_c, seed=seed)
    else:
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        rows = torch.randint(0, k_r, (M.shape[0],), generator=g).to(M.device)
        cols = torch.randint(0, k_c, (M.shape[1],), generator=g).to(M.device)
    for _ in range(iters):
        # block means given assignments
        B = torch.zeros(k_r, k_c, dtype=M.dtype, device=M.device)
        cnt = torch.zeros(k_r, k_c, dtype=M.dtype, device=M.device)
        B.index_put_((rows[:, None].expand_as(M), cols[None, :].expand_as(M)),
                     M, accumulate=True)
        cnt.index_put_((rows[:, None].expand_as(M), cols[None, :].expand_as(M)),
                       torch.ones_like(M), accumulate=True)
        B = B / cnt.clamp(min=1)
        # reassign rows: cost[i,a] = sum_j (M[i,j] - B[a,c_j])^2, expanded form
        Bc = B[:, cols]                                   # (k_r, n_cols)
        cost_r = ((M ** 2).sum(1)[:, None] - 2 * M @ Bc.T
                  + (Bc ** 2).sum(1)[None, :])            # (n_rows, k_r)
        rows = cost_r.argmin(1)
        # reassign cols: cost[j,b] = sum_i (M[i,j] - B[r_i,b])^2
        Br = B[rows, :]                                   # (n_rows, k_c)
        cost_c = ((M ** 2).sum(0)[:, None] - 2 * M.T @ Br
                  + (Br ** 2).sum(0)[None, :])            # (n_cols, k_c)
        cols = cost_c.argmin(1)
    B = torch.zeros(k_r, k_c, dtype=M.dtype, device=M.device)
    cnt = torch.zeros_like(B)
    B.index_put_((rows[:, None].expand_as(M), cols[None, :].expand_as(M)),
                 M, accumulate=True)
    cnt.index_put_((rows[:, None].expand_as(M), cols[None, :].expand_as(M)),
                   torch.ones_like(M), accumulate=True)
    B = B / cnt.clamp(min=1)
    return _fvu(B[rows][:, cols], M), (rows, cols, B)


@torch.no_grad()
def fit_bicluster(M, eps, k_max=256, restarts=2):
    """Double k until FVU <= eps (cross-associations flavor: separate row/col
    partitions, MDL-selected k = smallest that meets eps)."""
    k = 2
    best = None
    while k <= k_max:
        fvu = min([_bicluster_once(M, k, k, init='spectral')[0]]
                  + [_bicluster_once(M, k, k, seed=s)[0] for s in range(restarts)])
        best = (dl_bicluster(k, k, *M.shape), fvu, {'k': k})
        if fvu <= eps:
            return best
        k *= 2
    return best[0], best[1], {**best[2], 'hit_kmax': True}


@torch.no_grad()
def fit_toeplitz(M, eps):
    """c(delta) = diagonal means; then smallest Fourier truncation within eps."""
    n_r, n_c = M.shape
    idx = torch.arange(n_r, device=M.device)[:, None] - \
        torch.arange(n_c, device=M.device)[None, :] + (n_c - 1)   # 0..n_r+n_c-2
    n_d = n_r + n_c - 1
    sums = torch.zeros(n_d, dtype=M.dtype, device=M.device)
    cnts = torch.zeros_like(sums)
    sums.index_put_((idx,), M, accumulate=True)
    cnts.index_put_((idx,), torch.ones_like(M), accumulate=True)
    c = sums / cnts.clamp(min=1)
    fvu_full = _fvu(c[idx], M)
    if fvu_full > eps:
        return dl_toeplitz_full(n_r, n_c), fvu_full, {'mode': 'full', 'fail': True}
    # Fourier-truncate the diagonal profile
    C = torch.fft.rfft(c)
    order = C.abs().argsort(descending=True)
    for m in range(1, len(order) + 1):
        Ct = torch.zeros_like(C)
        keep = order[:m]
        Ct[keep] = C[keep]
        ct = torch.fft.irfft(Ct, n=n_d)
        fvu = _fvu(ct[idx], M)
        if fvu <= eps:
            return dl_toeplitz_fourier(m), fvu, {'mode': 'fourier', 'modes': m}
    return dl_toeplitz_full(n_r, n_c), fvu_full, {'mode': 'full'}


CODEBOOKS = {'svd': fit_svd, 'bicluster': fit_bicluster, 'toeplitz': fit_toeplitz}


@torch.no_grad()
def fit_conjunction(M, eps, k_max=64, outer=8):
    """Conjunction codebook: M ~ (bicluster B) elementwise* (toeplitz gate c),
    fit by alternating weighted LS. DL = DL(bicluster) + DL(c Fourier) + 1 scale.

    Blind-from-product sign caveat (LOG tick 3): a sign-oscillating positional
    factor is not identifiable from the product alone (per-diagonal signs cannot
    be absorbed by a block-constant factor); the real pipeline decomposes the
    two BRANCHES separately (spec section 3) so blindness never arises there.
    The init here assumes a positive gate: c_init = sqrt(diag-mean of M^2);
    the c-update itself is unconstrained, so mild sign structure can still be
    recovered given good blocks.
    """
    n_r, n_c = M.shape
    idx = torch.arange(n_r, device=M.device)[:, None] - \
        torch.arange(n_c, device=M.device)[None, :] + (n_c - 1)
    n_d = n_r + n_c - 1

    def diag_ratio(num_mat, den_mat):
        s = torch.zeros(n_d, dtype=M.dtype, device=M.device)
        w = torch.zeros_like(s)
        s.index_put_((idx,), num_mat, accumulate=True)
        w.index_put_((idx,), den_mat, accumulate=True)
        return s / w.clamp(min=1e-30)

    c0 = diag_ratio(M ** 2, torch.ones_like(M)).clamp(min=1e-30).sqrt()
    c0 = c0 / c0.mean()

    # spectral partition init on the gate-whitened matrix (same lesson as
    # fit_bicluster: random partition init gets stuck — battery tick 2 & 3)
    Mw = M / c0[idx].clamp(min=0.1 * float(c0.mean()))
    Uw, Sw, Vwt = torch.linalg.svd(Mw, full_matrices=False)

    best = None
    k = 2
    while k <= k_max:
        c = c0.clone()
        r = min(k, len(Sw))
        rows = _kmeans_labels(Uw[:, :r] * Sw[:r], k)
        cols = _kmeans_labels(Vwt[:r].T * Sw[:r], k)
        B = torch.zeros(k, k, dtype=M.dtype, device=M.device)
        for _ in range(outer):
            T = c[idx]
            # block means (weighted LS)
            num = torch.zeros(k, k, dtype=M.dtype, device=M.device)
            den = torch.zeros_like(num)
            ri = rows[:, None].expand_as(M)
            ci = cols[None, :].expand_as(M)
            num.index_put_((ri, ci), M * T, accumulate=True)
            den.index_put_((ri, ci), T * T, accumulate=True)
            B = num / den.clamp(min=1e-30)
            # row reassignment (weighted)
            Ci = torch.nn.functional.one_hot(cols, k).to(M.dtype)
            P = (M * T) @ Ci
            W = (T * T) @ Ci
            rows = (-2 * P @ B.T + W @ (B ** 2).T).argmin(1)
            Ri = torch.nn.functional.one_hot(rows, k).to(M.dtype)
            Pc = (M * T).T @ Ri
            Wc = (T * T).T @ Ri
            cols = (-2 * Pc @ B + Wc @ (B ** 2)).argmin(1)
            # gate update (unconstrained LS per diagonal)
            M1 = B[rows][:, cols]
            c = diag_ratio(M * M1, M1 * M1)
        M1 = B[rows][:, cols]
        fvu = _fvu(M1 * c[idx], M)
        dl_blocks = dl_bicluster(k, k, n_r, n_c)
        best = (dl_blocks + dl_toeplitz_full(n_r, n_c) + 32, fvu,
                {'k': k, 'c_mode': 'full'})
        if fvu <= eps:
            # Fourier-truncate the gate under the SAME total-eps constraint
            C = torch.fft.rfft(c)
            order = C.abs().argsort(descending=True)
            for m in range(1, min(len(order), 64) + 1):
                Ct = torch.zeros_like(C)
                Ct[order[:m]] = C[order[:m]]
                ct = torch.fft.irfft(Ct, n=n_d)
                if _fvu(M1 * ct[idx], M) <= eps:
                    return (dl_blocks + dl_toeplitz_fourier(m) + 32,
                            _fvu(M1 * ct[idx], M), {'k': k, 'modes': m})
            return best
        k *= 2
    return best[0], best[1], {**best[2], 'hit_kmax': True}


CODEBOOKS['conjunction'] = fit_conjunction
