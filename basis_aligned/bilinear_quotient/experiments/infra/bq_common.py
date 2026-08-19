"""Shared machinery for the bilinear-quotient program (see
`../bilinear-quotient-experiments.md`, section 0).

Layer under study:      y = D( (Lx) ⊙ (Rx) ),   x∈R^d, hidden h, out m
Per-output form:        Q_i = sym( Σ_k D_ik l_k r_k^T ),  y_i = x^T Q_i x
Reader u (linear functional on y):  Q_u = Σ_i u_i Q_i

Everything the layer computes lives in the stack Q (m,d,d); L,R,D are a gauge
choice of factorization. All error is measured with the Λ-weighted functional
metric (Gaussian tensor inner product, `lam_*` below), never raw Frobenius.
"""

from __future__ import annotations

import math

import torch


# ------------------------------------------------------------------ model

def init_params(d, h, m, seed, device='cpu', scale=1.0):
    g = torch.Generator().manual_seed(seed)

    def mat(a, b):
        return (torch.randn(a, b, generator=g) * scale / math.sqrt(b)).to(device)

    return {'L': mat(h, d), 'R': mat(h, d), 'D': mat(m, h)}


def forward(p, x):
    return ((x @ p['L'].T) * (x @ p['R'].T)) @ p['D'].T


def interaction(p):
    """Symmetric per-output forms Q (m,d,d) with y_i = x^T Q_i x."""
    B = torch.einsum('ik,ka,kb->iab', p['D'], p['L'], p['R'])
    return 0.5 * (B + B.transpose(1, 2))


def forward_Q(Q, x):
    return torch.einsum('na,iab,nb->ni', x, Q, x)


def reader_forms(Q, U):
    """U (r,m) rows are readers; returns Q_u (r,d,d)."""
    return torch.einsum('ri,iab->rab', U, Q)


# ------------------------------------------------------------------ Λ metric

def lam_inner(A, B, G=None):
    """⟨A|Λ|B⟩ = E_{x~N(0,G)}[ y_A(x)·y_B(x) ] for symmetric form stacks.

    Isserlis: E = Σ_i [ tr_G(A_i) tr_G(B_i) + 2 tr(A_i G B_i G) ].
    """
    if G is None:
        G = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    tA = torch.einsum('iab,ab->i', A, G)
    tB = torch.einsum('iab,ab->i', B, G)
    cross = torch.einsum('iab,bc,icd,da->', A, G, B, G)
    return tA @ tB + 2 * cross


def lam_norm2(A, G=None):
    return lam_inner(A, A, G)


def lam_relerr(A, Ahat, G=None):
    """‖A-Â‖²_Λ / ‖A‖²_Λ = E‖y-ŷ‖² / E‖y‖²  (functional FVU-like error)."""
    return float((lam_norm2(A - Ahat, G) / lam_norm2(A, G).clamp_min(1e-30)).item())


def lam_cos(A, B, G=None):
    ab = lam_inner(A, B, G)
    return float((ab / (lam_norm2(A, G) * lam_norm2(B, G)).clamp_min(1e-30).sqrt()).item())


# ------------------------------------------------------------------ whitening

def sqrtm_psd(Sigma, eps=1e-10):
    ev, V = torch.linalg.eigh(Sigma)
    return V @ torch.diag(ev.clamp_min(eps).sqrt()) @ V.T


def whiten(Q, Sigma, eps=1e-10):
    """Q ↦ Σ^{1/2} Q Σ^{1/2}: forms in the whitened input metric, so that the
    Λ-metric with G=I on the result equals the Λ-metric with G=Σ on Q."""
    ev, V = torch.linalg.eigh(Sigma)
    S = V @ torch.diag(ev.clamp_min(eps).sqrt()) @ V.T
    return torch.einsum('ab,ibc,cd->iad', S, Q, S)


def input_moment(x):
    return (x.T @ x) / x.shape[0]


# ------------------------------------------------- reader-weighted second moment

def reader_moment(Qu, w=None):
    """M = Σ_u w_u vec(Q_u) vec(Q_u)^T over the d(d+1)/2 symmetric coords
    (off-diagonals carry √2 so the vec is an isometry of the Frobenius norm)."""
    r, d, _ = Qu.shape
    iu, ju = torch.triu_indices(d, d)
    scale = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Qu)
    V = Qu[:, iu, ju] * scale                       # (r, d(d+1)/2)
    if w is not None:
        V = V * torch.as_tensor(w, dtype=V.dtype, device=V.device)[:, None].sqrt()
    return V.T @ V


def spectrum(M):
    ev = torch.linalg.eigvalsh(M).flip(0).clamp_min(0)
    return ev


def participation_ratio(ev):
    ev = ev.clamp_min(0)
    return float((ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30)).item())


# ------------------------------------------------------------------ nulls

def gauge_refactor(p, seed, h_new=None, device=None, iters=0):
    """NULL 2 (gauge scramble). Re-express the SAME function with a fresh random
    hidden basis: sample random l'_k, r'_k, least-squares-solve D' so that
    Q(p') = Q(p). Exact whenever span{sym(l'_k r'_k^T)} ⊇ span{Q_i}, which holds
    w.p. 1 once h_new ≥ d(d+1)/2. Function-preserving, hidden-basis-destroying.

    Returns (p_new, rel_residual).
    """
    Q = interaction(p)
    m, d, _ = Q.shape
    device = device or Q.device
    h_new = h_new or (d * (d + 1) // 2)
    g = torch.Generator().manual_seed(seed)
    L = (torch.randn(h_new, d, generator=g) / math.sqrt(d)).to(device)
    R = (torch.randn(h_new, d, generator=g) / math.sqrt(d)).to(device)
    S = 0.5 * (torch.einsum('ka,kb->kab', L, R) + torch.einsum('kb,ka->kab', L, R))
    iu, ju = torch.triu_indices(d, d)
    scale = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(S)
    A = (S[:, iu, ju] * scale).T                       # (d(d+1)/2, h_new)
    Y = (Q[:, iu, ju] * scale).T                       # (d(d+1)/2, m)
    Dnew = torch.linalg.lstsq(A.double(), Y.double()).solution.T.to(Q.dtype)
    p_new = {'L': L, 'R': R, 'D': Dnew}
    res = lam_relerr(Q, interaction(p_new))
    return p_new, res


def reader_shuffle(Qu, seed):
    """NULL 4. Randomly reassign which reader gets which form."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(Qu.shape[0], generator=g)
    return Qu[perm]


def form_shuffle(Qu, seed):
    """Stronger reader-shuffle: independently permute the reader index of each
    *form direction*, destroying which forms co-occur in a reader while keeping
    every marginal (each coordinate's multiset of values across readers)."""
    g = torch.Generator().manual_seed(seed)
    r, d, _ = Qu.shape
    iu, ju = torch.triu_indices(d, d)
    V = Qu[:, iu, ju].clone()
    for c in range(V.shape[1]):
        V[:, c] = V[torch.randperm(r, generator=g), c]
    out = torch.zeros_like(Qu)
    out[:, iu, ju] = V
    out[:, ju, iu] = V
    return out


# ------------------------------------ simultaneous block diagonalization (∗-algebra)

def commutant_basis(Qs, tol=None, max_dim=None):
    """Basis of {X symmetric : [X, Q_i] = 0 ∀i}, as (n_basis, d, d).

    Maehara–Murota: the commutant of the matrix ∗-algebra generated by {Q_i}.
    Computed as the numerical nullspace of X ↦ ([X,Q_1],…,[X,Q_n]) restricted to
    symmetric X, via the Gram matrix of that operator.
    """
    Qs = Qs.double()
    d = Qs.shape[-1]
    iu, ju = torch.triu_indices(d, d)
    K = iu.numel()
    basis = torch.zeros(K, d, d, dtype=Qs.dtype, device=Qs.device)
    basis[torch.arange(K), iu, ju] = 1.0
    basis[torch.arange(K), ju, iu] = 1.0
    nrm = basis.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-30)
    basis = basis / nrm[:, :, None]
    # C[i,k] = vec([basis_k, Q_i]);  Gram = Σ_i C_i^T C_i
    Gram = torch.zeros(K, K, dtype=Qs.dtype, device=Qs.device)
    for Q in Qs:
        Cm = (basis @ Q - Q @ basis).flatten(1)        # (K, d*d)
        Gram += Cm @ Cm.T
    ev, V = torch.linalg.eigh(Gram)
    if tol is None:
        scale = max(float(ev.max()), 1e-30)
        tol = scale * 1e-8
    keep = ev < tol
    if max_dim is not None and int(keep.sum()) > max_dim:
        keep = torch.zeros_like(keep)
        keep[:max_dim] = True
    coef = V[:, keep]                                   # (K, n_basis)
    return torch.einsum('kn,kab->nab', coef, basis), ev


def sbd(Qs, tol=None, gap_rel=1e-6, seed=0):
    """Simultaneous block diagonalization of a family of symmetric matrices.

    Returns (P, sizes, info): orthogonal P (d,d) whose columns are grouped so
    that P^T Q_i P is block diagonal with block sizes `sizes`, for every i.
    A generic element of the commutant is diagonalized; its eigenspaces are the
    finest common invariant subspaces (generically).
    """
    Qs = Qs.double()
    d = Qs.shape[-1]
    B, ev_gram = commutant_basis(Qs, tol=tol)
    n = B.shape[0]
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, generator=g).to(B)
    X = torch.einsum('n,nab->ab', c, B)
    X = 0.5 * (X + X.T)
    ev, P = torch.linalg.eigh(X)
    # group near-equal eigenvalues -> invariant subspaces. The split tolerance is
    # relative to |X| itself (NOT to the eigenvalue span: when the commutant is
    # trivial, X ∝ I and the span is pure numerical noise).
    scale = max(float(ev.abs().max()), 1e-30)
    sizes, start = [], 0
    for i in range(1, d + 1):
        if n == 1:                       # trivial commutant => irreducible
            sizes, start = [d], d
            break
        if i == d or (ev[i] - ev[i - 1]) > gap_rel * scale:
            sizes.append(i - start)
            start = i
    info = {'commutant_dim': n, 'eigs': ev.tolist(), 'gram_ev_min': float(ev_gram.min()),
            'gram_ev_max': float(ev_gram.max())}
    return P.to(Qs.dtype), sizes, info


def sbd_robust(Qs, n_blocks=None, commutant_dim=None, seed=0, max_blocks=None, n_draws=8):
    """Approximate SBD for a family that only APPROXIMATELY block-diagonalises.

    `sbd` uses the exact numerical nullspace of the commutator operator, which is
    trivial (span{I}) as soon as the family carries any off-block noise. Here the
    commutant is taken as the k lowest eigenvectors of the commutator Gram, with
    k chosen by the largest multiplicative gap in the low spectrum (or given).
    Because a symmetric commutant contributes one real dimension per irreducible
    block, k IS the block count, so the eigenvalues of a generic approximate-
    commutant element are split at the k-1 largest gaps — no tolerance needed.

    NOTE the commutant dimension is NOT the block count in general: an isotypic
    component of multiplicity μ carrying a real-type irrep contributes μ(μ+1)/2
    symmetric commutant dimensions but μ blocks. So k is chosen for the commutant
    and the block count is read off the eigenvalue clustering of a generic
    element (elbow on the sorted gaps).

    `defect` = (Gram eigenvalue at index k-1)/max: the residual non-commutativity
    of the chosen commutant. defect ≈ 0 means a genuine ∗-algebra decomposition;
    it is the continuous "how block-diagonalisable is this family" measure.
    """
    Qs = Qs.double()
    d = Qs.shape[-1]
    B_all, ev = commutant_basis(Qs, max_dim=None, tol=float('inf'))
    ev = ev / ev.max().clamp_min(1e-300)
    kmax = min(commutant_dim or d * (d + 1) // 2, len(ev) - 1, 3 * d)
    if commutant_dim is None:
        lo = ev.clamp_min(1e-18)
        ratios = [(float(lo[i + 1] / lo[i]), i + 1) for i in range(1, kmax)]
        commutant_dim = max(ratios)[1] if ratios else 1
    k = int(min(max(commutant_dim, 1), B_all.shape[0]))
    Bk = B_all[:k]
    g = torch.Generator().manual_seed(seed)
    best = None
    for _ in range(n_draws):                      # a few draws; keep the cleanest split
        c = torch.randn(k, generator=g).to(Bk)
        X = 0.5 * (torch.einsum('n,nab->ab', c, Bk) + torch.einsum('n,nba->ab', c, Bk))
        evx, P = torch.linalg.eigh(X)
        cuts = _eig_cuts(evx, n_blocks, max_blocks or d)
        sizes, prev = [], -1
        for cpt in cuts + [d - 1]:
            sizes.append(cpt - prev)
            prev = cpt
        mass, _ = block_mass(Qs, P.to(Qs.dtype), sizes)
        if best is None or (len(sizes), mass) > (len(best[2]), best[0]):
            best = (mass, P, sizes)
    mass, P, sizes = best
    return P.to(Qs.dtype), sizes, {'commutant_dim': k, 'n_blocks': len(sizes),
                                   'in_block_mass': mass, 'defect': float(ev[k - 1]),
                                   'gram_spectrum': [float(v) for v in ev[:min(len(ev), 40)]]}


def _eig_cuts(evx, n_blocks, max_blocks, tol_gap=1e-3):
    """Where to split a sorted eigenvalue list into blocks. With n_blocks given,
    take the n_blocks-1 largest gaps. Otherwise split at every gap exceeding
    tol_gap × the eigenvalue range — an explicit tolerance rather than an elbow,
    because within-block gaps are exactly zero in the noiseless case and the
    ratio between two numerically-zero gaps is meaningless."""
    gaps = evx[1:] - evx[:-1]
    if n_blocks is not None:
        return sorted(torch.topk(gaps, min(n_blocks - 1, len(gaps))).indices.tolist()) \
            if n_blocks > 1 else []
    rng = float(evx.max() - evx.min())
    if rng <= 0:
        return []
    order = torch.argsort(gaps, descending=True)
    n_sig = int((gaps > tol_gap * rng).sum())
    n_sig = min(n_sig, max_blocks - 1, len(gaps))
    return sorted(order[:n_sig].tolist())


def isotypic_groups(Qs, P, sizes, commutant_dim=None, rel=1e-2):
    """Merge fine blocks into isotypic components — the CANONICAL object.

    When an irrep appears with multiplicity > 1 the split of its isotypic
    component into individual blocks is not unique (any basis of the multiplicity
    space works), so only the merged component is well defined. Two fine blocks
    are in the same component iff the commutant contains an intertwiner between
    them, i.e. has mass in that off-diagonal position.
    """
    Qs = Qs.double()
    B, ev = commutant_basis(Qs, tol=float('inf'))
    ev = ev / ev.max().clamp_min(1e-300)
    if commutant_dim is None:
        lo = ev.clamp_min(1e-18)
        commutant_dim = max((float(lo[i + 1] / lo[i]), i + 1)
                            for i in range(1, min(len(ev) - 1, 3 * Qs.shape[-1])))[1]
    Bp = torch.einsum('ai,nab,bj->nij', P.double(), B[:commutant_dim], P.double())
    nb = len(sizes)
    offs = [0]
    for s in sizes:
        offs.append(offs[-1] + s)
    diag = [float((Bp[:, offs[i]:offs[i + 1], offs[i]:offs[i + 1]] ** 2).sum()) for i in range(nb)]
    parent = list(range(nb))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(nb):
        for j in range(i + 1, nb):
            off = float((Bp[:, offs[i]:offs[i + 1], offs[j]:offs[j + 1]] ** 2).sum())
            if off > rel * math.sqrt(max(diag[i] * diag[j], 1e-300)):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(nb):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for v in groups.values()]


def sbd_optimize(Qs, sizes, steps=3000, lr=5e-2, restarts=4, seed=0, G=None,
                 verbose=False):
    """SBD by direct search: find P ∈ O(d) minimising the mass of P^T Q P outside
    a given block pattern. The commutant route (`sbd`, `sbd_robust`) is algebraic
    and therefore brittle — it asks for matrices that commute exactly. This asks
    only for the decomposition that best explains the mass, so it degrades
    gracefully. Cost: one orthogonal matrix, gradient descent, a few restarts.

    Returns (P, off_block_mass, info). `sizes` is the block pattern to fit; sweep
    it and read off which pattern explains the mass at what description length.
    """
    Qs = Qs.double()
    d = Qs.shape[-1]
    assert sum(sizes) == d, f'block sizes {sum(sizes)} != d {d}'
    mask = torch.zeros(d, d, dtype=Qs.dtype, device=Qs.device)
    s = 0
    for b in sizes:
        mask[s:s + b, s:s + b] = 1.0
        s += b
    tot = (Qs ** 2).sum()
    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 100 + r)
        A = (torch.randn(d, d, generator=g) * 0.1).to(Qs).requires_grad_(True)
        opt = torch.optim.Adam([A], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for t in range(steps):
            P = torch.matrix_exp(A - A.T)
            T = torch.einsum('ai,mab,bj->mij', P, Qs, P)
            off = ((T ** 2) * (1 - mask)).sum() / tot
            opt.zero_grad()
            off.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            P = torch.matrix_exp(A - A.T)
            T = torch.einsum('ai,mab,bj->mij', P, Qs, P)
            off = float(((T ** 2) * (1 - mask)).sum() / tot)
        if verbose:
            print(f'    restart {r}: off-block {off:.5f}')
        if best is None or off < best[0]:
            best = (off, P.detach())
    off, P = best
    return P, 1 - off, {'off_block_mass': off, 'in_block_mass': 1 - off, 'sizes': sizes}


def jade(Qs, sweeps=30, tol=1e-10, verbose=False):
    """Orthogonal joint approximate diagonalisation (Cardoso–Souloumiac).

    Minimises the total off-diagonal mass of {PᵀQ_mP} over P ∈ O(d) by Givens
    rotations, each with a closed-form optimal angle (leading eigenvector of a
    2×2 Gram) — no local-minimum problem of the kind gradient descent on O(d)
    hits. A family that only block-diagonalises cannot be driven to a diagonal,
    so what survives is exactly the irreducible coupling: read the block
    structure off the residual coupling graph.
    """
    Qs = Qs.double().clone()
    m, d, _ = Qs.shape
    P = torch.eye(d, dtype=Qs.dtype, device=Qs.device)
    off_hist = []
    for sw in range(sweeps):
        moved = 0.0
        for p_ in range(d - 1):
            for q_ in range(p_ + 1, d):
                gx = Qs[:, p_, p_] - Qs[:, q_, q_]
                gy = 2 * Qs[:, p_, q_]
                Gxx, Gyy, Gxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
                # leading eigenvector of [[Gxx,Gxy],[Gxy,Gyy]]
                theta = 0.5 * torch.atan2(2 * Gxy, Gxx - Gyy) / 2
                c, s = torch.cos(theta), torch.sin(theta)
                if abs(float(s)) < 1e-15:
                    continue
                moved += float(abs(s))
                cp, cq = Qs[:, :, p_].clone(), Qs[:, :, q_].clone()
                Qs[:, :, p_] = c * cp + s * cq
                Qs[:, :, q_] = -s * cp + c * cq
                rp, rq = Qs[:, p_, :].clone(), Qs[:, q_, :].clone()
                Qs[:, p_, :] = c * rp + s * rq
                Qs[:, q_, :] = -s * rp + c * rq
                vp, vq = P[:, p_].clone(), P[:, q_].clone()
                P[:, p_] = c * vp + s * vq
                P[:, q_] = -s * vp + c * vq
        offm = float(((Qs ** 2).sum() - (torch.diagonal(Qs, dim1=1, dim2=2) ** 2).sum())
                     / (Qs ** 2).sum())
        off_hist.append(offm)
        if verbose:
            print(f'    sweep {sw}: off-diagonal mass {offm:.6f}  (rotation activity {moved:.2e})')
        if moved < tol:
            break
    return P, Qs, {'off_diag_mass': off_hist[-1], 'sweeps': len(off_hist),
                   'off_diag_history': off_hist}


def sbd_jade(Qs, tol=1e-6, sweeps=30, verbose=False):
    """Weights-only block recovery that degrades gracefully: JADE, then read the
    partition off the residual coupling graph at stratification tolerance `tol`.
    Returns (P_reordered, sizes, info)."""
    P, T, info = jade(Qs, sweeps=sweeps, verbose=verbose)
    parts = partition_from_coupling(T, tol=tol)
    Pr, sizes = reorder_by_partition(P, parts)
    mass, _ = block_mass(Qs, Pr, sizes)
    info.update({'in_block_mass': mass, 'sizes': sizes, 'n_blocks': len(sizes)})
    return Pr, sizes, info


def sbd_sparse(Qs, steps=4000, lr=3e-2, restarts=3, seed=0, init_P=None,
               tol=1e-6, verbose=False):
    """SBD without knowing the block sizes OR the ordering.

    Minimises Σ_m ‖PᵀQ_mP‖₁ over P ∈ O(d). Among orthogonal changes of basis the
    L1 norm is minimised by the sparsest representation, and a block-diagonal
    family (in any ordering) is sparse — so no block pattern has to be supplied.
    The partition is then read off as the connected components of the graph whose
    edges are the surviving off-diagonal couplings, which is exactly what a
    simultaneous block structure is.
    """
    Qs = Qs.double()
    d = Qs.shape[-1]
    nrm = (Qs ** 2).sum().sqrt()
    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 100 + r)
        A = (torch.randn(d, d, generator=g) * 0.05).to(Qs)
        if init_P is not None and r == 0:
            A = torch.zeros(d, d, dtype=Qs.dtype, device=Qs.device)
        A = A.requires_grad_(True)
        P0 = init_P.double() if (init_P is not None and r == 0) else None
        opt = torch.optim.Adam([A], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for t in range(steps):
            P = torch.matrix_exp(A - A.T)
            if P0 is not None:
                P = P0 @ P
            T = torch.einsum('ai,mab,bj->mij', P, Qs, P)
            loss = T.abs().sum() / nrm
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            P = torch.matrix_exp(A - A.T)
            if P0 is not None:
                P = P0 @ P
            T = torch.einsum('ai,mab,bj->mij', P, Qs, P)
            l1 = float(T.abs().sum() / nrm)
        if verbose:
            print(f'    restart {r}: L1/‖Q‖ {l1:.4f}')
        if best is None or l1 < best[0]:
            best = (l1, P.detach())
    l1, P = best
    T = torch.einsum('ai,mab,bj->mij', P, Qs, P)
    return P, partition_from_coupling(T, tol=tol), {'l1': l1}


def _components(w, iu, ju, d, thresh):
    parent = list(range(d))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in torch.nonzero(w > thresh).flatten().tolist():
        parent[find(int(iu[e]))] = find(int(ju[e]))
    comp = {}
    for i in range(d):
        comp.setdefault(find(i), []).append(i)
    return sorted(comp.values(), key=lambda v: (-len(v), v[0]))


def partition_from_coupling(T, tol=1e-6, thresh=None):
    """Block structure implied by which coordinates still interact: connected
    components of the coupling graph of the family T (m,d,d).

    The threshold is not a free parameter — it is the stratification tolerance.
    We return the FINEST partition whose blocks still carry (1-tol) of the mass;
    in-block mass decreases monotonically as the threshold rises, so this is a
    binary search. tol IS the ε of the preserved/dead cut, made explicit.
    """
    d = T.shape[-1]
    Wt = (T ** 2).sum(0).sqrt()
    Wt = Wt - torch.diag(torch.diagonal(Wt))
    iu, ju = torch.triu_indices(d, d, offset=1)
    w = Wt[iu, ju]
    if thresh is not None:
        return _components(w, iu, ju, d, thresh)
    tot = float((T ** 2).sum())

    def mass(th):
        parts = _components(w, iu, ju, d, th)
        keep = torch.zeros(d, d, dtype=torch.bool, device=T.device)
        for p in parts:
            idx = torch.tensor(p, device=T.device)
            keep[idx[:, None], idx[None, :]] = True
        return float((T ** 2 * keep).sum() / tot), parts

    cand = torch.cat([torch.zeros(1, device=w.device), torch.sort(w).values])
    lo, hi = 0, len(cand) - 1                      # lo always feasible
    if mass(float(cand[hi]))[0] >= 1 - tol:
        return mass(float(cand[hi]))[1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mass(float(cand[mid]))[0] >= 1 - tol:
            lo = mid
        else:
            hi = mid
    return mass(float(cand[lo]))[1]


def reorder_by_partition(P, parts):
    """Column-permute P so each partition block is contiguous; return (P, sizes)."""
    idx = [i for part in parts for i in part]
    return P[:, idx], [len(p) for p in parts]


def support_basis(Qs, thresh=1e-3):
    """Orthonormal basis of the joint range of {Q_i} — the subspace the family
    actually acts on. SBD is run here: outside it every Q_i is zero, so the
    commutant is unconstrained and would fragment into spurious 1×1 blocks."""
    m, d, _ = Qs.shape
    U, s, _ = torch.linalg.svd(Qs.transpose(0, 1).reshape(d, m * d).double())
    k = int((s > thresh * s.max()).sum())
    return U[:, :k].to(Qs.dtype), s


def restrict(Qs, U):
    """Express the family in an orthonormal sub-basis U (d,k): Q ↦ Uᵀ Q U."""
    return torch.einsum('ip,mij,jq->mpq', U, Qs, U)


def block_mass(Qs, P, sizes):
    """Fraction of Σ_i ‖Q_i‖²_F that lands inside the block-diagonal pattern."""
    T = torch.einsum('ba,ibc,cd->iad', P, Qs.to(P), P)
    d = T.shape[-1]
    mask = torch.zeros(d, d, dtype=torch.bool, device=T.device)
    s = 0
    for b in sizes:
        mask[s:s + b, s:s + b] = True
        s += b
    tot = (T ** 2).sum()
    return float(((T ** 2 * mask).sum() / tot.clamp_min(1e-30)).item()), T


# ------------------------------------------------------------------ CP / Waring

def fit_cp(Q, rank, steps=4000, lr=3e-2, G=None, seed=0, sym=False, init=None,
           verbose=False, device=None):
    """Fit Q ≈ sym(Σ_k d_k ⊗ l_k ⊗ r_k) under the Λ metric. `sym=True` fits the
    Waring form (r_k = l_k). Returns (params, rel_err_history[-1], params_hist)."""
    device = device or Q.device
    m, d, _ = Q.shape
    g = torch.Generator().manual_seed(seed)
    if init is None:
        L = (torch.randn(rank, d, generator=g) / math.sqrt(d)).to(device)
        R = L.clone() if sym else (torch.randn(rank, d, generator=g) / math.sqrt(d)).to(device)
        D = (torch.randn(m, rank, generator=g) / math.sqrt(rank)).to(device)
    else:
        L, R, D = (init[k].clone() for k in ('L', 'R', 'D'))
    params = [D, L] if sym else [D, L, R]
    for t in params:
        t.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    nrm = lam_norm2(Q, G).clamp_min(1e-30)
    for s in range(steps):
        Rm = L if sym else R
        Qh = interaction({'L': L, 'R': Rm, 'D': D})
        loss = lam_norm2(Q - Qh, G) / nrm
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and s % max(1, steps // 5) == 0:
            print(f'    cp r={rank} step {s} relerr {loss.item():.3e}')
    with torch.no_grad():
        Rm = L if sym else R
        out = {'L': L.detach(), 'R': Rm.detach(), 'D': D.detach()}
        err = lam_relerr(Q, interaction(out), G)
    return out, err


# ------------------------------------------------------------------ dictionary

def fit_dictionary(Qu, n_atoms, steps=4000, lr=3e-2, l1=1e-2, G=None, seed=0,
                   verbose=False):
    """Sparse dictionary over reader forms: Q_u ≈ Σ_a C_ua A_a, ‖C‖_1 penalised,
    atoms unit-Λ-norm. Returns (atoms, codes, rel_err)."""
    r, d, _ = Qu.shape
    g = torch.Generator().manual_seed(seed)
    A = (torch.randn(n_atoms, d, d, generator=g) / d).to(Qu)
    A = 0.5 * (A + A.transpose(1, 2))
    C = (torch.randn(r, n_atoms, generator=g) * 0.1).to(Qu)
    A.requires_grad_(True)
    C.requires_grad_(True)
    opt = torch.optim.Adam([A, C], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    nrm = lam_norm2(Qu, G).clamp_min(1e-30)
    for s in range(steps):
        As = 0.5 * (A + A.transpose(1, 2))
        scale = lam_norm2_each(As, G).clamp_min(1e-12).sqrt()
        An = As / scale[:, None, None]
        Qh = torch.einsum('ra,aij->rij', C, An)
        loss = lam_norm2(Qu - Qh, G) / nrm + l1 * C.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and s % max(1, steps // 5) == 0:
            print(f'    dict k={n_atoms} step {s} loss {loss.item():.3e}')
    with torch.no_grad():
        As = 0.5 * (A + A.transpose(1, 2))
        An = As / lam_norm2_each(As, G).clamp_min(1e-12).sqrt()[:, None, None]
        Qh = torch.einsum('ra,aij->rij', C, An)
        err = lam_relerr(Qu, Qh, G)
    return An.detach(), C.detach(), err


def lam_norm2_each(A, G=None):
    """Per-slice Λ norm² (treating each slice as a 1-output family)."""
    if G is None:
        G = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    t = torch.einsum('iab,ab->i', A, G)
    cross = torch.einsum('iab,bc,icd,da->i', A, G, A, G)
    return t ** 2 + 2 * cross


# ------------------------------------------------------------------ training

def train(p, data_fn, steps, lr=3e-3, wd=0.0, loss_kind='mse', log_every=0,
          checkpoint_every=0, eval_fn=None, cosine=False):
    """cosine=True anneals lr→0. Use it whenever the target is exactly realisable:
    Adam normalises by the gradient RMS, so at machine-precision loss it keeps
    taking lr-sized steps on pure noise and the solution drifts back off exact."""
    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cosine else None
    hist, ckpts = [], []
    for s in range(steps):
        x, y = data_fn()
        out = forward(p, x)
        if loss_kind == 'mse':
            loss = ((out - y) ** 2).mean()
        else:
            loss = torch.nn.functional.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if checkpoint_every and (s % checkpoint_every == 0 or s == steps - 1):
            rec = {'step': s, 'loss': float(loss.item())}
            if eval_fn is not None:
                rec.update(eval_fn(p))
            hist.append(rec)
            ckpts.append({k: v.detach().clone() for k, v in p.items()})
            if log_every:
                print('  ' + '  '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}'
                                       for k, v in rec.items()))
    for k in p:
        p[k].requires_grad_(False)
    return p, hist, ckpts


# ------------------------------------------------------------------ misc

def effective_rank(Q, thresh=1e-6):
    """Rank of the map W: Sym²(V) → R^m, i.e. of the flattened form stack."""
    m, d, _ = Q.shape
    iu, ju = torch.triu_indices(d, d)
    scale = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Q)
    V = Q[:, iu, ju] * scale
    s = torch.linalg.svdvals(V.double())
    return int((s > thresh * s.max()).sum()), s


def row_space_kernel(Q, thresh=1e-6):
    """Row space and kernel of W in Sym²(V) coordinates (as (k, d, d) stacks)."""
    m, d, _ = Q.shape
    iu, ju = torch.triu_indices(d, d)
    scale = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Q)
    V = Q[:, iu, ju] * scale
    U_, s, Vh = torch.linalg.svd(V.double(), full_matrices=True)
    keep = s > thresh * s.max()
    k = int(keep.sum())

    def unvec(rows):
        out = torch.zeros(rows.shape[0], d, d, dtype=torch.float64, device=rows.device)
        sc = scale.double().to(rows.device)
        out[:, iu.to(rows.device), ju.to(rows.device)] = rows / sc
        out[:, ju.to(rows.device), iu.to(rows.device)] = rows / sc
        return out.to(Q.dtype)

    return unvec(Vh[:k]), unvec(Vh[k:]), s
