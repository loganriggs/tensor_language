"""Tier 0.3 (recovery) + 0.4 (ablation) — spec §2 pass/fail gates.

Run: python -m mechdecomp.tier0 [lam=0.02] [m=40]
"""

import sys

import torch

from .estep import solve_codes
from .mstep import resample_dead, update_dictionary
from .objective import r2
from .toys import orthonormal_dgp

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def dedup(D, C, X, thresh=0.95):
    """Merge near-duplicate atoms: keep the higher-code-mass one, reinit the other
    to a high-residual direction (handled by the next resample_dead pass)."""
    m = D.shape[1]
    mass = C.abs().sum(1)
    G = (D.T @ D).abs()
    merged = 0
    for i in range(m):
        for j in range(i + 1, m):
            if G[i, j] > thresh and mass[i] > 0 and mass[j] > 0:
                lo = i if mass[i] < mass[j] else j
                C[lo] = 0
                mass[lo] = 0
                merged += 1
    return D, C, merged


def kmeans_init(X, m, seed=0):
    """Spec init (c): k-means on unit-normalized activations, then the top right-
    singular direction of each cluster as its atom."""
    from sklearn.cluster import KMeans
    Xn = (X / X.norm(dim=0, keepdim=True)).T.cpu().numpy()
    km = KMeans(n_clusters=m, n_init=4, random_state=seed).fit(Xn)
    D = torch.zeros(X.shape[0], m, dtype=X.dtype, device=X.device)
    for j in range(m):
        idx = torch.from_numpy((km.labels_ == j).nonzero()[0]).to(X.device)
        if len(idx) == 0:
            D[:, j] = torch.randn(X.shape[0], dtype=X.dtype, device=X.device)
        else:
            _, _, Vh = torch.linalg.svd(X[:, idx].T, full_matrices=False)
            D[:, j] = Vh[0]
    return D / D.norm(dim=0, keepdim=True)


def svd_init(W, X, m):
    """Spec init (b): top-m right singular vectors of W X (the map's active input
    directions). Fills to m with residual-input directions if m > rank."""
    Y = W @ X
    # right singular vectors of Y over data = principal INPUT directions W acts on;
    # use X-side: top singular vecs of X weighted by ||W·||. Practical: SVD of X, then
    # of the map applied. Use eigenvectors of X Xᵀ (input directions), ordered by
    # how much W stretches them.
    U, S, _ = torch.linalg.svd(X, full_matrices=False)          # U: (d_in, r) input dirs
    r = min(m, U.shape[1])
    D = torch.zeros(X.shape[0], m, dtype=X.dtype, device=X.device)
    D[:, :r] = U[:, :r]
    if m > r:
        g = torch.randn(X.shape[0], m - r, dtype=X.dtype, device=X.device)
        D[:, r:] = g
    return D / D.norm(dim=0, keepdim=True)


def train(W, X, m=40, lam=0.02, rounds=25, seed=0, verbose=True, init="kmeans",
          gain_weighted=False, prune_tol=2e-3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if init == "kmeans":
        D = kmeans_init(X, m, seed)
    elif init == "svd":
        D = svd_init(W, X, m)
    else:
        D = torch.randn(X.shape[0], m, generator=g).to(X)
    D = D / D.norm(dim=0, keepdim=True)
    C = None
    for t in range(rounds):
        C = solve_codes(W, D, X, lam, C0=C, gain_weighted=gain_weighted)
        if t % 3 == 2:
            D, C, nm = dedup(D, C, X)
        D = update_dictionary(W, D, C, X)
        D, ndead = resample_dead(W, D, C, X)
        if verbose and t % 5 == 0:
            print(f"  round {t}: R2 {r2(W, D, C, X):.4f} "
                  f"L0 {(C.abs() > 1e-8).sum(0).float().mean():.2f} dead {ndead}", flush=True)
    C = solve_codes(W, D, X, lam, C0=C, gain_weighted=gain_weighted)
    D, C, _ = dedup(D, C, X)
    keep = C.abs().sum(1) > 0                      # drop merged-away atoms for good
    D, C = D[:, keep], C[keep]
    C = solve_codes(W, D, X, lam, C0=C, gain_weighted=gain_weighted)
    D, C = prune(W, D, C, X, lam, tol=prune_tol)
    return D, C


def prune(W, D, C, X, lam, tol=2e-3):
    """Backward selection at the dictionary level: drop any atom whose removal
    costs less than tol R² after re-coding. Removes composite/parasite atoms
    (their points re-code onto pure atoms); pure atoms are protected by their
    singleton datapoints."""
    base = r2(W, D, C, X)
    improved = True
    while improved and D.shape[1] > 1:
        improved = False
        order = torch.argsort(C.abs().sum(1))      # try low-mass atoms first
        for j in order.tolist():
            keep = torch.ones(D.shape[1], dtype=torch.bool)
            keep[j] = False
            D2 = D[:, keep]
            C2 = solve_codes(W, D2, X, lam, C0=C[keep])
            if base - r2(W, D2, C2, X) < tol:
                D, C = D2, C2
                base = r2(W, D, C, X)
                improved = True
                break
    return D, C


def recovery_metrics(D, C, E, S):
    cos = (E.T @ D).abs()                        # (m_true, m) — sign-fixed by abs
    best, which = cos.max(1)
    # support F1 per true feature using its best-matched atom
    f1s = []
    for j in range(E.shape[1]):
        pred = C[which[j]].abs() > 1e-6
        true = S[:, j]
        tp = (pred & true).sum().item()
        p = tp / max(1, pred.sum().item()); r = tp / max(1, true.sum().item())
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return best, torch.tensor(f1s), which


def main(lam=0.02, m=40):
    W, X, E, S = orthonormal_dgp(N=10_000, device=DEV)
    W, X, E = W.double(), X.double(), E.double()
    D, C = train(W, X, m=m, lam=lam)
    best, f1s, which = recovery_metrics(D, C, E, S)
    print(f"\nTier 0.3 recovery: min max-cos {best.min():.4f} (gate >0.99) | "
          f"mean {best.mean():.4f} | min F1 {f1s.min():.3f} (gate >0.95) | mean F1 {f1s.mean():.3f}")
    ok3 = best.min() > 0.99 and f1s.min() > 0.95

    # Tier 0.4 ablation: W' = W − W d d^T for each matched atom
    print("\nTier 0.4 ablation (per true feature: ||ΔWx|| on active vs inactive):")
    ok4 = True
    for j in range(E.shape[1]):
        d = D[:, which[j]]
        Wp = W - torch.outer(W @ d, d)
        delta = ((Wp @ X - W @ X) ** 2).sum(0).sqrt()
        on, off = delta[S[:, j]], delta[~S[:, j]]
        ratio = (on.mean() / off.mean().clamp_min(1e-9)).item()
        ok4 &= ratio > 20
        print(f"  feat {j}: active {on.mean():.3f}  inactive {off.mean():.4f}  ratio {ratio:.0f}x")
    print(f"\nGATES: recovery {'PASS' if ok3 else 'FAIL'} | ablation {'PASS' if ok4 else 'FAIL'}")
    return ok3 and ok4


if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    main(lam, m)
