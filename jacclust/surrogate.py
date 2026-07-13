"""Ground-truth-free validation: the per-cluster LINEAR SURROGATE test.

Real models have no mechanism labels, so "G-clusters differ from x-clusters" proves nothing. But the
method's *claim* is falsifiable without labels:

    if a cluster groups datapoints on which the layer applied (nearly) the same linear map,
    then a SINGLE linear map should approximate the layer's action on that cluster.

So: cluster on TRAIN tokens, fit y ≈ A_c x per cluster by least squares on TRAIN, assign HELD-OUT
tokens to clusters by nearest centroid in the same metric, and score held-out R².

Baselines that could win: raw-cosine clusters, spectrum-matched random-eigenvector metric (G_rand),
random cluster assignment, and a single global linear map. All at matched k, 5 k-means seeds.

Note the projected kernel is closed-form: J(x)P = D[diag(Lx) RP + diag(Rx) LP], so the Gram of the
projected Jacobian is exactly gram(D, L@P, R@P) — still weights-only, no Jacobians materialized.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans

from .metric import embed, gram


def gram_with(D, L, R, P):
    """Gram of x -> J(x) P.  J(x)P = D[diag(Lx) RP + diag(Rx) LP]: the GATES stay Lx, Rx, so P goes
    inside the middle Grams, NOT on the outer L,R (using P P^T = P for a symmetric projector).

        G_P = L^T (M ⊙ R P R^T) L + L^T (M ⊙ R P L^T) R
            + R^T (M ⊙ L P R^T) L + R^T (M ⊙ L P L^T) R

    NB: the naive `gram(D, L@P, R@P)` is WRONG (it also projects the gates) — caught by the identity
    test in tests/, which is why that test exists.
    """
    M = D.T @ D
    return (L.T @ (M * (R @ P @ R.T)) @ L
            + L.T @ (M * (R @ P @ L.T)) @ R
            + R.T @ (M * (L @ P @ R.T)) @ L
            + R.T @ (M * (L @ P @ L.T)) @ R)


def projected_gram(D, L, R, r):
    """G_P for J(x)P, P = I - E E^T with E the top-r eigenvectors of G. Weights only."""
    G = gram(D, L, R)
    w, V = torch.linalg.eigh(G)
    E = V[:, torch.argsort(-w)][:, :r]
    P = torch.eye(G.shape[0], dtype=G.dtype, device=G.device) - E @ E.T
    return gram_with(D, L, R, P), P


def random_spectrum_gram(G, seed=0):
    """Same eigenvalues as G, random eigenvectors. Isolates what G's *eigenvectors* contribute."""
    w = torch.linalg.eigvalsh(G).clamp_min(0)
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(G.shape[0], G.shape[0], generator=g).to(G))[0]
    return Q @ torch.diag(w) @ Q.T


def _fit_eval(Xtr, Ytr, Xte, Yte, ctr, cte, k):
    """Per-cluster least squares on train; held-out R² using train-fit maps."""
    num = 0.0
    for c in range(k):
        itr, ite = ctr == c, cte == c
        if ite.sum() == 0:
            continue
        if itr.sum() < Xtr.shape[1] + 1:            # too few points to fit: fall back to global
            A = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
        else:
            A = np.linalg.lstsq(Xtr[itr], Ytr[itr], rcond=None)[0]
        num += ((Xte[ite] @ A - Yte[ite]) ** 2).sum()
    den = ((Yte - Yte.mean(0)) ** 2).sum()
    return 1.0 - num / den


def surrogate_r2(Ztr, Zte, Xtr, Ytr, Xte, Yte, k, seed):
    """Cluster in the Z metric, assign held-out by nearest (cosine) centroid, score held-out R²."""
    Zn = Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True).clip(1e-12)
    km = KMeans(k, n_init=6, random_state=seed).fit(Zn)
    ctr = km.labels_
    Zen = Zte / np.linalg.norm(Zte, axis=1, keepdims=True).clip(1e-12)
    cen = km.cluster_centers_
    cen = cen / np.linalg.norm(cen, axis=1, keepdims=True).clip(1e-12)
    cte = np.argmax(Zen @ cen.T, 1)
    return _fit_eval(Xtr, Ytr, Xte, Yte, ctr, cte, k)


def global_r2(Xtr, Ytr, Xte, Yte):
    A = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
    return 1.0 - ((Xte @ A - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()


def random_cluster_r2(Xtr, Ytr, Xte, Yte, k, seed):
    rs = np.random.RandomState(seed)
    return _fit_eval(Xtr, Ytr, Xte, Yte, rs.randint(0, k, len(Xtr)), rs.randint(0, k, len(Xte)), k)
