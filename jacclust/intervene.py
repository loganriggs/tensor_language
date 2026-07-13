"""Intervention validation on a real model (open problem 4).

Claim: a Jacobian cluster groups tokens on which the layer applied ~the same linear map.
Causal test, no labels needed:

    replace the bilinear MLP's output y(x) with a per-cluster LINEAR SURROGATE A_c x, where
      within-cluster : c = the token's own cluster        -> should barely hurt the LM
      across-cluster : c = a different cluster            -> should hurt a lot, IF clusters are mechanisms
      global         : one A for all tokens               -> the floor

    differential = CE(across) - CE(within)  in nats.

A metric whose clusters carry mechanism has a LARGE differential and a LOW within-CE.
Controls that could win: raw-x clusters, spectrum-matched G_rand clusters, and (for the differential)
random cluster assignment, whose within/across are the same partition by construction.

Every surrogate is fit on TRAIN tokens; CE is measured on HELD-OUT windows.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans


def ridge_fit(X, Y, lam=0.1):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def fit_cluster_maps(Ztr, Xtr, Ytr, k, seed, lam=0.1):
    """k-means in the Z metric (Euclidean: gain retained, per tick 6); ridge map per cluster."""
    km = KMeans(k, n_init=6, random_state=seed).fit(Ztr)
    A = np.zeros((k, Xtr.shape[1], Ytr.shape[1]))
    for c in range(k):
        m = km.labels_ == c
        A[c] = ridge_fit(Xtr[m], Ytr[m], lam) if m.sum() >= 5 else ridge_fit(Xtr, Ytr, lam)
    return km, A


@torch.no_grad()
def patched_ce(model, mlp, toks, assign_fn, A, mode, seed=0):
    """Replace the MLP's output with A_{c(x)} x. mode: 'within' | 'across' | 'global' | 'clean'."""
    orig = mlp.forward
    rs = np.random.RandomState(seed)
    k = A.shape[0]

    def fwd(x):
        h = mlp.norm(x)
        if mode == "clean":
            y = mlp.D(mlp.L(h) * mlp.R(h))
        else:
            flat = h.reshape(-1, h.shape[-1])
            if mode == "global":
                c = np.zeros(flat.shape[0], dtype=int)
                Ause = A[:1]
            else:
                c = assign_fn(flat)
                if mode == "across":
                    c = (c + 1 + rs.randint(0, k - 1, size=c.shape)) % k     # a DIFFERENT cluster
                Ause = A
            At = torch.tensor(Ause[c], dtype=flat.dtype, device=flat.device)  # (N, d_in, d_out)
            y = torch.einsum("nd,ndo->no", flat, At).reshape(h.shape[:-1] + (A.shape[2],))
        return x + y if mlp.residual == "add" else torch.lerp(x, y, mlp.scale)

    mlp.forward = fwd
    try:
        logits = model(toks[:, :-1])
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), toks[:, 1:].reshape(-1))
    finally:
        mlp.forward = orig
    return float(ce)
