#!/usr/bin/env python3
"""Shared exact math for the MLP0 hybrid-pair linear-separability probes (CPU, weights only, zero forwards).

R536's hybrid-pair addendum defines two portable targets on MLP0's 4608-dim product activation
g(p,q) = (L(p+q)) * (R(p+q)) = g_T(p) + g_I(p,q) + g_C(q):
  token target:   observed Dg = Dg_T + Dg_I  (donor token, base context),  target Dg_T
  context target: observed Dg = Dg_I + Dg_C  (donor context, base token),  target Dg_I
A DAS projector P must map the observed difference to the target. This library computes, in closed form under
a stated input model, the IRREDUCIBLE residual of the best LINEAR map (Wiener bound, any rank) and of the best
rank-k linear map (reduced-rank regression), in the W_D-weighted output metric (what downstream sees) and in
the raw product metric. Lower residual = more separable. These are lower bounds for the orthogonal-projector
class DAS actually optimizes, and for any downstream loss that is dominated by the MLP0 output difference.

Input model (stated, not fitted): token part p_t = wte_t / rms(wte_t) over the 50257 trained tokens, uniform;
context part q ~ zero-mean, E[qq^T] = rho^2 I (isotropic; Gaussian where 4th moments are needed), independent
of the token. rho = ||q||/||p|| is scanned. Under this model the target and nuisance differences are exactly
uncorrelated, so the Wiener map is P* = S_tgt (S_tgt + S_nui)^{-1}.
"""
from __future__ import annotations
import time
import numpy as np
import torch

BLOB = ("/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/"
        "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin")
V_TRAINED = 50257
CHUNK = 4096


def load_mlp0():
    sd = torch.load(BLOB, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    L = sd["transformer.h.0.mlp.Left.weight"].double()
    R = sd["transformer.h.0.mlp.Right.weight"].double()
    D = sd["transformer.h.0.mlp.Down.weight"].double()
    wte = sd["transformer.wte.weight"][:V_TRAINED].double()
    p = wte / wte.pow(2).mean(1, keepdim=True).sqrt()      # unit-rms token part (proxy for normalized input)
    return L, R, D, p


def token_moments(p):
    n = p.shape[0]
    mu = p.mean(0)
    M = (p.T @ p) / n                                        # uncentered 2nd moment
    S = M - torch.outer(mu, mu)                              # centered covariance
    return M, S, mu


def cov_gT(L, R, p):
    """Centered covariance over tokens of g_T(p) = (Lp)*(Rp), chunked, float64."""
    n = p.shape[0]
    h = L.shape[0]
    s1 = torch.zeros(h, dtype=torch.float64)
    s2 = torch.zeros(h, h, dtype=torch.float64)
    for i in range(0, n, CHUNK):
        pc = p[i:i + CHUNK]
        G = (pc @ L.T) * (pc @ R.T)
        s1 += G.sum(0)
        s2 += G.T @ G
    mu = s1 / n
    return s2 / n - torch.outer(mu, mu), mu


def cov_gI_given_token_moment(L, R, Mp, rho):
    """Cov of (L a)*(R q) + (L q)*(R a) with a ~ 2nd moment Mp (independent of q), E[qq^T] = rho^2 I."""
    LML = L @ Mp @ L.T
    LMR = L @ Mp @ R.T
    RMR = R @ Mp @ R.T
    LL = L @ L.T
    RR = R @ R.T
    LR = L @ R.T
    out = LML * RR + LMR * LR.T + LMR.T * LR + RMR * LL
    return rho ** 2 * out


def cov_gC_gaussian(L, R, rho):
    """Cov of g_C(q) = (Lq)*(Rq) for Gaussian q with E[qq^T] = rho^2 I (Isserlis)."""
    LL = L @ L.T
    RR = R @ R.T
    LR = L @ R.T
    return rho ** 4 * (LL * RR + LR * LR.T)


def wiener_residuals(S_tgt, S_nui, D, ks, ridge=1e-10):
    """Irreducible residual fractions of the best linear map from (tgt+nui) to tgt.

    Returns dict with product-metric and D-output-metric residual fractions for the unconstrained map and the
    reduced-rank ladder ks (output metric). Residual fraction: 1 = nothing recovered, 0 = exact separation.
    """
    tot = S_tgt + S_nui
    tr = tot.diagonal().mean()
    tot = tot + ridge * tr * torch.eye(tot.shape[0], dtype=tot.dtype)
    ew, ev = torch.linalg.eigh(tot)
    ew = ew.clamp(min=ridge * tr)
    inv_half = ev @ torch.diag(ew.rsqrt()) @ ev.T
    # product metric
    B = S_tgt @ inv_half                                     # S_tgt tot^{-1/2}
    num_p = (S_tgt.diagonal().sum() - (B * B).sum()).item()
    den_p = S_tgt.diagonal().sum().item()
    # output metric: Y = D t, X = tgt+nui ; residual_k = tr(D S_tgt D^T) - sum_{i<=k} sigma_i^2(D S_tgt tot^{-1/2})
    A = D @ B                                                # 1152 x 4608
    sig2 = torch.linalg.svdvals(A).pow(2)
    den_o = (D @ S_tgt @ D.T).diagonal().sum().item()
    cum = torch.cumsum(sig2, 0)
    ladder = {}
    for k in ks:
        kk = min(k, sig2.numel())
        ladder[int(k)] = float((den_o - cum[kk - 1].item()) / den_o)
    res_o_full = float((den_o - cum[-1].item()) / den_o)
    return {"residual_fraction_product_metric": num_p / den_p,
            "residual_fraction_output_metric": res_o_full,
            "residual_fraction_output_metric_rank_ladder": ladder,
            "target_energy_output_metric": den_o,
            "nuisance_energy_output_metric": (D @ S_nui @ D.T).diagonal().sum().item()}


def pure_target_rank_ladder(S_tgt, D, ks):
    """Output-metric energy NOT captured by the top-k eigen-directions of D S_tgt D^T (rho=0 reference)."""
    Y = D @ S_tgt @ D.T
    Y = 0.5 * (Y + Y.T)
    lam = torch.linalg.eigvalsh(Y).clamp(min=0).flip(0)
    lam_n = lam / lam.sum()
    eff = float(torch.exp(-(lam_n * torch.log(lam_n + 1e-300)).sum()))
    cum = torch.cumsum(lam_n, 0)
    return {int(k): float(1.0 - cum[min(k, lam.numel()) - 1].item()) for k in ks}, eff


def mc_check_token_target(L, R, p, rho, n, seed):
    """Monte Carlo: traces of Cov(Dg_T), Cov(Dg_I) and their normalized cross inner product (should be ~0)."""
    g = torch.Generator().manual_seed(seed)
    d = p.shape[1]
    i = torch.randint(0, p.shape[0], (n,), generator=g)
    j = torch.randint(0, p.shape[0], (n,), generator=g)
    pb, pd_ = p[i], p[j]
    q = torch.randn(n, d, dtype=torch.float64, generator=g) * rho
    dT = (pd_ @ L.T) * (pd_ @ R.T) - (pb @ L.T) * (pb @ R.T)
    dp = pd_ - pb
    dI = (dp @ L.T) * (q @ R.T) + (q @ L.T) * (dp @ R.T)
    return _mc_stats(dT, dI)


def mc_check_context_target(L, R, p, rho, n, seed):
    g = torch.Generator().manual_seed(seed)
    d = p.shape[1]
    i = torch.randint(0, p.shape[0], (n,), generator=g)
    pb = p[i]
    qb = torch.randn(n, d, dtype=torch.float64, generator=g) * rho
    qd = torch.randn(n, d, dtype=torch.float64, generator=g) * rho
    dq = qd - qb
    dI = (pb @ L.T) * (dq @ R.T) + (dq @ L.T) * (pb @ R.T)
    dC = (qd @ L.T) * (qd @ R.T) - (qb @ L.T) * (qb @ R.T)
    return _mc_stats(dI, dC)


def _mc_stats(dt, dn):
    et = dt.pow(2).sum(1).mean().item()          # E||D tgt||^2 = 2 tr(Cov tgt) for difference constructions
    en = dn.pow(2).sum(1).mean().item()
    cross = (dt * dn).sum(1).mean().item()
    return {"mc_E_sq_target": et, "mc_E_sq_nuisance": en,
            "mc_cross_normalized": cross / np.sqrt(et * en)}
