#!/usr/bin/env python3
"""Corpus-weighted extension of mlp0_hybrid_separability_lib (CPU, exact, zero forwards).

Token distributions are taken from the frozen terminal-copy-induction v2 row caches: natural = fit_natural.pt
rows (192 x 257 token ids), code = ood_code.pt rows (192 x 257). Each corpus supplies an empirical unigram
distribution over the trained vocabulary (49k token occurrences); the first/second 96 documents give the
within-corpus split halves used as the noise floor. Everything else (unit-rms token rows, isotropic context,
Wiener/rank-k bounds) follows the base library.
"""
from __future__ import annotations
import torch
import mlp0_hybrid_separability_lib as LIB

ROWCACHE = LIB.BLOB  # placeholder to keep import order explicit
NAT = "/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache_terminal_copy_induction_v2/fit_natural.pt"
CODE = "/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache_terminal_copy_induction_v2/ood_code.pt"


def unigram(rows, V=LIB.V_TRAINED):
    ids = rows.reshape(-1)
    ids = ids[(ids >= 0) & (ids < V)]
    w = torch.bincount(ids, minlength=V).double()
    return w / w.sum()


def corpus_weights():
    nat = torch.load(NAT, map_location="cpu", weights_only=False)["rows"]
    code = torch.load(CODE, map_location="cpu", weights_only=False)["rows"]
    return {"natural": unigram(nat), "code": unigram(code),
            "natural_h0": unigram(nat[:96]), "natural_h1": unigram(nat[96:]),
            "code_h0": unigram(code[:96]), "code_h1": unigram(code[96:]),
            "n_tokens": {"natural": int(nat.numel()), "code": int(code.numel())}}


def weighted_token_moments(p, w):
    mu = w @ p
    M = p.T @ (p * w[:, None])
    return M, M - torch.outer(mu, mu), mu


def weighted_cov_gT(L, R, p, w):
    """Centered covariance of g_T over the weighted token distribution (only tokens with w>0 are touched)."""
    idx = torch.nonzero(w > 0).squeeze(1)
    pw = p[idx]; ww = w[idx]
    h = L.shape[0]
    s1 = torch.zeros(h, dtype=torch.float64); s2 = torch.zeros(h, h, dtype=torch.float64)
    for i in range(0, pw.shape[0], LIB.CHUNK):
        pc = pw[i:i + LIB.CHUNK]; wc = ww[i:i + LIB.CHUNK]
        G = (pc @ L.T) * (pc @ R.T)
        s1 += (G * wc[:, None]).sum(0)
        s2 += G.T @ (G * wc[:, None])
    return s2 - torch.outer(s1, s1), s1


def wiener_map(S_tgt, S_nui, ridge=1e-10):
    tot = S_tgt + S_nui
    tr = tot.diagonal().mean()
    tot = tot + ridge * tr * torch.eye(tot.shape[0], dtype=tot.dtype)
    return torch.linalg.solve(tot, S_tgt.T).T          # S_tgt tot^{-1}


def residual_under(P, S_tgt, S_nui, D):
    """Output-metric residual fraction of fixed map P evaluated under stats (S_tgt, S_nui)."""
    I = torch.eye(P.shape[0], dtype=P.dtype)
    A = D @ (P - I); Bm = D @ P
    num = (A @ S_tgt @ A.T).diagonal().sum() + (Bm @ S_nui @ Bm.T).diagonal().sum()
    den = (D @ S_tgt @ D.T).diagonal().sum()
    return float(num / den)


def response_distance(Pa, Pb, Sigma_delta, D):
    """Codex's d_response: ||D(Pa-Pb)Sigma^{1/2}||_F / mean(||D Pa Sigma^{1/2}||_F, ||D Pb Sigma^{1/2}||_F)."""
    def fro(Q):
        M = D @ Q
        return float(torch.sqrt((M @ Sigma_delta @ M.T).diagonal().sum().clamp(min=0)))
    return fro(Pa - Pb) / (0.5 * (fro(Pa) + fro(Pb)))
