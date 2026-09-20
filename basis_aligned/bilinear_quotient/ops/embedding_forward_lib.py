"""Embedding-forward folding helpers (v609 lineage): exact single-token tables and Gram-only tensor folds for the early bilinear MLPs.

Single token t at position 0: rotary is the identity and the only key is the query's own, so every attention pattern is the scalar
(q.k/128)(q2.k2/128) of the token itself and every residual quantity up to the MLP-1 input is an exact table over the vocabulary
(verified in v609: rel-L2 2e-7 / 5e-7 against the model's own MLP-input hooks). Folds never form a 1152^3 tensor: all energies and
mode Grams come through 4608x4608 Grams of Down, L, R against a DxD second moment (the Aug-28 HOSVD trick).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

QUANTILES = (0.90, 0.95, 0.99)


def rms(x):
    return F.rms_norm(x, (x.size(-1),))


def attn_single(block, n, v1, H, hd):
    """Attention at a lone position 0. Returns (c_proj output, v1 to pass on, per-head scalar pattern)."""
    a = block.attn
    D = n.shape[-1]
    q = a.c_q(n).view(-1, H, hd); k = a.c_k(n).view(-1, H, hd); q2 = a.c_q2(n).view(-1, H, hd); k2 = a.c_k2(n).view(-1, H, hd)
    v = a.c_v(n).view(-1, H, hd)
    if v1 is None:
        v1 = v
    v = (1 - a.lamb) * v + a.lamb * v1
    score = ((rms(q) * rms(k)).sum(-1) / hd) * ((rms(q2) * rms(k2)).sum(-1) / hd)      # [N, H]
    y = (score[..., None] * v).reshape(-1, D)
    return a.c_proj(y), v1, score


@torch.no_grad()
def single_token_tables(model, batch=4096, dev="cuda"):
    """Exact pre-norm MLP inputs X0, X1 [V, D] and their source-family splits; returns forwards used (one per batch)."""
    blocks = model.transformer.h; b0, b1 = blocks[0], blocks[1]
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size
    lam0, lam1 = b0.lambdas.detach().float(), b1.lambdas.detach().float()
    E = model.transformer.wte.weight.detach().float()
    X0 = torch.empty(V, D, device=dev); fam0 = {k: torch.empty(V, D, device=dev) for k in ("r", "a0")}
    fam1 = {k: torch.empty(V, D, device=dev) for k in ("r", "a0", "m0", "a1")}
    forwards = 0
    for s in range(0, V, batch):
        ids = torch.arange(s, min(s + batch, V), device=dev)
        x0 = rms(E[ids])
        live0 = lam0[0] * x0 + lam0[1] * x0
        y0, v1, _ = attn_single(b0, rms(live0), None, H, hd)
        xa = live0 + y0
        fam0["r"][ids] = live0; fam0["a0"][ids] = y0; X0[ids] = xa
        n0 = rms(xa); w0 = b0.mlp.Down(b0.mlp.Left(n0) * b0.mlp.Right(n0)) + b0.mlp.Down_bias
        live1 = lam1[0] * (xa + w0) + lam1[1] * x0
        y1, _, _ = attn_single(b1, rms(live1), v1, H, hd)
        fam1["r"][ids] = lam1[0] * live0 + lam1[1] * x0; fam1["a0"][ids] = lam1[0] * y0; fam1["m0"][ids] = lam1[0] * w0; fam1["a1"][ids] = y1
        forwards += 1
    X1 = sum(fam1.values())
    return {"X0": X0, "X1": X1, "fam0": fam0, "fam1": fam1, "lambdas": (lam0, lam1), "forwards": forwards}


def unigram_weights(rows_path, V, smooth=0.1, dev="cuda"):
    rows = torch.load(rows_path, map_location="cpu")
    counts = torch.bincount(rows.reshape(-1).long(), minlength=V).double()
    p = counts + smooth
    return (p / p.sum()).to(dev), int(rows.numel()), int((counts > 0).sum())


def second_moment(Xa, Xb, p):
    """sum_t p_t x^a_t x^b_t^T in fp64."""
    return (Xa.double() * p[:, None]).T @ Xb.double()


def energy_ranks(evals, quantiles=QUANTILES):
    ev = evals.clamp_min(0).flip(0); c = ev.cumsum(0) / ev.sum()
    return {str(q): int((c < q).sum().item()) + 1 for q in quantiles}


def mlp_weights(block):
    return (block.mlp.Left.weight.detach().double(), block.mlp.Right.weight.detach().double(), block.mlp.Down.weight.detach().double())


def fold_ranks(block, Sigma, top=16):
    """Symmetric-form mode Grams and energies of T x_2 Sigma^(1/2) x_3 Sigma^(1/2) for the bilinear MLP tensor T[o,i,j] = sum_k Down[o,k] L[k,i] R[k,j]."""
    L, R, Dw = mlp_weights(block)
    Gd = Dw.T @ Dw
    GLL, GRR, GLR = L @ Sigma @ L.T, R @ Sigma @ R.T, L @ Sigma @ R.T
    e_unsym = float((Gd * GLL * GRR).sum()); e_cross = float((Gd * GLR * GLR.T).sum())
    M2 = 0.25 * (L.T @ (Gd * GRR) @ L + L.T @ (Gd * GLR.T) @ R + R.T @ (Gd * GLR) @ L + R.T @ (Gd * GLL) @ R)
    w, U = torch.linalg.eigh(Sigma); S_half = (U * w.clamp_min(0).sqrt()) @ U.T
    in_ev = torch.linalg.eigvalsh(S_half @ M2 @ S_half)
    out_ev = torch.linalg.eigvalsh(0.5 * Dw @ (GLL * GRR + GLR * GLR.T) @ Dw.T)
    return {"energy_sym": 0.5 * (e_unsym + e_cross), "energy_unsym": e_unsym, "input_ranks": energy_ranks(in_ev),
            "output_ranks": energy_ranks(out_ev), "input_evals_top": in_ev.flip(0)[:top].tolist()}
