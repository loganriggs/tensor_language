"""Attention-simplification programs for bilin18 heads (v702 lineage).

A program replaces ONE head's off-diagonal pattern P(i, j) (j < i); the diagonal, values (incl. the lambda-mixed token branch) and c_proj stay
native. Programs:
  kernel      P := kappa(i - j)                                   kappa in R^513 (mean real pattern per offset, or fitted)
  runmean     P := -m / i                                          one number m (the running-mean subtractor, v626/v627)
  lowrank_r   P := kappa(i - j) + [P_r(i, j) - kappa_r(i - j)]      P_r = native pattern with the head's four QK maps SVD-truncated to rank r;
                                                                   kappa_r = its own positional mean; the content deviation rides on the exact kernel
  gate        P := kappa(i - j) A(tok_i) B(tok_j)                   three tables (blocks 0-2 only; v623/v624/v628)
  native      P unchanged
The patched squared_attention receives q, k, v, q2, k2 already computed by the native maps; lowrank programs recompute q/k from the captured
attention input with truncated maps (rms-norm and rotary as in the model).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def truncated_maps(attn, h, r, hd):
    """Rank-r truncations of the head's c_q, c_k, c_q2, c_k2 rows (each 128 x 1152)."""
    out = {}
    for name in ("c_q", "c_k", "c_q2", "c_k2"):
        W = getattr(attn, name).weight[h * hd:(h + 1) * hd, :].detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        out[name] = (U[:, :r] * S[:r]) @ Vh[:r]
    return out


def lowrank_pattern(attn, n, maps, hd, causal):
    """Native-form pattern for one head from truncated maps: rms-norm, rotary, product of the two bilinear forms, causal mask."""
    Bn, Tn, _ = n.shape
    q = F.rms_norm(n @ maps["c_q"].T, (hd,)); k = F.rms_norm(n @ maps["c_k"].T, (hd,))
    q2 = F.rms_norm(n @ maps["c_q2"].T, (hd,)); k2 = F.rms_norm(n @ maps["c_k2"].T, (hd,))
    cos, sin = attn.rotary(q[:, :, None, :])
    rot = lambda x: _apply_rotary(x[:, :, None, :], cos, sin)[:, :, 0]
    q, k, q2, k2 = rot(q), rot(k), rot(q2), rot(k2)
    pat = (torch.einsum("bqd,bkd->bqk", q, k) / hd) * (torch.einsum("bqd,bkd->bqk", q2, k2) / hd)
    return pat.masked_fill(~causal, 0.0)


def _apply_rotary(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


def offset_mean(pat, dmat, off, Tn):
    """Mean of pat over entries with the same offset (queries >= 8), returned as a [Tn + 1] kernel (fp32)."""
    qmask = (torch.arange(Tn, device=pat.device) >= 8)[:, None] & off
    dflat = dmat[qmask]; X = pat[:, qmask]                                   # [B, n]
    counts = torch.bincount(dflat, minlength=Tn + 1).double() * X.shape[0]
    sums = torch.zeros(Tn + 1, dtype=torch.float64, device=pat.device).index_add_(0, dflat, X.double().sum(0))
    return torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums)).float()


def apply_program(pat_h, program, ctx):
    """Return the replaced off-diagonal pattern for one head. ctx: dict with dmat, off (bool [T,T]), pos, idx, attn, n, hd, causal."""
    kind = program["kind"]; off = ctx["off"][None]; dmat = ctx["dmat"]
    if kind == "native":
        return pat_h
    if kind == "kernel":
        prog = program["kappa"][dmat][None].expand_as(pat_h)
    elif kind == "runmean":
        prog = (-program["m"] / ctx["pos"].clamp_min(1).float())[None, :, None].expand_as(pat_h)
    elif kind == "gate":
        idx = ctx["idx"]; prog = program["kappa"][dmat][None] * program["A"][idx][:, :, None] * program["B"][idx][:, None, :]
    elif kind == "lowrank":
        pr = lowrank_pattern(ctx["attn"], ctx["n"], program["maps"], ctx["hd"], ctx["causal"])
        prog = program["kappa"][dmat][None] + (pr - program["kappa_r"][dmat][None])
    else:
        raise ValueError(kind)
    return torch.where(off, prog, pat_h)
