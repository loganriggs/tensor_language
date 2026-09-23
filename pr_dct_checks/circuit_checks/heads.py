"""Per-head forward of bilin18's squared attention (jacclust.tt_model.CausalBilinearSelfAttention, squared_attn=True), with
optional freezing of individual heads at their clean per-head outputs. Replicates the module's forward line by line; parity
against `attn(x, v1)` is checked by `head_parity`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import einsum


def _rot(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3).type_as(x)


def bilinear_attn_forward(attn, x, v1=None, head_mask=None, clean_z=None, modes=None):
    """Returns (y, v1_out, z) with z the per-head outputs [B, H, T, D] before c_proj.
    head_mask: float/bool [H], 1 = hold that head at clean_z (or detach it if clean_z is None)."""
    assert attn.squared_attn, "only the squared-attention branch is replicated"
    B, T, C = x.size(); H, D = attn.n_head, attn.head_dim
    q = attn.c_q(x).view(B, T, H, D); k = attn.c_k(x).view(B, T, H, D)
    q2 = attn.c_q2(x).view(B, T, H, D); k2 = attn.c_k2(x).view(B, T, H, D); v = attn.c_v(x).view(B, T, H, D)
    if v1 is None:
        v1 = v
    v = (1 - attn.lamb) * v + attn.lamb * v1.view_as(v)
    modes = modes or {}
    if modes.get("values"):                      # hold the values at their clean value (zero tangent): only the pattern path is live
        v = v.detach()
    if modes.get("value_mask") is not None:      # per-head value freeze: mask [H], 1 = that head's values held at clean
        vm = modes["value_mask"].to(v.dtype).view(1, 1, H, 1)
        v = vm * v.detach() + (1 - vm) * v
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,)); q, k = _rot(q, cos, sin), _rot(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (D,)), F.rms_norm(k2, (D,)); q2, k2 = _rot(q2, cos, sin), _rot(k2, cos, sin)
    scores = einsum(q, k, "... q h d, ... k h d -> ... h q k"); scores2 = einsum(q2, k2, "... q h d, ... k h d -> ... h q k")
    if modes.get("a"):                           # first pattern factor (q.k) frozen
        scores = scores.detach()
    if modes.get("b"):                           # second pattern factor (q2.k2) frozen
        scores2 = scores2.detach()
    pattern = (scores / D) * (scores2 / D)
    if modes.get("pattern"):                     # whole pattern frozen: only the value path is live
        pattern = pattern.detach()
    causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    pattern = pattern.masked_fill(causal.logical_not(), 0.0)
    z = einsum(pattern, v, "... h q k, ... k h d -> ... h q d")                        # [B, H, T, D]
    if head_mask is not None:
        m = head_mask.to(z.dtype).view(1, H, 1, 1)
        frozen = z.detach() if clean_z is None else clean_z.to(z.dtype)
        z = m * frozen + (1 - m) * z
    y = attn.c_proj(z.transpose(1, 2).contiguous().view(B, T, C))
    return y, v1, z


def head_parity(attn, x, v1):
    """Max relative difference between the module's forward and this replica with no freezing."""
    with torch.no_grad():
        y_ref, v1_ref = attn(x, v1); y, v1_out, _ = bilinear_attn_forward(attn, x, v1)
    return float((y_ref - y).abs().max() / y_ref.abs().max().clamp_min(1e-30))
