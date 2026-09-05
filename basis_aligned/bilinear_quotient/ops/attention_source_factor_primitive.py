#!/usr/bin/env python3
"""Exact, layer/head-generic source terms for Bilin18 attention.

For one head and final query, ``p[b, k]`` is the product of the two QK
scores and ``u[b, k]`` is the source value after that head's slice of the
output projection.  Their product is therefore the exact residual-stream
write contributed by source ``k``.
"""

from __future__ import annotations

import sys


def _linear(value, weight, F):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def replay_attention_with_source_factors(
    state, first_value, attention, final_positions, head_index: int, torch, F,
    *, include_qk_factors: bool = False,
):
    """Replay an attention module and expose exact source factors for one head."""
    batch, length, width = state.shape
    heads = 9
    if width % heads:
        raise ValueError("residual width is not divisible by nine heads")
    head_width = width // heads
    if not 0 <= head_index < heads:
        raise ValueError("head index is outside [0, 9)")
    if tuple(final_positions.shape) != (batch,):
        raise ValueError("final_positions must have one entry per row")

    def projected(layer):
        return _linear(state, layer.weight, F).view(batch, length, heads, head_width)

    q, k = projected(attention.c_q), projected(attention.c_k)
    q2, k2 = projected(attention.c_q2), projected(attention.c_k2)
    raw_value = projected(attention.c_v)
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value.view_as(raw_value)
    cos, sin = attention.rotary(q)
    apply_rotary = sys.modules[type(attention).__module__].apply_rotary_emb
    q = apply_rotary(F.rms_norm(q, (head_width,)), cos, sin)
    k = apply_rotary(F.rms_norm(k, (head_width,)), cos, sin)
    q2 = apply_rotary(F.rms_norm(q2, (head_width,)), cos, sin)
    k2 = apply_rotary(F.rms_norm(k2, (head_width,)), cos, sin)
    pattern = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_width
    pattern *= torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head_width
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    pattern = pattern.masked_fill(~causal, 0)
    all_heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    flat = all_heads.transpose(1, 2).contiguous().view(batch, length, width)
    write = _linear(flat, attention.c_proj.weight, F)

    rows = torch.arange(batch, device=state.device)
    p = pattern[rows, head_index, final_positions].float()
    head_slice = attention.c_proj.weight[
        :, head_index * head_width:(head_index + 1) * head_width
    ]
    u = _linear(value[:, :, head_index].float(), head_slice.float(), F)
    head_write = torch.einsum("bk,bkd->bd", p, u)
    factors = {"p": p, "u": u, "head": head_write}
    if include_qk_factors:
        factors.update({
            "q": q[rows, final_positions, head_index].float(),
            "k": k[:, :, head_index].float(),
            "q2": q2[rows, final_positions, head_index].float(),
            "k2": k2[:, :, head_index].float(),
        })
    return write, factors


def source_terms(factors, source_positions, torch):
    """Return exact ``p*u`` terms for one source position per row."""
    batch = factors["p"].shape[0]
    if tuple(source_positions.shape) != (batch,):
        raise ValueError("source_positions must have one entry per row")
    rows = torch.arange(batch, device=factors["p"].device)
    return factors["p"][rows, source_positions].unsqueeze(-1) * factors["u"][rows, source_positions]


def install_source_terms(write, factors, final_positions, source_positions, replacement_terms, torch):
    """Replace exactly one head/source term at each row's final query."""
    native = source_terms(factors, source_positions, torch)
    if replacement_terms.shape != native.shape:
        raise ValueError("replacement term has the wrong shape")
    write = write.clone()
    rows = torch.arange(write.shape[0], device=write.device)
    write[rows, final_positions] += (replacement_terms - native).to(write.dtype)
    return write
