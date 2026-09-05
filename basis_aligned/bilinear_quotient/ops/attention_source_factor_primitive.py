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


def replace_head_source_subset(native, donor, mask, mode: str, torch):
    """Return an exact mixed head after replacing a row-varying source subset.

    ``mask[b, k]`` chooses sources. ``score`` replaces only ``p``; ``value``
    replaces the full effective OV-projected ``u``; ``joint`` replaces both.
    The function operates below the head boundary and never mutates its inputs.
    """
    if mode not in {"score", "value", "joint"}:
        raise ValueError("source-subset mode must be score, value, or joint")
    for label, factors in (("native", native), ("donor", donor)):
        if set(factors) < {"p", "u", "head"}:
            raise ValueError(f"{label} factors must contain p, u, and head")
        p, u, head = factors["p"], factors["u"], factors["head"]
        if p.ndim != 2 or u.ndim != 3 or head.ndim != 2 \
                or u.shape[:2] != p.shape or head.shape != (p.shape[0], u.shape[2]):
            raise ValueError(f"{label} factor shapes are inconsistent")
    if native["p"].shape != donor["p"].shape \
            or native["u"].shape != donor["u"].shape \
            or native["head"].shape != donor["head"].shape:
        raise ValueError("native and donor factor shapes differ")
    if tuple(mask.shape) != tuple(native["p"].shape) or mask.dtype != torch.bool:
        raise ValueError("source subset mask must be boolean with shape [batch,sources]")
    if mask.device != native["p"].device or any(
        tensor.device != native["p"].device
        for factors in (native, donor) for tensor in (factors["p"], factors["u"], factors["head"])
    ):
        raise ValueError("source subset factors and mask must share one device")
    chosen_p = donor["p"] if mode in {"score", "joint"} else native["p"]
    chosen_u = donor["u"] if mode in {"value", "joint"} else native["u"]
    weights = mask.to(native["p"].dtype)
    old = torch.einsum("bk,bkd->bd", native["p"] * weights, native["u"])
    new = torch.einsum("bk,bkd->bd", chosen_p * weights, chosen_u)
    return native["head"] - old + new


def install_source_terms(write, factors, final_positions, source_positions, replacement_terms, torch):
    """Replace exactly one head/source term at each row's final query."""
    native = source_terms(factors, source_positions, torch)
    if replacement_terms.shape != native.shape:
        raise ValueError("replacement term has the wrong shape")
    write = write.clone()
    rows = torch.arange(write.shape[0], device=write.device)
    write[rows, final_positions] += (replacement_terms - native).to(write.dtype)
    return write
