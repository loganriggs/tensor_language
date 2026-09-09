#!/usr/bin/env python3
"""Exact, layer/head-generic source terms for Bilin18 attention.

For one head and final query, ``p[b, k]`` is the product of the two QK
scores and ``u[b, k]`` is the source value after that head's slice of the
output projection.  Their product is therefore the exact residual-stream
write contributed by source ``k``.
"""

from __future__ import annotations

import sys


SOURCE_FACTORS = ("q", "k", "q2", "k2", "u")


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
        if not {"p", "u", "head"}.issubset(factors):
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


def mixed_source_terms(native, donor, selected, torch):
    """Evaluate exact per-source writes for a Q/K/Q2/K2/U factor mixture.

    Q and Q2 are final-query vectors with shape ``[batch, head_width]``;
    K and K2 are source vectors ``[batch, source, head_width]``; U is the
    output-projected source value ``[batch, source, residual_width]``.
    Inputs are already head-normalized/rotary-transformed where applicable.
    """
    selected = frozenset(selected)
    unknown = selected - set(SOURCE_FACTORS)
    if unknown:
        raise ValueError(f"unknown source factors: {sorted(unknown)}")
    required = set(SOURCE_FACTORS)
    for label, factors in (("native", native), ("donor", donor)):
        if not required.issubset(factors):
            raise ValueError(f"{label} factors must contain {sorted(required)}")
        q, k, q2, k2, u = (factors[name] for name in SOURCE_FACTORS)
        if (q.ndim != 2 or q2.shape != q.shape or k.ndim != 3 or k2.shape != k.shape
                or u.ndim != 3 or k.shape[0] != q.shape[0] or k.shape[2] != q.shape[1]
                or u.shape[:2] != k.shape[:2]):
            raise ValueError(f"{label} source factor shapes are inconsistent")
    if any(native[name].shape != donor[name].shape for name in SOURCE_FACTORS):
        raise ValueError("native and donor source factor shapes differ")
    if any(native[name].device != native["q"].device
           or donor[name].device != native["q"].device for name in SOURCE_FACTORS):
        raise ValueError("native and donor source factors must share one device")
    chosen = {
        name: donor[name] if name in selected else native[name]
        for name in SOURCE_FACTORS
    }
    width = chosen["q"].shape[1]
    score1 = torch.einsum("bd,btd->bt", chosen["q"], chosen["k"]) / width
    score2 = torch.einsum("bd,btd->bt", chosen["q2"], chosen["k2"]) / width
    return (score1 * score2).unsqueeze(-1) * chosen["u"]


def source_factor_mobius(native, donor, torch):
    """Return all 32 exact Möbius terms of the five-factor source game.

    Mask zero is the native per-source write. Every nonempty dividend sums to
    donor minus native, retaining source and residual dimensions.
    """
    values, dividends = {}, {}
    for mask in range(1 << len(SOURCE_FACTORS)):
        selected = tuple(
            name for index, name in enumerate(SOURCE_FACTORS) if mask & (1 << index)
        )
        values[mask] = mixed_source_terms(native, donor, selected, torch)
        dividend = values[mask].clone()
        submask = (mask - 1) & mask
        while submask:
            dividend -= dividends[submask]
            submask = (submask - 1) & mask
        if mask:
            dividend -= dividends[0]
        dividends[mask] = dividend
    return dividends


def factor_names(mask):
    """Canonical names for a source-factor bit mask."""
    return tuple(name for index, name in enumerate(SOURCE_FACTORS) if int(mask) & (1 << index))
