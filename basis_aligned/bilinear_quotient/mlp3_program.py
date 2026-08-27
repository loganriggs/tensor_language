#!/usr/bin/env python3
"""Checkpoint-independent execution primitive for decoded MLP3 programs."""

from __future__ import annotations

import torch


SEMANTIC_ID = "mlp3|table[token]+ridge[attn3.c_proj,mlp2]"


def validate_decoded(decoded, *, vocabulary=50257, width=1152):
    if decoded.get("semantic_id") != SEMANTIC_ID:
        raise ValueError("decoded stream has wrong MLP3 semantic id")
    table = decoded["table"]
    weight = decoded["weight"]
    bias = decoded["bias"]
    if tuple(table.shape) != (vocabulary, width):
        raise ValueError("decoded MLP3 table has wrong shape")
    if tuple(weight.shape) != (2*width, width):
        raise ValueError("decoded MLP3 ridge has wrong shape")
    if tuple(bias.shape) != (width,):
        raise ValueError("decoded MLP3 bias has wrong shape")
    return decoded


def execute_decoded(decoded, token_ids, attn3_c_proj, mlp2_output):
    """Evaluate ``table[token] + [a3,m2] @ W + bias`` without checkpoint MLP3."""
    width = attn3_c_proj.shape[-1]
    validate_decoded(decoded, vocabulary=decoded["table"].shape[0], width=width)
    if token_ids.shape != attn3_c_proj.shape[:-1]:
        raise ValueError("token and attn3 shapes do not align")
    if mlp2_output.shape != attn3_c_proj.shape:
        raise ValueError("attn3 and mlp2 shapes do not align")
    device = attn3_c_proj.device
    dtype = attn3_c_proj.dtype
    table = decoded["table"].to(device=device, dtype=dtype)
    weight = decoded["weight"].to(device=device, dtype=dtype)
    bias = decoded["bias"].to(device=device, dtype=dtype)
    local = torch.cat([attn3_c_proj, mlp2_output], dim=-1)
    return table[token_ids] + local @ weight + bias
