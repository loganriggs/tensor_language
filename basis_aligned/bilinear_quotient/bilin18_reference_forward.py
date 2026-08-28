#!/usr/bin/env python3
"""Independent, data-free mathematical reference forward for bilin18."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rotary_tables(sequence, head_dim, device, dtype):
    inverse = 1.0/(10000**(torch.arange(0, head_dim, 2, dtype=torch.float32)/head_dim))
    frequencies = torch.outer(torch.arange(sequence, dtype=torch.float32), inverse)
    return (frequencies.cos().bfloat16().to(device=device, dtype=dtype),
            frequencies.sin().bfloat16().to(device=device, dtype=dtype))


def rotate(x, cosine, sine):
    half = x.shape[-1]//2
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first*cosine+second*sine,
                      -first*sine+second*cosine), -1)


@torch.no_grad()
def reference_forward(model, token_ids):
    """Literal architecture equations, independent of Block/Attention.forward."""
    batch, sequence = token_ids.shape
    width = model.config.n_embd
    x = F.rms_norm(model.transformer.wte(token_ids), (width,))
    x0 = x; shared_value = None
    for block in model.transformer.h:
        x = block.lambdas[0]*x+block.lambdas[1]*x0
        normalized = F.rms_norm(x, (width,))
        attention = block.attn
        heads, head_dim = attention.n_head, attention.head_dim
        cosine, sine = rotary_tables(sequence, head_dim, token_ids.device, x.dtype)
        cosine, sine = cosine[None, :, None, :], sine[None, :, None, :]

        def qk(linear):
            value = linear(normalized).view(batch, sequence, heads, head_dim)
            return rotate(F.rms_norm(value, (head_dim,)), cosine, sine)

        value = attention.c_v(normalized).view(batch, sequence, heads, head_dim)
        if shared_value is None:
            shared_value = value
        mixed_value = (1-attention.lamb)*value+attention.lamb*shared_value.view_as(value)
        q, k = qk(attention.c_q), qk(attention.c_k)
        q2, k2 = qk(attention.c_q2), qk(attention.c_k2)
        score1 = torch.einsum("bqhd,bkhd->bhqk", q, k)/head_dim
        score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2)/head_dim
        causal = torch.tril(torch.ones(sequence, sequence, device=token_ids.device,
                                       dtype=torch.bool))
        pattern = (score1*score2).masked_fill(~causal, 0.0)
        routed = torch.einsum("bhqk,bkhd->bqhd", pattern, mixed_value)
        x = x+attention.c_proj(routed.reshape(batch, sequence, width))
        mlp = block.mlp
        z = F.rms_norm(x, (width,))
        x = x+mlp.Down(mlp.Left(z)*mlp.Right(z))+mlp.Down_bias
    logits = model.lm_head(F.rms_norm(x, (width,)))
    return 30*torch.tanh(logits/30)
