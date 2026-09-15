"""Pure tensor helpers for headwise squared-attention response folds."""
from __future__ import annotations

import torch


def projected_head_writes(q, k, q2, k2, value, output_weight):
    """Return [B,H,Q,O] writes whose head sum is the attention output.

    Inputs q/k/q2/k2/value use [B,T,H,D]. ``output_weight`` uses the native
    linear layout [O,H*D]. Q and K inputs must already include RMS and rotary.
    """
    if not (q.shape == k.shape == q2.shape == k2.shape == value.shape):
        raise ValueError("all attention factors must share [B,T,H,D]")
    if q.ndim != 4:
        raise ValueError("attention factors must be four-dimensional")
    batch,tokens,heads,width=q.shape
    if output_weight.ndim != 2 or output_weight.shape[1] != heads*width:
        raise ValueError("output weight must have shape [O,H*D]")
    score1=torch.einsum("bqhd,bkhd->bhqk",q,k)/width
    score2=torch.einsum("bqhd,bkhd->bhqk",q2,k2)/width
    causal=torch.ones(tokens,tokens,dtype=torch.bool,device=q.device).tril()
    pattern=(score1*score2).masked_fill(~causal,0)
    head_values=torch.einsum("bhqk,bkhd->bhqd",pattern,value)
    weights=output_weight.reshape(output_weight.shape[0],heads,width)
    return torch.einsum("bhqd,ohd->bhqo",head_values,weights)
