"""Tensor-only suffix with explicit per-position attention backgrounds.

This is an exact executor for a conditional program, not an attention generator.
Every supplied tensor, including backgrounds and initial states, is a charged port.
"""
import torch
import torch.nn.functional as F


def execute(blocks, start, initial_embedding, attention_writes, readout, eps):
    """Apply affine residual/RMS/bilinear blocks and selected softcapped readout.

    blocks: dictionaries with left/right/down/bias/lambdas tensors.
    start, initial_embedding: (..., d); attention_writes: one (..., d) per block.
    readout: (outputs, d), e.g. two fixed is/are readers. No model object required.
    eps must match the original model dtype even when testing in float64.
    """
    if len(blocks) != len(attention_writes):
        raise ValueError('one attention background required per block')
    if start.shape != initial_embedding.shape:
        raise ValueError('start and embedding shapes differ')
    x = start
    for block, attention in zip(blocks, attention_writes):
        if attention.shape != x.shape:
            raise ValueError('attention background shape differs')
        h = block['lambdas'][0]*x + block['lambdas'][1]*initial_embedding
        h = h + attention
        n = F.rms_norm(h, (h.shape[-1],), eps=eps)
        products = F.linear(n, block['left'])*F.linear(n, block['right'])
        x = h + (F.linear(products, block['down']) + block['bias'])
    normalized = F.rms_norm(x, (x.shape[-1],), eps=eps)
    return 30*torch.tanh(F.linear(normalized, readout)/30), x


def response(blocks, base_start, edited_start, initial_embedding, attention_writes,
             readout, eps, include_quadratic=True):
    """Finite-response chain with exact RMS geometry and optional product omission.

    Bias cancels from each local response but stays in the baseline recurrence.
    Omitting the quadratic product is not a full Taylor approximation: RMS retains
    the complete edited norm. Background preparation costs are not eliminated.
    """
    if len(blocks) != len(attention_writes):
        raise ValueError('one attention background required per block')
    base = base_start
    delta = edited_start - base_start
    for block, attention in zip(blocks, attention_writes):
        h = block['lambdas'][0]*base + block['lambdas'][1]*initial_embedding + attention
        d = block['lambdas'][0]*delta
        lh, rh = F.linear(h, block['left']), F.linear(h, block['right'])
        ld, rd = F.linear(d, block['left']), F.linear(d, block['right'])
        p0 = F.linear(lh*rh, block['down'])
        difference = ld*rh + lh*rd
        if include_quadratic:
            difference = difference + ld*rd
        dp = F.linear(difference, block['down'])
        s0 = h.square().mean(-1, keepdim=True) + eps
        s1 = (h+d).square().mean(-1, keepdim=True) + eps
        delta = d + dp/s1 + p0*(1/s1-1/s0)
        base = h + p0/s0 + block['bias']
    def logits(x):
        return 30*torch.tanh(F.linear(F.rms_norm(x, (x.shape[-1],), eps=eps), readout)/30)
    return logits(base), logits(base+delta), delta
