"""Exact donor-state generator at the earlier post-attention7 boundary.

Inputs: g7 [batch, D] and normalized initial token embedding [batch, D].
The eventual package must own the embedding table and all weights listed below.
This module alone is not a packaged or behaviorally verified extraction.
"""
import torch
import torch.nn.functional as F


def generate(p, g7, initial):
    z = F.rms_norm(g7, (g7.shape[-1],))
    hidden = (z @ p['left'].T) * (z @ p['right'].T)
    mlp = hidden @ p['down'].T + p['bias']
    raw = p['lambda8'][0] * (g7 + mlp) + p['lambda8'][1] * initial
    return F.rms_norm(raw, (raw.shape[-1],))


def source_gram(sources):
    """Ordered source Gram; retain both cross orders in the RMS denominator."""
    return torch.einsum('...sd,...td->...st', sources, sources) / sources.shape[-1]
