"""Baseline derivative paths with selected MLP/attention write branches.

The native baseline is unchanged. Dropping a derivative branch is not an
ablation of the primal model. Residual skip paths and scales remain explicit.
"""
from full_suffix_readers import readout_reader, mlp_pullback
from finite_attention_readers import pullback as attention_pullback
import torch


def baseline_reader(blocks, trace, first, pairs, positions, eps, mlp_layers, attention_layers):
    count = len(blocks)
    if not set(mlp_layers).union(attention_layers).issubset(range(count)):
        raise ValueError('layer indices must be relative to supplied suffix')
    q = readout_reader(trace['states'][-1], trace['states'][-1], pairs, positions, eps)
    for k in reversed(range(count)):
        b = blocks[k]
        h = trace['mlp_inputs'][k]
        qh = mlp_pullback(b, h, h, q, eps) if k in mlp_layers else q
        if k in attention_layers:
            raw = trace['raw_attention_inputs'][k]
            qa = attention_pullback(b['attention'], raw, raw, first, b['mixture'], b['heads'],
                                    b['cos'], b['sin'], eps, b['head_eps'], qh)
            qh = qh + qa
        q = b['lambdas'][0] * qh
    return q


def single_attention_terms(blocks, trace, first, pairs, positions, eps):
    """Return all-MLP reader and exact terms with one attention Jacobian factor.

    Sum excludes paths with two or more attention write derivatives. Each term
    retains every residual/MLP branch; no expansion of the primal polynomial.
    """
    q = readout_reader(trace['states'][-1], trace['states'][-1], pairs, positions, eps)
    terms = torch.zeros((len(blocks),)+q.shape, dtype=q.dtype, device=q.device)
    for k in reversed(range(len(blocks))):
        b = blocks[k]
        h = trace['mlp_inputs'][k]
        qh = mlp_pullback(b, h, h, q, eps)
        if k+1 < len(blocks):
            active = terms[k+1:].reshape((-1,)+q.shape[1:])
            terms[k+1:] = (b['lambdas'][0]*mlp_pullback(b, h, h, active, eps)).reshape(terms[k+1:].shape)
        raw = trace['raw_attention_inputs'][k]
        terms[k] = b['lambdas'][0]*attention_pullback(b['attention'], raw, raw, first,
            b['mixture'], b['heads'], b['cos'], b['sin'], eps, b['head_eps'], qh)
        q = b['lambdas'][0]*qh
    return q, terms


def earliest_attention_reader(blocks, trace, first, pairs, positions, eps):
    """All MLP write derivatives, only the earliest attention write derivative.

    Downstream tokenwise derivatives need only the readout position. The earliest
    attention pullback then spreads the reader to earlier input positions.
    """
    full = readout_reader(trace['states'][-1], trace['states'][-1], pairs, positions, eps)
    batch = torch.arange(len(positions), device=positions.device)
    q = full[:, batch, positions]
    for k in reversed(range(len(blocks))):
        b = blocks[k]
        h = trace['mlp_inputs'][k][batch, positions]
        q = mlp_pullback(b, h, h, q, eps)
        if k == 0:
            qh = torch.zeros_like(full)
            qh[:, batch, positions] = q
            raw = trace['raw_attention_inputs'][k]
            qa = attention_pullback(b['attention'], raw, raw, first, b['mixture'], b['heads'],
                b['cos'], b['sin'], eps, b['head_eps'], qh)
            return b['lambdas'][0]*(qh+qa)
        q = b['lambdas'][0]*q
    raise ValueError('at least one block required')
