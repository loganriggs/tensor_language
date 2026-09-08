"""Exact earlier-writer maps into the physical L11H3 value reader."""

# BQGATE: LIBRARY
from __future__ import annotations


def writer_interfaces(model):
    width = model.config.n_embd // model.config.n_head
    writers = [("embedding", model.transformer.wte.weight.T)]
    for layer, block in enumerate(model.transformer.h[:11]):
        for head in range(model.config.n_head):
            sl = slice(head * width, (head + 1) * width)
            writers.append((f"L{layer:02d}H{head:02d}:attn_out", block.attn.c_proj.weight[:, sl]))
        writers.append((f"MLP{layer:02d}:down", block.mlp.Down.weight))
    return writers


def trace_one_covariance(torch, rows):
    rows = rows.detach().float()
    if rows.ndim != 2 or rows.shape[1] != 128 or not bool(torch.isfinite(rows).all()):
        raise ValueError("task tensor bank must be finite [rows,128]")
    denominator = rows.square().sum()
    if float(denominator) <= 0:
        raise ValueError("task tensor bank is zero")
    covariance = rows.T @ rows / denominator
    if abs(float(torch.trace(covariance)) - 1.0) > 1e-5:
        raise ValueError("task covariance trace changed")
    return covariance


def enrichment(torch, composed, covariance):
    composed = composed.detach().float()
    covariance = covariance.detach().float().to(composed.device)
    if composed.ndim != 2 or composed.shape[0] != 128 or covariance.shape != (128, 128):
        raise ValueError("composed writer or covariance shape changed")
    denominator = composed.square().sum()
    if float(denominator) <= 0:
        raise ValueError("composed writer is zero")
    fraction = torch.sum(composed * (covariance @ composed)) / denominator
    value = 128.0 * float(fraction)
    if not 0.0 <= value <= 128.0 + 1e-4:
        raise ValueError("enrichment violates covariance bounds")
    return value


def top_labels(records, key, count):
    return [row["label"] for row in sorted(records, key=lambda row: (-row[key], row["label"]))[:count]]


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 1.0
