#!/usr/bin/env python3
"""Exact-weight read/write geometry for physical residual subspaces."""
from __future__ import annotations


ATTENTION_READERS = ("c_q", "c_k", "c_q2", "c_k2", "c_v")


def projected_fraction(torch, matrix, basis, role):
    """Return Frobenius fraction and rank-normalized isotropic enrichment."""
    matrix, basis = matrix.detach().float(), basis.detach().float()
    if basis.ndim != 2 or matrix.ndim != 2:
        raise ValueError("matrix and basis must be rank two")
    ambient, rank = basis.shape
    if rank < 1 or rank > ambient:
        raise ValueError("invalid basis rank")
    identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
    if float((basis.T @ basis - identity).abs().max()) > 1e-5:
        raise ValueError("basis must be orthonormal")
    matrix = matrix.to(basis.device)
    if role == "writer":
        if matrix.shape[0] != ambient:
            raise ValueError("writer output is not in the basis ambient space")
        projected = basis.T @ matrix
    elif role == "reader":
        if matrix.shape[1] != ambient:
            raise ValueError("reader input is not in the basis ambient space")
        projected = matrix @ basis
    else:
        raise ValueError("role must be reader or writer")
    denominator = float(matrix.square().sum())
    if denominator <= 0:
        raise ValueError("zero weight matrix")
    fraction = float(projected.square().sum()) / denominator
    if fraction < -1e-7 or fraction > 1 + 1e-5:
        raise ValueError("projected energy fraction violates orthogonal bound")
    fraction = min(1.0, max(0.0, fraction))
    return {"fraction": fraction, "enrichment": fraction / (rank / ambient)}


def weight_interfaces(model):
    """Enumerate all 180 writer and 846 reader tensors with stable labels."""
    width = model.config.n_embd // model.config.n_head
    records = []
    for layer, block in enumerate(model.transformer.h):
        output = block.attn.c_proj.weight
        for head in range(model.config.n_head):
            sl = slice(head * width, (head + 1) * width)
            records.append(("writer", f"L{layer:02d}H{head:02d}:attn_out", output[:, sl]))
        records.append(("writer", f"MLP{layer:02d}:down", block.mlp.Down.weight))
        for attr in ATTENTION_READERS:
            matrix = getattr(block.attn, attr).weight
            for head in range(model.config.n_head):
                sl = slice(head * width, (head + 1) * width)
                records.append(("reader", f"L{layer:02d}H{head:02d}:{attr[2:]}", matrix[sl]))
        records.append(("reader", f"MLP{layer:02d}:left", block.mlp.Left.weight))
        records.append(("reader", f"MLP{layer:02d}:right", block.mlp.Right.weight))
    return records


def score_interfaces(torch, model, bases):
    records = []
    for role, label, matrix in weight_interfaces(model):
        record = {"role": role, "label": label}
        for name, basis in bases.items():
            score = projected_fraction(torch, matrix, basis, role)
            record[f"{name}_fraction"] = score["fraction"]
            record[f"{name}_enrichment"] = score["enrichment"]
        records.append(record)
    return records


def top_labels(records, role, key, count=20):
    selected = [record for record in records if record["role"] == role]
    return [record["label"] for record in sorted(selected, key=lambda row: (-row[key], row["label"]))[:count]]


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 1.0
