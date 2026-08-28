#!/usr/bin/env python3
"""Global orthogonal residual-basis gauge for RMS-normalized tensor programs."""
from __future__ import annotations

import torch


def _matrix(value, name):
    value = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite matrix")
    return value


def validate_orthogonal(frame, tolerance=1e-10):
    frame = _matrix(frame, "frame")
    if frame.shape[0] != frame.shape[1]:
        raise ValueError("frame must be square")
    error = torch.linalg.matrix_norm(frame.T@frame-torch.eye(frame.shape[0]), ord=2)
    if float(error) > tolerance:
        raise ValueError("frame is not orthogonal")
    return frame


def canonical_frame(anchor, relative_gap_tolerance=1e-10):
    """Fix the generic global O(D) gauge from a tall residual-space anchor.

    Returns ``Q`` such that ``anchor @ Q`` is invariant when every residual-space
    object is first rotated by an arbitrary orthogonal matrix. Repeated or vanishing
    singular values are rejected rather than assigned an unstable canonical frame.
    """
    anchor = _matrix(anchor, "anchor")
    if anchor.shape[0] < anchor.shape[1] or relative_gap_tolerance <= 0:
        raise ValueError("anchor must be tall and tolerance positive")
    _, singular, vh = torch.linalg.svd(anchor, full_matrices=False)
    scale = float(singular[0]) if singular.numel() else 0.0
    if not scale or float(singular[-1]) <= relative_gap_tolerance*scale:
        raise ValueError("anchor is rank deficient at the requested tolerance")
    if singular.numel() > 1 \
            and float((singular[:-1]-singular[1:]).min()) <= relative_gap_tolerance*scale:
        raise ValueError("anchor has a non-identifiable repeated singular stratum")
    frame = vh.T
    canonical = anchor@frame
    pivots = canonical.abs().argmax(dim=0)
    signs = torch.sign(canonical[pivots, torch.arange(canonical.shape[1])])
    if bool((signs == 0).any()):
        raise ValueError("canonical column sign is undefined")
    frame = frame*signs
    return {"frame": frame, "singular_values": singular,
            "canonical_anchor": anchor@frame, "pivot_rows": pivots}


def transform_read_weight(weight, frame):
    """Transform PyTorch ``[out,residual]`` weights that read residual state."""
    weight = _matrix(weight, "read weight")
    frame = validate_orthogonal(frame)
    if weight.shape[1] != frame.shape[0]:
        raise ValueError("read weight width differs from residual width")
    return weight@frame


def transform_write_weight(weight, frame):
    """Transform PyTorch ``[residual,in]`` weights that write residual state."""
    weight = _matrix(weight, "write weight")
    frame = validate_orthogonal(frame)
    if weight.shape[0] != frame.shape[0]:
        raise ValueError("write weight height differs from residual width")
    return frame.T@weight


def transform_residual_rows(rows, frame):
    """Transform embeddings, unembeddings, biases, tables, or residual states."""
    rows = _matrix(rows, "residual rows")
    frame = validate_orthogonal(frame)
    if rows.shape[1] != frame.shape[0]:
        raise ValueError("row width differs from residual width")
    return rows@frame


def transform_quadratic_factors(A, B, C, frame):
    """Transform ``(z@A)*(z@B)@C`` factors into the rotated residual frame."""
    A = _matrix(A, "A"); B = _matrix(B, "B"); C = _matrix(C, "C")
    frame = validate_orthogonal(frame)
    if A.shape != B.shape or A.shape[0] != frame.shape[0] \
            or C.shape != (A.shape[1], frame.shape[0]):
        raise ValueError("quadratic factors do not share the residual width")
    return frame.T@A, frame.T@B, C@frame
