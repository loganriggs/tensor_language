#!/usr/bin/env python3
"""Exact rank-one projective minimax center for construction-specific axes."""
from __future__ import annotations


class ProjectiveBisectorError(ValueError):
    pass


def _unit_column(torch, value, label):
    vector = value.float().reshape(-1)
    norm = vector.norm()
    if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise ProjectiveBisectorError(f"{label} must be finite and nonzero")
    return (vector / norm)[:, None]


def projective_minimax_center(torch, left, right):
    """Return sign-aligned bisector and exact worst squared overlap for two lines."""
    u, v = _unit_column(torch, left, "left"), _unit_column(torch, right, "right")
    if u.shape != v.shape:
        raise ProjectiveBisectorError("axes have different dimensions")
    cosine = float((u.T @ v).squeeze())
    if cosine < 0:
        v, cosine = -v, -cosine
    center = u + v
    center_norm = center.norm()
    if not bool(torch.isfinite(center_norm)) or float(center_norm) <= 0:
        raise ProjectiveBisectorError("sign-aligned center is degenerate")
    center = center / center_norm
    exact_bound = (1.0 + min(1.0, max(0.0, cosine))) / 2.0
    observed = min(float((u.T @ center).square()), float((v.T @ center).square()))
    return {
        "center": center,
        "absolute_axis_cosine": cosine,
        "exact_worst_squared_overlap": exact_bound,
        "observed_worst_squared_overlap": observed,
        "closure_abs_error": abs(observed - exact_bound),
    }
