#!/usr/bin/env python3
"""Factor-only invariants of a partially symmetric quadratic tensor.

For ``T(z,z) = sum_j c_j (a_j^T z)(b_j^T z)``, only
``S_j = (a_j b_j^T + b_j a_j^T)/2`` is observable.  These routines measure the
represented tensor without constructing its ``din x din x dout`` dense array.
"""
from __future__ import annotations

import torch


def _validate(A, B, C):
    A = A.detach().double().cpu()
    B = B.detach().double().cpu()
    C = C.detach().double().cpu()
    if A.ndim != 2 or B.shape != A.shape or C.ndim != 2 \
            or C.shape[0] != A.shape[1]:
        raise ValueError("expected A,B [din,components] and C [components,dout]")
    if not all(torch.isfinite(x).all() for x in (A, B, C)):
        raise ValueError("factors must be finite")
    return A, B, C


def symmetric_factor_gram(A, B):
    """Return ``G[j,k] = <sym(a_j b_j^T), sym(a_k b_k^T)>_F``."""
    A = A.detach().double().cpu()
    B = B.detach().double().cpu()
    if A.ndim != 2 or B.shape != A.shape or not torch.isfinite(A).all() \
            or not torch.isfinite(B).all():
        raise ValueError("A and B must be aligned finite matrices")
    aa = A.T @ A
    bb = B.T @ B
    ab = A.T @ B
    return .5*(aa*bb + ab*ab.T)


def tensor_frobenius_sq(A, B, C):
    """Squared Frobenius norm of the represented symmetric-input tensor."""
    A, B, C = _validate(A, B, C)
    gram = symmetric_factor_gram(A, B)
    value = torch.sum(gram*(C @ C.T))
    if value < -1e-9:
        raise RuntimeError("numerical violation of nonnegative tensor norm")
    return float(value.clamp_min(0))


def output_unfolding_gram(A, B, C):
    """Return the HOSVD output-mode Gram ``T_(out) T_(out)^T``."""
    A, B, C = _validate(A, B, C)
    gram = symmetric_factor_gram(A, B)
    result = C.T @ gram @ C
    return .5*(result+result.T)


def output_mode_spectrum(A, B, C, tolerance=1e-10):
    """Exact output-mode singular spectrum and basis-independent rank summaries."""
    gram = output_unfolding_gram(A, B, C)
    eigenvalues = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
    maximum = float(eigenvalues[0]) if eigenvalues.numel() else 0.0
    cutoff = tolerance*maximum
    rank = int((eigenvalues > cutoff).sum()) if maximum else 0
    total = float(eigenvalues.sum())
    stable_rank = total/maximum if maximum else 0.0
    probabilities = eigenvalues[eigenvalues > 0]/total if total else eigenvalues[:0]
    entropy_rank = float(torch.exp(-(probabilities*probabilities.log()).sum())) \
        if probabilities.numel() else 0.0
    singular_values = eigenvalues.sqrt()
    return {"singular_values": singular_values, "rank": rank,
            "stable_rank": stable_rank, "entropy_rank": entropy_rank,
            "frobenius_sq": total, "tolerance": tolerance}


def energy_rank(singular_values, fraction):
    """Smallest matrix rank retaining ``fraction`` of squared Frobenius energy."""
    singular = torch.as_tensor(singular_values, dtype=torch.float64)
    if singular.ndim != 1 or not torch.isfinite(singular).all() \
            or bool((singular < 0).any()) or not 0 < fraction <= 1:
        raise ValueError("need finite nonnegative singular values and 0 < fraction <= 1")
    energy = singular.square()
    total = float(energy.sum())
    if total == 0:
        return 0
    return int(torch.searchsorted(energy.cumsum(0), fraction*total).item()+1)


def best_rank_relative_frobenius_error(singular_values, rank):
    """Eckart--Young lower bound for any grouped rank-``rank`` approximation."""
    singular = torch.as_tensor(singular_values, dtype=torch.float64)
    if singular.ndim != 1 or rank < 0 or rank > singular.numel() \
            or not torch.isfinite(singular).all() or bool((singular < 0).any()):
        raise ValueError("invalid singular spectrum or rank")
    total = singular.square().sum()
    if total == 0:
        return 0.0
    return float((singular[rank:].square().sum()/total).sqrt())


def energy_majorization(left_singular, right_singular, tolerance=1e-10):
    """Compare normalized descending squared spectra in the majorization order."""
    left = torch.as_tensor(left_singular, dtype=torch.float64).square()
    right = torch.as_tensor(right_singular, dtype=torch.float64).square()
    if left.ndim != 1 or right.ndim != 1 or not left.numel() or not right.numel() \
            or not torch.isfinite(left).all() or not torch.isfinite(right).all() \
            or float(left.sum()) <= 0 or float(right.sum()) <= 0:
        raise ValueError("spectra must be nonempty, finite, and nonzero")
    width = max(left.numel(), right.numel())
    left = torch.nn.functional.pad(left/left.sum(), (0, width-left.numel())).sort(
        descending=True).values
    right = torch.nn.functional.pad(right/right.sum(), (0, width-right.numel())).sort(
        descending=True).values
    difference = left.cumsum(0)-right.cumsum(0)
    left_dominates = bool((difference >= -tolerance).all())
    right_dominates = bool((difference <= tolerance).all())
    if left_dominates and right_dominates:
        relation = "equal"
    elif left_dominates:
        relation = "left_majorizes_right"
    elif right_dominates:
        relation = "right_majorizes_left"
    else:
        relation = "incomparable_crossing"
    signs = torch.sign(difference[torch.abs(difference) > tolerance])
    crossings = int((signs[1:] != signs[:-1]).sum()) if signs.numel() > 1 else 0
    return {"relation": relation, "strict_sign_crossings": crossings,
            "maximum_left_cumulative_advantage": float(difference.max()),
            "maximum_right_cumulative_advantage": float((-difference).max()),
            "compared_modes": width, "tolerance": tolerance}


def explicit_symmetric_tensor(A, B, C):
    """Small-test oracle; never use for the 1152-dimensional production model."""
    A, B, C = _validate(A, B, C)
    return .5*(torch.einsum("ij,kj,jo->iko", A, B, C)
               + torch.einsum("ij,kj,jo->iko", B, A, C))
