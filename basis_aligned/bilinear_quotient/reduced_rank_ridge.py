#!/usr/bin/env python3
"""Closed-form nested reduced-rank ridge regression from sufficient statistics."""

from __future__ import annotations

import torch


def fit_factors(centered_xtx, centered_xty, regularizer):
    """Return factors whose prefixes minimize penalized fit loss at each rank.

    For ``A=X'X+lambda I`` and ``C=X'Y``, completing the square reduces the
    rank-constrained problem to the Frobenius approximation of ``A^-1/2 C``.
    If that matrix is ``U S Vh``, the optimal coefficient is
    ``A^-1/2 U[:r] S[:r] Vh[:r]``.
    """
    gram = centered_xtx.detach().double().cpu()
    cross = centered_xty.detach().double().cpu()
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("centered_xtx must be square")
    if cross.ndim != 2 or cross.shape[0] != gram.shape[0]:
        raise ValueError("centered_xty has incompatible shape")
    if not regularizer > 0:
        raise ValueError("regularizer must be positive")
    metric = (gram+gram.T)/2 + regularizer*torch.eye(
        gram.shape[0], dtype=gram.dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    if not torch.isfinite(eigenvalues).all() or not bool((eigenvalues > 0).all()):
        raise ValueError("penalized input metric is not positive definite")
    inverse_sqrt = (eigenvectors*eigenvalues.rsqrt()) @ eigenvectors.T
    whitened_cross = inverse_sqrt @ cross
    u, singular, vh = torch.linalg.svd(whitened_cross, full_matrices=False)
    left = inverse_sqrt @ (u*singular)
    return left, vh, singular, eigenvalues


def coefficient(left, right, rank):
    if rank < 0 or rank > right.shape[0]:
        raise ValueError("rank outside fitted path")
    return left[:, :rank] @ right[:rank]


def objective_without_y_constant(weight, centered_xtx, centered_xty, regularizer):
    """Rank-comparable part of ``||Y-XW||_F^2 + lambda||W||_F^2``."""
    weight = weight.double()
    gram = centered_xtx.double()
    cross = centered_xty.double()
    return (torch.trace(weight.T @ gram @ weight)
            - 2*torch.sum(weight*cross)
            + regularizer*torch.sum(weight.square()))
