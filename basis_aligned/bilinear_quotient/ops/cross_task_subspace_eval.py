"""Gauge-invariant cross-task causal-subspace comparisons."""

# BQGATE: LIBRARY
from __future__ import annotations

import torch


class CrossTaskSubspaceError(RuntimeError):
    pass


def energy_basis(matrix, *, retained=0.95):
    """Smallest orthonormal right-singular subspace retaining frozen fit energy."""
    value = torch.as_tensor(matrix).float()
    if (value.ndim != 2 or min(value.shape) < 1 or not 0 < retained <= 1
            or not torch.isfinite(value).all()):
        raise CrossTaskSubspaceError("matrix or retained-energy target is invalid")
    _left, singular, vh = torch.linalg.svd(value, full_matrices=False)
    energy = singular.square()
    if float(energy.sum()) <= 0:
        raise CrossTaskSubspaceError("matrix has no energy")
    rank = int((energy.cumsum(0) < float(retained) * energy.sum()).sum()) + 1
    basis = torch.linalg.qr(vh[:rank].T).Q
    explained = float(energy[:rank].sum() / energy.sum())
    return basis, singular, explained


def projection_energy(matrix, basis):
    """Fraction of matrix energy retained by a column-orthonormal basis."""
    value, q = torch.as_tensor(matrix).float(), torch.as_tensor(basis).float()
    if value.ndim != 2 or q.ndim != 2 or value.shape[1] != q.shape[0]:
        raise CrossTaskSubspaceError("matrix and basis shapes do not compose")
    denominator = value.square().sum()
    if float(denominator) <= 0:
        raise CrossTaskSubspaceError("matrix has no energy")
    return float((value @ q).square().sum() / denominator)


def principal_cosines(first, second):
    """Singular values of ``U'V``: invariant principal-angle cosines."""
    left, right = torch.as_tensor(first).float(), torch.as_tensor(second).float()
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise CrossTaskSubspaceError("basis shapes are incompatible")
    return torch.linalg.svdvals(left.T @ right)


def shared_midpoint_basis(first, second, *, cosine_threshold=0.8):
    """Canonical midpoint modes for principal pairs above a preregistered cosine bar."""
    left, right = torch.as_tensor(first).float(), torch.as_tensor(second).float()
    cross_left, singular, cross_right_t = torch.linalg.svd(left.T @ right, full_matrices=False)
    keep = singular >= float(cosine_threshold)
    if not bool(keep.any()):
        return left.new_zeros((left.shape[0], 0)), singular
    modes_left = left @ cross_left[:, keep]
    modes_right = right @ cross_right_t.T[:, keep]
    signs = (modes_left * modes_right).sum(0).sign()
    signs[signs == 0] = 1
    midpoint = modes_left + modes_right * signs
    return torch.linalg.qr(midpoint).Q, singular
