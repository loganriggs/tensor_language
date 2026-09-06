"""Translate residual subspaces into exact attention/MLP weight contractions."""

# BQGATE: LIBRARY
from __future__ import annotations

import torch


class SubspaceWeightAtlasError(RuntimeError):
    pass


def orthonormal_basis(basis, *, tolerance=1e-6):
    """Validate and return a finite column-orthonormal ``[d, rank]`` basis."""
    value = torch.as_tensor(basis).float()
    if value.ndim != 2 or not 0 < value.shape[1] <= value.shape[0] or not torch.isfinite(value).all():
        raise SubspaceWeightAtlasError("basis must be finite, rank-two, and nonempty")
    gram = value.T @ value
    identity = torch.eye(value.shape[1], dtype=value.dtype, device=value.device)
    if float((gram - identity).abs().max()) > tolerance:
        raise SubspaceWeightAtlasError("basis columns are not orthonormal")
    return value


def _slice_rows(weight, head, head_dim):
    return weight[head * head_dim:(head + 1) * head_dim].float()


def attention_subspace_factors(attention, source_basis, target_basis=None):
    """Return exact per-head read/write factors between two residual subspaces.

    ``q/k/q2/k2/v`` are the weight reads of the source basis. ``o`` is the target
    subspace's read of the head output. ``ov`` is their contracted linear value path.
    Frobenius norms of these objects are invariant to orthogonal basis rotations.
    """
    source = orthonormal_basis(source_basis)
    target = source if target_basis is None else orthonormal_basis(target_basis)
    if source.shape[0] != target.shape[0]:
        raise SubspaceWeightAtlasError("source and target ambient dimensions differ")
    n_head = int(attention.n_head)
    head_dim = int(attention.head_dim)
    width = n_head * head_dim
    names = [name for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v")
             if hasattr(attention, name)]
    if width != source.shape[0] or not hasattr(attention, "c_proj"):
        raise SubspaceWeightAtlasError("attention shape does not match subspace")
    output_weight = attention.c_proj.weight.detach().float()
    result = {}
    for head in range(n_head):
        factors = {name.removeprefix("c_"): _slice_rows(
            getattr(attention, name).weight.detach(), head, head_dim) @ source
            for name in names}
        output_slice = output_weight[:, head * head_dim:(head + 1) * head_dim]
        factors["o"] = target.T @ output_slice
        factors["ov"] = factors["o"] @ factors["v"]
        factors["scores"] = {name: float(torch.linalg.matrix_norm(value))
                             for name, value in factors.items()}
        result[head] = factors
    return result


def map_head_subspace_to_residual(attention, head, head_basis):
    """Map a head-output subspace through its exact output projection and orthogonalize."""
    head_basis = orthonormal_basis(head_basis)
    head = int(head)
    if head_basis.shape[0] != int(attention.head_dim) or not 0 <= head < int(attention.n_head):
        raise SubspaceWeightAtlasError("head basis or head index is invalid")
    start = head * int(attention.head_dim)
    output_slice = attention.c_proj.weight.detach().float()[:, start:start + int(attention.head_dim)]
    mapped = output_slice @ head_basis
    left, singular, _right = torch.linalg.svd(mapped, full_matrices=False)
    keep = singular > singular.max().clamp_min(1e-30) * 1e-6
    if not bool(keep.any()):
        raise SubspaceWeightAtlasError("head subspace is annihilated by output projection")
    return left[:, keep], singular[keep]


def mlp_subspace_tensor(mlp, source_basis, target_basis=None):
    """Return the exact quadratic weight tensor restricted between residual subspaces.

    For a bilinear MLP and ``x=U c``, the non-bias target coordinates equal
    ``einsum('aij,i,j->a', tensor, c, c)``. Squared MLPs use the same formula with
    identical left/right factors.
    """
    source = orthonormal_basis(source_basis)
    target = source if target_basis is None else orthonormal_basis(target_basis)
    if hasattr(mlp, "Left") and hasattr(mlp, "Right") and hasattr(mlp, "Down"):
        left = mlp.Left.weight.detach().float() @ source
        right = mlp.Right.weight.detach().float() @ source
        down = target.T @ mlp.Down.weight.detach().float()
    elif hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
        left = mlp.c_fc.weight.detach().float() @ source
        right = left
        down = target.T @ mlp.c_proj.weight.detach().float()
    else:
        raise SubspaceWeightAtlasError("unsupported MLP weight factorization")
    tensor = torch.einsum("an,ni,nj->aij", down, left, right)
    return {"left": left, "right": right, "down": down, "tensor": tensor,
            "scores": {"left": float(torch.linalg.matrix_norm(left)),
                       "right": float(torch.linalg.matrix_norm(right)),
                       "down": float(torch.linalg.matrix_norm(down)),
                       "tensor": float(torch.linalg.vector_norm(tensor))}}
