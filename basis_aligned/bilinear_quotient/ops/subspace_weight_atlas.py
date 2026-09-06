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


def head_bank_value_read_map(attention, heads, bank_basis):
    """Return the exact map from residual input into a multi-head value subspace.

    ``bank_basis`` is expressed in the concatenated raw-value coordinates of
    ``heads``.  The returned ``[rank, d_model]`` matrix is
    ``U.T @ block_rows(c_v, heads)`` and is invariant up to a left orthogonal
    gauge rotation.
    """
    heads = tuple(int(head) for head in heads)
    if not heads or len(set(heads)) != len(heads):
        raise SubspaceWeightAtlasError("heads must be a nonempty unique sequence")
    head_dim, n_head = int(attention.head_dim), int(attention.n_head)
    if any(not 0 <= head < n_head for head in heads) or not hasattr(attention, "c_v"):
        raise SubspaceWeightAtlasError("head bank or attention value weights are invalid")
    basis = orthonormal_basis(bank_basis)
    if basis.shape[0] != len(heads) * head_dim:
        raise SubspaceWeightAtlasError("bank basis does not match concatenated head width")
    value_rows = torch.cat([_slice_rows(
        attention.c_v.weight.detach(), head, head_dim) for head in heads], dim=0)
    return basis.T @ value_rows


def map_head_bank_subspace_to_residual(attention, heads, bank_basis):
    """Map a concatenated multi-head output subspace through exact ``c_proj``."""
    heads = tuple(int(head) for head in heads)
    basis = orthonormal_basis(bank_basis)
    head_dim, n_head = int(attention.head_dim), int(attention.n_head)
    if (not heads or len(set(heads)) != len(heads)
            or any(not 0 <= head < n_head for head in heads)
            or basis.shape[0] != len(heads) * head_dim
            or not hasattr(attention, "c_proj")):
        raise SubspaceWeightAtlasError("head bank, basis, or output projection is invalid")
    output_weight = attention.c_proj.weight.detach().float()
    output_bank = torch.cat([
        output_weight[:, head * head_dim:(head + 1) * head_dim] for head in heads
    ], dim=1)
    mapped = output_bank @ basis
    left, singular, _right = torch.linalg.svd(mapped, full_matrices=False)
    keep = singular > singular.max().clamp_min(1e-30) * 1e-6
    if not bool(keep.any()):
        raise SubspaceWeightAtlasError("head-bank subspace is annihilated by output projection")
    return left[:, keep], singular[keep]


def attention_writer_to_read_map(attention, head, read_map):
    """Contract one attention head's output weights into a downstream read map."""
    read = torch.as_tensor(read_map).float()
    head, head_dim = int(head), int(attention.head_dim)
    if (read.ndim != 2 or not torch.isfinite(read).all()
            or read.shape[1] != int(attention.n_head) * head_dim
            or not 0 <= head < int(attention.n_head) or not hasattr(attention, "c_proj")):
        raise SubspaceWeightAtlasError("read map or attention writer is invalid")
    output = attention.c_proj.weight.detach().float()[:, head * head_dim:(head + 1) * head_dim]
    contraction = read @ output
    return {"contraction": contraction, "score": float(torch.linalg.matrix_norm(contraction))}


def mlp_writer_to_read_map(mlp, read_map):
    """Contract a bilinear/squared MLP output factor into a downstream read map."""
    read = torch.as_tensor(read_map).float()
    output = (mlp.Down.weight.detach().float() if hasattr(mlp, "Down")
              else mlp.c_proj.weight.detach().float() if hasattr(mlp, "c_proj") else None)
    if (read.ndim != 2 or not torch.isfinite(read).all() or output is None
            or read.shape[1] != output.shape[0]):
        raise SubspaceWeightAtlasError("read map or MLP writer is invalid")
    contraction = read @ output
    return {"contraction": contraction, "score": float(torch.linalg.matrix_norm(contraction))}


def mlp_writer_to_read_tensor(mlp, read_map):
    """Contract the complete quadratic MLP operation into a downstream read map.

    ``mlp_writer_to_read_map`` measures only whether the output factor can enter
    the read subspace.  That is a useful incidence test, but it cannot distinguish
    MLPs whose Left/Right factors respond differently to the live residual.  For
    a bilinear MLP this routine returns the exact tensor

    ``T[a,i,j] = sum_n (read @ Down)[a,n] Left[n,i] Right[n,j]``.

    Squared MLPs use the same expression with identical Left/Right factors.  The
    Frobenius score is invariant to an orthogonal change of readout coordinates.
    The normalized score removes the separate Frobenius scales of all four
    factors; it is a structural alignment score, not an activation-weighted one.
    """
    read = torch.as_tensor(read_map).float()
    if hasattr(mlp, "Left") and hasattr(mlp, "Right") and hasattr(mlp, "Down"):
        left = mlp.Left.weight.detach().float()
        right = mlp.Right.weight.detach().float()
        down = mlp.Down.weight.detach().float()
    elif hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
        left = mlp.c_fc.weight.detach().float()
        right = left
        down = mlp.c_proj.weight.detach().float()
    else:
        raise SubspaceWeightAtlasError("unsupported MLP weight factorization")
    if (read.ndim != 2 or not torch.isfinite(read).all() or read.shape[1] != down.shape[0]
            or left.ndim != 2 or right.ndim != 2 or down.ndim != 2
            or left.shape != right.shape or down.shape[1] != left.shape[0]):
        raise SubspaceWeightAtlasError("read map or MLP factors have incompatible shapes")
    output = read @ down
    tensor = torch.einsum("an,ni,nj->aij", output, left, right)
    score = float(torch.linalg.vector_norm(tensor))
    denominator = (float(torch.linalg.matrix_norm(read))
                   * float(torch.linalg.matrix_norm(down))
                   * float(torch.linalg.matrix_norm(left))
                   * float(torch.linalg.matrix_norm(right)))
    return {"output": output, "left": left, "right": right, "tensor": tensor,
            "score": score, "normalized_score": score / denominator if denominator > 0 else 0.0}


def activation_conditioned_mlp_write(mlp, read_map, base_input, donor_input):
    """Evaluate the exact read-contracted MLP donor-minus-base response.

    Static tensor norms measure which writes are possible.  This contraction
    evaluates which write is activated by a paired task distribution while
    retaining the exact weight factorization.  Inputs may have any leading
    dimensions followed by ``d_model``; the result has the same leading
    dimensions followed by the read rank.
    """
    read = torch.as_tensor(read_map).float()
    base, donor = torch.as_tensor(base_input).float(), torch.as_tensor(donor_input).float()
    if hasattr(mlp, "Left") and hasattr(mlp, "Right") and hasattr(mlp, "Down"):
        left = mlp.Left.weight.detach().float()
        right = mlp.Right.weight.detach().float()
        down = mlp.Down.weight.detach().float()
    elif hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
        left = mlp.c_fc.weight.detach().float()
        right = left
        down = mlp.c_proj.weight.detach().float()
    else:
        raise SubspaceWeightAtlasError("unsupported MLP weight factorization")
    if (base.shape != donor.shape or base.ndim < 1 or base.shape[-1] != left.shape[1]
            or left.shape != right.shape or down.shape[1] != left.shape[0]
            or read.ndim != 2 or read.shape[1] != down.shape[0]
            or not torch.isfinite(base).all() or not torch.isfinite(donor).all()
            or not torch.isfinite(read).all()):
        raise SubspaceWeightAtlasError("conditioned MLP inputs, factors, or read map are invalid")
    base_hidden = (base @ left.T) * (base @ right.T)
    donor_hidden = (donor @ left.T) * (donor @ right.T)
    response = (donor_hidden - base_hidden) @ (read @ down).T
    return {"response": response, "base_hidden": base_hidden,
            "donor_hidden": donor_hidden, "read_down": read @ down}


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
