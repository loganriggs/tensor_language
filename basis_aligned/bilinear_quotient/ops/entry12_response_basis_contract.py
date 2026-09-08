"""Cross-fitted analytic target/control bases for complete residual-state deltas."""

from __future__ import annotations


class ResponseBasisError(ValueError):
    pass


def orthonormal_columns(torch, columns, *, relative_threshold=1e-6):
    if columns.ndim != 2 or not 0 < relative_threshold < 1:
        raise ResponseBasisError("columns must be a matrix and threshold must be in (0,1)")
    if columns.shape[1] == 0:
        return columns.clone(), []
    u, singular, _vh = torch.linalg.svd(columns.float(), full_matrices=False)
    if not len(singular) or float(singular[0]) == 0:
        return columns[:, :0].float(), [float(value) for value in singular]
    rank = int((singular > singular[0] * relative_threshold).sum())
    return u[:, :rank].contiguous(), [float(value) for value in singular]


def fit_bases(torch, target_a1, target_a2, p_rows, *, relative_threshold=1e-6):
    vectors = [target_a1.float().reshape(-1), target_a2.float().reshape(-1)]
    if vectors[0].shape != vectors[1].shape or p_rows.ndim != 2 or p_rows.shape[1] != len(vectors[0]):
        raise ResponseBasisError("target vectors and P rows have incompatible widths")
    if float(vectors[0].norm()) == 0 or float(vectors[1].norm()) == 0:
        raise ResponseBasisError("target means must be nonzero")
    if float(vectors[0] @ vectors[1]) < 0:
        vectors[1] = -vectors[1]
    shared, shared_singular = orthonormal_columns(
        torch, (vectors[0] + vectors[1])[:, None], relative_threshold=relative_threshold)
    union, union_singular = orthonormal_columns(
        torch, torch.stack(vectors, dim=1), relative_threshold=relative_threshold)
    p_basis, p_singular = orthonormal_columns(
        torch, p_rows.float().T, relative_threshold=relative_threshold)
    target_columns = torch.stack(vectors, dim=1)
    complemented = target_columns - p_basis @ (p_basis.T @ target_columns)
    complement, complement_singular = orthonormal_columns(
        torch, complemented, relative_threshold=relative_threshold)
    return {
        "shared_dim_rank1": shared,
        "construction_union_rank_le_2": union,
        "p_complement_union_rank_le_2": complement,
    }, {
        "shared_singular_values": shared_singular,
        "union_singular_values": union_singular,
        "p_singular_values": p_singular,
        "p_rank": int(p_basis.shape[1]),
        "complement_singular_values": complement_singular,
    }


def projected_absolute(torch, off, on, basis, semantic_positions):
    if off.shape != on.shape or off.ndim != 3 or basis.ndim != 2 or basis.shape[0] != off.shape[-1]:
        raise ResponseBasisError("state tensors or basis have incompatible shapes")
    delta = on.float() - off.float()
    projected = (delta @ basis.float()) @ basis.float().T
    result = off.clone()
    for row, stop in enumerate(semantic_positions):
        stop = int(stop)
        if not 0 <= stop < off.shape[1]:
            raise ResponseBasisError("semantic position out of range")
        result[row, :stop + 1] = (off[row, :stop + 1].float()
                                  + projected[row, :stop + 1]).to(result)
    return result
