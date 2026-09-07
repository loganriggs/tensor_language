"""Differentiable, fixed-dose projectors for absolute native head-response clamps."""

# BQGATE: LIBRARY
from __future__ import annotations


class HeadResponseProjectorError(RuntimeError):
    pass


def orthonormal_basis(torch, raw):
    """Map a finite ``[head_dim, rank]`` parameter to an orthonormal basis."""
    if getattr(raw, "ndim", None) != 2 or not 0 < raw.shape[1] <= raw.shape[0]:
        raise HeadResponseProjectorError("projector parameter shape is invalid")
    if not bool(torch.isfinite(raw).all()):
        raise HeadResponseProjectorError("projector parameter is non-finite")
    basis, _ = torch.linalg.qr(raw.float(), mode="reduced")
    return basis


def project_last_dimension(torch, response, basis):
    """Apply ``UU^T`` on the last response dimension without a learned dose."""
    if (getattr(response, "ndim", None) is None or response.ndim < 1
            or getattr(basis, "ndim", None) != 2
            or response.shape[-1] != basis.shape[0]):
        raise HeadResponseProjectorError("response and basis dimensions do not match")
    return torch.einsum("...d,dk,ek->...e", response.float(), basis, basis)


def absolute_projected_head_response(torch, base, donor, raw_bases_by_head):
    """Return ``base + (donor-base) U_h U_h^T`` at exactly the named heads.

    ``base`` and ``donor`` are native ``[batch, sequence, head, head_dim]`` tensors.
    Unnamed heads remain exactly at the base response. Each value in
    ``raw_bases_by_head`` is orthonormalized independently. There is deliberately no
    amplitude parameter: the intervention dose is fixed at one.
    """
    if (getattr(base, "ndim", None) != 4 or base.shape != donor.shape
            or not raw_bases_by_head):
        raise HeadResponseProjectorError("native head response shape or projector map is invalid")
    head_count, head_dim = int(base.shape[2]), int(base.shape[3])
    heads = tuple(int(head) for head in raw_bases_by_head)
    if len(heads) != len(set(heads)) or any(not 0 <= head < head_count for head in heads):
        raise HeadResponseProjectorError("projector head selection is invalid")
    changed = base.float().clone()
    for head in heads:
        basis = orthonormal_basis(torch, raw_bases_by_head[head])
        if basis.shape[0] != head_dim:
            raise HeadResponseProjectorError("projector width differs from native head width")
        delta = donor[..., head, :].float() - base[..., head, :].float()
        changed[..., head, :] = base[..., head, :].float() + project_last_dimension(
            torch, delta, basis
        )
    return changed


def projector_frobenius_distance(torch, left, right):
    """Gauge-invariant distance ``||UU^T - VV^T||_F`` between equal-rank bases."""
    if getattr(left, "ndim", None) != 2 or left.shape != right.shape:
        raise HeadResponseProjectorError("fold bases must have equal two-dimensional shapes")
    return torch.linalg.matrix_norm(left @ left.T - right @ right.T, ord="fro")


def principal_angle_cosines(torch, left, right):
    """Gauge-invariant singular values of ``U^T V`` for fold-stability reports."""
    if getattr(left, "ndim", None) != 2 or left.shape != right.shape:
        raise HeadResponseProjectorError("fold bases must have equal two-dimensional shapes")
    return torch.linalg.svdvals(left.T @ right)
