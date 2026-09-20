"""Weights-first symmetric Tucker primitives; no model or activation fitting.

T[v,i,j] = sum_k C[v,k] sym(A[k,i] B[k,j]). Only small cores and
gate-space Gram blocks are built. Norms concern the numerator in the declared
input coordinates, not normalized-model behavior or OOD fidelity.
"""
from __future__ import annotations

import math
import torch


def mode_grams(C, A, B):
    """Exact Grams of the symmetric tensor; use reduced ambient frames first.

    Torch counterpart of mlp2_implicit_folded_tensor.implicit_folded_mode_grams.
    Unlike tensor_inner this needs O(gates**2) workspace.
    """
    aa, bb, ab, cc = A @ A.T, B @ B.T, A @ B.T, C.T @ C
    output = C @ ((aa * bb + ab * ab.T) / 2) @ C.T
    cross = A.T @ (cc * ab.T) @ B
    inputs = (A.T @ (cc * bb) @ A + B.T @ (cc * aa) @ B
              + cross + cross.T) / 4
    return (output + output.T) / 2, (inputs + inputs.T) / 2


def project_core(C, A, B, W, P):
    """Contract all three tensor modes, preserving shared input symmetry."""
    left, right = A @ P, B @ P
    core = torch.einsum('ak,kp,kq->apq', W.T @ C, left, right)
    return (core + core.transpose(-1, -2)) / 2


def tensor_inner(C, A, B, D, L, R, block_size=128):
    """Exact symmetric tensor inner product, with bounded gate-Gram workspace.

    Use float64 when subtracting nearly equal energies. No V*V or V*d*d
    object is formed; gradients are supported for optimization if required.
    """
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    value = C.new_zeros(())
    for start in range(0, A.shape[0], block_size):
        stop = start + block_size
        aa, bb = A[start:stop], B[start:stop]
        gram = ((aa @ L.T) * (bb @ R.T)
                + (aa @ R.T) * (bb @ L.T)) / 2
        value = value + ((C[:, start:stop].T @ D) * gram).sum()
    return value


def require_orthonormal(factor):
    eye = torch.eye(factor.shape[1], dtype=factor.dtype, device=factor.device)
    if not torch.allclose(factor.T @ factor, eye, atol=2e-5, rtol=2e-5):
        raise ValueError('orthonormal columns required for energy accounting')


def projection_error_squared(total_energy, projected_core, retained_core):
    """Pythagorean error for orthonormal W/P used to obtain projected_core.

    Caller must check those frames with require_orthonormal. Retaining this
    split avoids subtracting target and reconstruction tensors explicitly.
    """
    outside = total_energy - projected_core.square().sum()
    tolerance = 1e-7 * max(float(total_energy.abs()), 1.0)
    if float(outside) < -tolerance:
        raise ValueError('negative projection residual: check frames/precision')
    return outside.clamp_min(0) + (projected_core - retained_core).square().sum()


def sparse_core(core, interactions):
    """Best m-entry core for fixed orthonormal frames, in tensor Frobenius norm.

    Count each unordered (p,q) once. Off-diagonal entries carry twice the
    energy of diagonal entries. This is exact hard sparsity, not near-zero
    fraction. Changing the orthogonal frames changes the optimum support.
    """
    if core.ndim != 3 or core.shape[1] != core.shape[2]:
        raise ValueError('expected [output_rank,input_rank,input_rank]')
    if not torch.allclose(core, core.transpose(-1, -2)):
        raise ValueError('core must be symmetric')
    p, q = torch.triu_indices(core.shape[1], core.shape[2], device=core.device)
    values = core[:, p, q]
    if not 0 <= interactions <= values.numel():
        raise ValueError('interaction budget out of range')
    energy = values.square() * torch.where(p == q, 1, 2)
    mask = torch.zeros_like(values, dtype=torch.bool).flatten()
    mask[energy.flatten().topk(interactions).indices] = True
    kept = values * mask.reshape_as(values)
    result = torch.zeros_like(core)
    result[:, p, q] = kept
    result[:, q, p] = kept
    return result


def execute(z, W, P, core, *, h=None, rms_eps=0.0):
    """Execute numerator, optionally divide by mean(h**2)+native RMS epsilon.

    h must be the actual pre-normalization state Ez, including all residual
    sources once. Affine norm weights must already be folded into A and B.
    This is a layer contribution, excluding final norm/softcap/residual readout.
    """
    s = z @ P
    prediction = torch.einsum('np,apq,nq->na', s, core, s) @ W.T
    if h is not None:
        if rms_eps < 0:
            raise ValueError('rms_eps must be nonnegative')
        denominator = h.square().mean(-1, keepdim=True) + rms_eps
        if not bool((denominator > 0).all()):
            raise ValueError('RMS denominator must be positive')
        prediction = prediction / denominator
    return prediction


def executable_price(W, P, core):
    """Price an explicit dense-factor / sparse-core executor (not this einsum).

    Floating values and support metadata are separate. Unique s_p*s_q
    products can be shared across output features. No quantization assumed.
    The dense factor cost is charged even if core support makes columns dead.
    """
    p, q = torch.triu_indices(core.shape[1], core.shape[2], device=core.device)
    support = core[:, p, q] != 0
    nnz = int(support.sum())
    return dict(factor_values=W.numel() + P.numel(), core_values=nnz,
                stored_values=W.numel() + P.numel() + nnz,
                support_index_bits=nnz * math.ceil(math.log2(max(2, support.numel()))),
                unique_pair_products=int(support.any(dim=0).sum()),
                core_weight_multiplies=nnz,
                dense_factor_multiplies=W.numel() + P.numel())
