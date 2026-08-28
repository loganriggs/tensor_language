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


def tensor_inner_product(A1, B1, C1, A2, B2, C2):
    """Frobenius inner product between two factored symmetric-input tensors."""
    A1, B1, C1 = _validate(A1, B1, C1)
    A2, B2, C2 = _validate(A2, B2, C2)
    if A1.shape[0] != A2.shape[0] or C1.shape[1] != C2.shape[1]:
        raise ValueError("tensor input/output dimensions differ")
    cross = .5*((A1.T@A2)*(B1.T@B2) + (A1.T@B2)*(B1.T@A2))
    return float(torch.sum(cross*(C1@C2.T)))


def relative_tensor_frobenius_error(A1, B1, C1, A2, B2, C2):
    """Exact relative coefficient-tensor error between two factor programs."""
    norm1 = tensor_frobenius_sq(A1, B1, C1)
    norm2 = tensor_frobenius_sq(A2, B2, C2)
    if norm1 == 0:
        raise ValueError("reference tensor is zero")
    error_sq = norm1+norm2-2*tensor_inner_product(A1, B1, C1, A2, B2, C2)
    return (max(0.0, error_sq)/norm1)**.5


def tensor_frobenius_error(A1, B1, C1, A2, B2, C2):
    """Absolute Frobenius norm of the coefficient-tensor difference."""
    norm1 = tensor_frobenius_sq(A1, B1, C1)
    norm2 = tensor_frobenius_sq(A2, B2, C2)
    error_sq = norm1+norm2-2*tensor_inner_product(A1, B1, C1, A2, B2, C2)
    return max(0.0, error_sq)**.5


def execute_quadratic(A, B, C, z):
    A, B, C = _validate(A, B, C)
    z = torch.as_tensor(z, dtype=torch.float64)
    if z.shape[-1] != A.shape[0] or not torch.isfinite(z).all():
        raise ValueError("input has wrong width or nonfinite values")
    return ((z@A)*(z@B))@C


def quadratic_jvp(A, B, C, z, direction):
    """Exact Jacobian-vector product of the homogeneous quadratic map."""
    A, B, C = _validate(A, B, C)
    z = torch.as_tensor(z, dtype=torch.float64)
    direction = torch.as_tensor(direction, dtype=torch.float64)
    if z.shape != direction.shape or z.shape[-1] != A.shape[0] \
            or not torch.isfinite(z).all() or not torch.isfinite(direction).all():
        raise ValueError("state and direction must be aligned finite tensors")
    return (((direction@A)*(z@B)+(z@A)*(direction@B))@C)


def quadratic_jacobian(A, B, C, z):
    """Exact Jacobian, with shape ``[..., din, dout]`` for row-vector JVPs."""
    A, B, C = _validate(A, B, C)
    z = torch.as_tensor(z, dtype=torch.float64)
    if z.shape[-1] != A.shape[0] or not torch.isfinite(z).all():
        raise ValueError("input has wrong width or nonfinite values")
    return (torch.einsum("...j,ij,jo->...io", z@B, A, C)
            + torch.einsum("...j,ij,jo->...io", z@A, B, C))


def residual_output_unfolding_spectral_norm(A1, B1, C1, A2, B2, C2):
    """Spectral norm of the residual tensor's output-mode unfolding."""
    A1, B1, C1 = _validate(A1, B1, C1)
    A2, B2, C2 = _validate(A2, B2, C2)
    if A1.shape[0] != A2.shape[0] or C1.shape[1] != C2.shape[1]:
        raise ValueError("tensor input/output dimensions differ")
    gram = output_unfolding_gram(
        torch.cat((A1, A2), dim=1), torch.cat((B1, B2), dim=1),
        torch.cat((C1, -C2), dim=0))
    return float(torch.linalg.eigvalsh(gram)[-1].clamp_min(0).sqrt())


def midpoint_residual_lipschitz_bound(A1, B1, C1, A2, B2, C2, z1, z2):
    """Tight state-pair bound using the exact residual Jacobian at the midpoint.

    A homogeneous quadratic obeys ``e(z2)-e(z1)=J_e((z1+z2)/2)(z2-z1)``.
    The returned induced 2-norm therefore bounds this particular secant exactly.
    """
    z1 = torch.as_tensor(z1, dtype=torch.float64)
    z2 = torch.as_tensor(z2, dtype=torch.float64)
    if z1.shape != z2.shape:
        raise ValueError("states must have identical shapes")
    midpoint = (z1+z2)/2
    jacobian = (quadratic_jacobian(A1, B1, C1, midpoint)
                - quadratic_jacobian(A2, B2, C2, midpoint))
    return torch.linalg.matrix_norm(jacobian, ord=2)


def residual_secant_diagnostics(A1, B1, C1, A2, B2, C2, z_live, z_composed):
    """Diagnose how a replacement residual changes under an upstream state shift.

    Returns row-level observed drift plus exact-midpoint, local-operator, and global
    output-unfolding certificates. No model outputs or labels are required.
    """
    z_live = torch.as_tensor(z_live, dtype=torch.float64)
    z_composed = torch.as_tensor(z_composed, dtype=torch.float64)
    if z_live.shape != z_composed.shape or z_live.ndim < 1:
        raise ValueError("live and composed states must have identical nonempty shapes")
    shift = z_composed-z_live
    midpoint = (z_live+z_composed)/2
    residual_live = (execute_quadratic(A1, B1, C1, z_live)
                     - execute_quadratic(A2, B2, C2, z_live))
    residual_composed = (execute_quadratic(A1, B1, C1, z_composed)
                         - execute_quadratic(A2, B2, C2, z_composed))
    observed = residual_composed-residual_live
    reconstructed = (quadratic_jvp(A1, B1, C1, midpoint, shift)
                     - quadratic_jvp(A2, B2, C2, midpoint, shift))
    shift_norm = torch.linalg.vector_norm(shift, dim=-1)
    observed_norm = torch.linalg.vector_norm(observed, dim=-1)
    reconstruction_error = torch.linalg.vector_norm(
        observed-reconstructed, dim=-1)
    local_coefficient = midpoint_residual_lipschitz_bound(
        A1, B1, C1, A2, B2, C2, z_live, z_composed)
    unfolding_norm = residual_output_unfolding_spectral_norm(
        A1, B1, C1, A2, B2, C2)
    global_coefficient = unfolding_norm*(
        torch.linalg.vector_norm(z_live, dim=-1)
        + torch.linalg.vector_norm(z_composed, dim=-1))
    return {
        "input_shift_norm": shift_norm,
        "observed_residual_drift_norm": observed_norm,
        "midpoint_reconstruction_error": reconstruction_error,
        "local_operator_coefficient": local_coefficient,
        "local_upper_bound": local_coefficient*shift_norm,
        "global_unfolding_coefficient": global_coefficient,
        "global_upper_bound": global_coefficient*shift_norm,
    }


def rms_sphere_residual_lipschitz_bound(A1, B1, C1, A2, B2, C2):
    """Global residual-map Lipschitz bound when both states have RMS norm one.

    ``e(z')-e(z)=DeltaT(z'+z,z'-z)`` and ``||z||=sqrt(din)`` imply
    ``||e(z')-e(z)|| <= 2 sqrt(din) ||DeltaT||_F ||z'-z||``.
    """
    din = A1.shape[0]
    if A2.shape[0] != din:
        raise ValueError("input widths differ")
    return 2*din**.5*tensor_frobenius_error(A1, B1, C1, A2, B2, C2)


def rms_sphere_residual_spectral_bound(A1, B1, C1, A2, B2, C2):
    """Sharper global RMS-sphere bound using the output unfolding operator norm."""
    din = A1.shape[0]
    if A2.shape[0] != din:
        raise ValueError("input widths differ")
    return 2*din**.5*residual_output_unfolding_spectral_norm(
        A1, B1, C1, A2, B2, C2)


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
