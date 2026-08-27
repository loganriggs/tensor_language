"""Pure construction and pricing primitives for executable MLP0 Down programs.

The GPU collector supplies fit-row sufficient statistics.  This module turns them
into deterministic low-rank maps, canonicalizes the factor gauge, prices the exact
wire format, and constructs assignment-preserving centroid nulls.  It neither loads
the language model nor reads evaluation rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


D_MODEL = 1152
HIDDEN = 4608
VOCAB = 50257
COEFFICIENT_BYTES = 2  # all admitted constants are physically evaluated as bf16
PROGRAM_OVERHEAD_BYTES = 4096


def packed_assignment_bytes(vocab: int, occupied: int) -> int:
    if vocab <= 0 or occupied <= 0:
        raise ValueError("vocab and occupied state counts must be positive")
    bits = max(1, math.ceil(math.log2(occupied)))
    return math.ceil(vocab * bits / 8)


def program_price_bytes(
    rank: int,
    *,
    occupied: int = 0,
    vocab: int = VOCAB,
    d_model: int = D_MODEL,
    hidden: int = HIDDEN,
) -> dict[str, int]:
    """Exact registered raw-byte price for the fixed decoder wire format."""
    if rank <= 0 or min(vocab, d_model, hidden) <= 0 or occupied < 0:
        raise ValueError("invalid program dimensions")
    factor = COEFFICIENT_BYTES * rank * (d_model + hidden)
    # The additive gauge is fixed to one output intercept.  The fit compiler absorbs
    # -A B mu_h into it; mu_h is never present in the executable bundle.
    affine = COEFFICIENT_BYTES * d_model
    # Hierarchies reserve one assignment code for the zero-baseline unseen sentinel;
    # it has no serialized centroid vector.
    assignments = packed_assignment_bytes(vocab, occupied + 1) if occupied else 0
    centroids = COEFFICIENT_BYTES * occupied * d_model
    total = PROGRAM_OVERHEAD_BYTES + factor + affine + assignments + centroids
    return {
        "program_overhead": PROGRAM_OVERHEAD_BYTES,
        "factors": factor,
        "affine_constants": affine,
        "assignments": assignments,
        "centroids": centroids,
        "total": total,
    }


def matched_hierarchy_rank(
    continuous_rank: int,
    occupied: int,
    *,
    vocab: int = VOCAB,
    d_model: int = D_MODEL,
    hidden: int = HIDDEN,
) -> int:
    """Largest hierarchy rank no more expensive than the continuous rung."""
    ceiling = program_price_bytes(
        continuous_rank, vocab=vocab, d_model=d_model, hidden=hidden
    )["total"]
    low, high = 1, continuous_rank
    while low <= high:
        middle = (low + high) // 2
        price = program_price_bytes(
            middle, occupied=occupied, vocab=vocab, d_model=d_model, hidden=hidden
        )["total"]
        if price <= ceiling:
            low = middle + 1
        else:
            high = middle - 1
    if high < 1:
        raise RuntimeError("hierarchy overhead leaves no positive matched rank")
    return high


def canonical_balanced_factors(
    output_basis: torch.Tensor,
    coefficient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize ``output_basis @ output_basis.T @ coefficient.T``.

    ``output_basis`` is [D,r] orthonormal and ``coefficient`` is [H,D].  The
    returned balanced factors A[D,r], B[r,H] have the same product.  A deterministic
    maximum-loading sign rule removes ordinary SVD sign gauge.
    """
    if output_basis.ndim != 2 or coefficient.ndim != 2:
        raise ValueError("basis and coefficient must be matrices")
    d_model, rank = output_basis.shape
    if coefficient.shape[1] != d_model or rank <= 0:
        raise ValueError("incompatible factor dimensions")
    core = output_basis.T @ coefficient.T
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    root = singular.clamp_min(0).sqrt()
    left = (output_basis @ u) * root.unsqueeze(0)
    right = root.unsqueeze(1) * vh
    for column in range(rank):
        pivot = int(left[:, column].abs().argmax())
        if float(left[pivot, column]) < 0:
            left[:, column].neg_()
            right[column].neg_()
    return left, right


def fit_reduced_rank_from_statistics(
    covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
    rank: int,
    *,
    ridge_fraction: float = 1e-4,
) -> dict[str, torch.Tensor | float]:
    """Fit a reduced-rank ridge map from centered sufficient statistics.

    The full coefficient minimizes the fixed ridge objective.  Its predictions are
    then projected onto the leading output eigenvectors under the empirical input
    covariance, which is the registered data-conditioned rank ordering.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if cross_covariance.ndim != 2 or cross_covariance.shape[0] != covariance.shape[0]:
        raise ValueError("cross covariance has the wrong input dimension")
    if not (0 < rank <= cross_covariance.shape[1]) or ridge_fraction <= 0:
        raise ValueError("invalid rank or ridge fraction")
    covariance = covariance.float()
    cross_covariance = cross_covariance.float()
    scale = float(torch.trace(covariance) / covariance.shape[0])
    ridge = ridge_fraction * max(scale, torch.finfo(covariance.dtype).tiny)
    regularized = covariance + ridge * torch.eye(
        covariance.shape[0], dtype=covariance.dtype, device=covariance.device
    )
    coefficient = torch.linalg.solve(regularized, cross_covariance)
    prediction_covariance = coefficient.T @ covariance @ coefficient
    prediction_covariance = (prediction_covariance + prediction_covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(prediction_covariance)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    basis = eigenvectors[:, order]
    left, right = canonical_balanced_factors(basis, coefficient)
    return {
        "left": left,
        "right": right,
        "coefficient": coefficient,
        "output_basis": basis,
        "prediction_eigenvalues": eigenvalues[order],
        "ridge": ridge,
    }


def deterministic_centroid_derangement(
    centroids: torch.Tensor,
    masses: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Minimum-cost no-fixed-point centroid permutation in mass/norm space."""
    if centroids.ndim != 2 or masses.ndim != 1 or centroids.shape[0] != masses.shape[0]:
        raise ValueError("centroids and masses have incompatible shapes")
    if centroids.shape[0] < 2 or bool((masses <= 0).any()):
        raise ValueError("derangement needs at least two positive-mass states")
    from scipy.optimize import linear_sum_assignment

    features = torch.stack([masses.float().log(), centroids.float().norm(dim=1)], dim=1)
    mean = features.mean(0, keepdim=True)
    scale = features.std(0, unbiased=False, keepdim=True).clamp_min(1e-8)
    normalized = (features - mean) / scale
    cost = torch.cdist(normalized, normalized).double().cpu().numpy()
    np.fill_diagonal(cost, np.inf)
    source, target = linear_sum_assignment(cost)
    permutation = np.empty(len(source), dtype=np.int64)
    permutation[source] = target
    if np.any(permutation == np.arange(len(permutation))):
        raise RuntimeError("linear assignment returned a fixed point")
    permuted = centroids[torch.as_tensor(permutation, device=centroids.device)]
    mismatch = normalized - normalized[torch.as_tensor(permutation, device=normalized.device)]
    return permuted, {
        "permutation": permutation.tolist(),
        "fixed_points": 0,
        "mean_standardized_mass_norm_mismatch": float(mismatch.norm(dim=1).mean()),
        "max_standardized_mass_norm_mismatch": float(mismatch.norm(dim=1).max()),
        "exact_mass_norm_matching": bool(torch.equal(features, features[permutation])),
    }


def common_exact_product_price(
    *, d_model: int = D_MODEL, hidden: int = HIDDEN
) -> dict[str, int]:
    coefficients = 2 * d_model * hidden
    return {
        "left_right_coefficients": coefficients,
        "left_right_checkpoint_float32_bytes": coefficients * 4,
        "native_products_per_position": hidden,
    }


def pack_assignments(assignments: torch.Tensor, n_states: int) -> bytes:
    """Pack fixed-width unsigned assignments without pickle/container overhead."""
    values = assignments.detach().cpu().to(torch.int64).tolist()
    if n_states <= 0 or any(value < 0 or value >= n_states for value in values):
        raise ValueError("assignment outside declared state alphabet")
    width = max(1, math.ceil(math.log2(n_states)))
    output = bytearray()
    accumulator = 0
    available = 0
    for value in values:
        accumulator |= int(value) << available
        available += width
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_assignments(payload: bytes, count: int, n_states: int) -> torch.Tensor:
    if count < 0 or n_states <= 0:
        raise ValueError("invalid assignment shape")
    width = max(1, math.ceil(math.log2(n_states)))
    values = []
    accumulator = 0
    available = 0
    iterator = iter(payload)
    for _ in range(count):
        while available < width:
            try:
                accumulator |= next(iterator) << available
            except StopIteration as error:
                raise ValueError("truncated packed assignments") from error
            available += 8
        values.append(accumulator & ((1 << width) - 1))
        accumulator >>= width
        available -= width
    result = torch.tensor(values, dtype=torch.long)
    if bool((result >= n_states).any()):
        raise ValueError("packed assignment uses undeclared state")
    return result


def serialize_program(path: Path, program: dict[str, Any]) -> dict[str, Any]:
    """Write the registered fixed-layout program and return its physical receipt."""
    required = {"rank", "intercept", "left", "right", "centroids", "assignments"}
    if not required.issubset(program):
        raise ValueError(f"missing program fields: {sorted(required - set(program))}")
    rank = int(program["rank"])
    intercept = program["intercept"].detach().cpu().to(torch.bfloat16).contiguous()
    left = program["left"].detach().cpu().to(torch.bfloat16).contiguous()
    right = program["right"].detach().cpu().to(torch.bfloat16).contiguous()
    centroids = program["centroids"].detach().cpu().to(torch.bfloat16).contiguous()
    assignments = program["assignments"].detach().cpu().to(torch.long).contiguous()
    if (tuple(intercept.shape) != (left.shape[0],)
            or left.shape[1] != rank or right.shape[0] != rank
            or centroids.ndim != 2 or centroids.shape[1] != left.shape[0]):
        raise ValueError("program tensor shapes violate the fixed decoder contract")
    n_centroids = int(centroids.shape[0])
    n_states = n_centroids + 1 if n_centroids else 0
    if n_centroids == 0:
        if assignments.numel() != 0:
            raise ValueError("continuous program cannot carry assignments")
        packed = b""
    else:
        packed = pack_assignments(assignments, n_states)
    header = {
        "schema_version": 1,
        "format": "mlp0_native_down_bf16_le_v1",
        "rank": rank,
        "d_model": int(left.shape[0]),
        "hidden": int(right.shape[1]),
        "vocab": int(assignments.numel()),
        "n_centroids": n_centroids,
        "n_states": n_states,
        "assignment_bytes": len(packed),
        "additive_gauge": "single output intercept; -AB*mu_h absorbed",
        "tensor_order": ["intercept", "left", "right", "centroids", "assignments"],
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded_header) + 1 > PROGRAM_OVERHEAD_BYTES:
        raise RuntimeError("program header exceeds registered fixed overhead")
    body = b"".join([
        intercept.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(),
        left.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(),
        right.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(),
        centroids.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(),
        packed,
    ])
    payload = encoded_header + b"\n" + bytes(PROGRAM_OVERHEAD_BYTES - len(encoded_header) - 1) + body
    expected = program_price_bytes(
        rank, occupied=n_centroids, vocab=int(assignments.numel()),
        d_model=int(left.shape[0]), hidden=int(right.shape[1])
    )["total"]
    if len(payload) != expected:
        raise RuntimeError(f"physical/analytic price mismatch: {len(payload)} != {expected}")
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "header": header,
    }


def load_program(path: Path) -> dict[str, torch.Tensor | int | dict[str, Any]]:
    payload = path.read_bytes()
    raw_header = payload[:PROGRAM_OVERHEAD_BYTES].split(b"\n", 1)[0]
    header = json.loads(raw_header)
    rank, d_model, hidden = (int(header[key]) for key in ("rank", "d_model", "hidden"))
    vocab, n_states = (int(header[key]) for key in ("vocab", "n_states"))
    n_centroids = int(header["n_centroids"])
    offset = PROGRAM_OVERHEAD_BYTES

    def take_bf16(count: int, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal offset
        size = count * COEFFICIENT_BYTES
        array = np.frombuffer(payload[offset:offset + size], dtype="<u2").copy()
        offset += size
        return torch.from_numpy(array).view(torch.bfloat16).reshape(shape)

    intercept = take_bf16(d_model, (d_model,))
    left = take_bf16(d_model * rank, (d_model, rank))
    right = take_bf16(rank * hidden, (rank, hidden))
    centroids = take_bf16(n_centroids * d_model, (n_centroids, d_model))
    assignment_bytes = int(header["assignment_bytes"])
    assignments = (
        unpack_assignments(payload[offset:offset + assignment_bytes], vocab, n_states)
        if n_states else torch.empty(0, dtype=torch.long)
    )
    offset += assignment_bytes
    if offset != len(payload):
        raise ValueError("program has trailing or truncated bytes")
    return {
        "rank": rank, "intercept": intercept,
        "left": left, "right": right, "centroids": centroids,
        "assignments": assignments, "header": header,
    }
