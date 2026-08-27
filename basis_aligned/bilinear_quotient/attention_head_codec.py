"""Canonical quantized bytes for one product-attention head.

Exact quotient implemented here:
  * independent joint Q/K SO(2) rotations in every RoPE frequency plane;
  * sign flips of Q,K,Q2,K2 with even total parity;
  * exchange of the two multiplicative QK branches.

Projection magnitude is deliberately retained: RMSNorm epsilon makes scale only an
approximate symmetry.  This codec is a CPU primitive; pricing full frozen maps is a
separate step after its adoption tests pass.
"""

from __future__ import annotations

import itertools
import json
import struct

import numpy as np
import torch


class DegenerateAttentionPlane(ValueError):
    pass


NAMES = ("q", "k", "q2", "k2")


def _validate(maps):
    if set(maps) != set(NAMES):
        raise ValueError(f"expected maps {NAMES}")
    shapes = {tuple(maps[name].shape) for name in NAMES}
    if len(shapes) != 1:
        raise ValueError("all four maps must have one common shape")
    rows, _ = next(iter(shapes))
    if rows % 2:
        raise ValueError("RoPE head dimension must be even")
    if not all(torch.isfinite(maps[name]).all() for name in NAMES):
        raise ValueError("maps must be finite")


def _canonicalize_branch(query, key, degeneracy_tol):
    """Gauge-fix joint SO(2) rotations plane by plane using a stable pivot."""
    query = query.detach().to(dtype=torch.float64, device="cpu").clone()
    key = key.detach().to(dtype=torch.float64, device="cpu").clone()
    half = query.shape[0]//2
    for plane in range(half):
        rows = torch.stack((
            torch.cat((query[plane], key[plane])),
            torch.cat((query[plane+half], key[plane+half])),
        ))
        norms = torch.linalg.vector_norm(rows, dim=0)
        viable = torch.nonzero(norms > degeneracy_tol, as_tuple=False).flatten()
        if viable.numel() == 0:
            if float(torch.linalg.vector_norm(rows)) == 0.0:
                continue
            raise DegenerateAttentionPlane(
                f"plane {plane} has no stable canonical pivot")
        pivot_index = int(viable[0])
        # A pivot close to the threshold makes a byte-level gauge choice unstable.
        if float(norms[pivot_index]) < 2*degeneracy_tol:
            raise DegenerateAttentionPlane(
                f"plane {plane} pivot lies in the ambiguity band")
        a, b = rows[:, pivot_index]
        radius = torch.sqrt(a*a+b*b)
        rotation = torch.stack((torch.stack((a/radius, b/radius)),
                                torch.stack((-b/radius, a/radius))))
        query[[plane, plane+half]] = rotation @ query[[plane, plane+half]]
        key[[plane, plane+half]] = rotation @ key[[plane, plane+half]]
    return query, key


def _candidate_bytes(maps, quantization_step, degeneracy_tol, semantic_id):
    first = _canonicalize_branch(maps["q"], maps["k"], degeneracy_tol)
    second = _canonicalize_branch(maps["q2"], maps["k2"], degeneracy_tol)
    ordered = first+second
    arrays = []
    for value in ordered:
        quantized = torch.round(value/quantization_step).to(torch.int32).numpy()
        arrays.append(np.asarray(quantized, dtype="<i4", order="C").tobytes())
    header = json.dumps({
        "codec": "product_attention_head_exact_quotient_v1",
        "semantic_id": semantic_id,
        "shape": list(ordered[0].shape),
        "quantization_step": quantization_step,
        "order": list(NAMES),
    }, sort_keys=True, separators=(",", ":")).encode()
    return struct.pack("<I", len(header))+header+b"".join(arrays)


def canonical_head_bytes(maps, quantization_step, *, degeneracy_tol=1e-10,
                         semantic_id="bilin18.product_attention.head"):
    """Return the lexicographically minimal bytes over the exact finite gauges."""
    _validate(maps)
    if not quantization_step > 0:
        raise ValueError("quantization_step must be positive")
    candidates = []
    for signs in itertools.product((-1, 1), repeat=4):
        if np.prod(signs) != 1:
            continue
        signed = {name: signs[index]*maps[name]
                  for index, name in enumerate(NAMES)}
        for swapped in (False, True):
            candidate = signed if not swapped else {
                "q": signed["q2"], "k": signed["k2"],
                "q2": signed["q"], "k2": signed["k"],
            }
            candidates.append(_candidate_bytes(candidate, quantization_step,
                                               degeneracy_tol, semantic_id))
    return min(candidates)
