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
import zlib

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


def _canonical_svd(value, rank, degeneracy_tol):
    left, singular, right = torch.linalg.svd(value, full_matrices=False)
    if not 0 < rank <= singular.numel():
        raise ValueError("rank outside matrix dimensions")
    scale = float(singular[0].abs().clamp_min(torch.finfo(singular.dtype).tiny))
    relevant = singular[:rank]
    gaps = (relevant[:-1]-relevant[1:]).abs()/scale
    if gaps.numel() and bool(torch.any(gaps <= degeneracy_tol)):
        raise DegenerateAttentionPlane("repeated retained singular-value stratum")
    if rank < singular.numel():
        boundary_gap = float((singular[rank-1]-singular[rank]).abs()/scale)
        if boundary_gap <= degeneracy_tol:
            raise DegenerateAttentionPlane("ambiguous truncation boundary")
    left = left[:, :rank].clone()
    singular = singular[:rank].clone()
    right = right[:rank].clone()
    for component in range(rank):
        pivot = int(torch.argmax(right[component].abs()))
        if float(right[component, pivot]) < 0:
            left[:, component].neg_()
            right[component].neg_()
    return left, singular, right


def _lowrank_candidate_bytes(maps, rank, quantization_step, degeneracy_tol,
                             semantic_id):
    first = _canonicalize_branch(maps["q"], maps["k"], degeneracy_tol)
    second = _canonicalize_branch(maps["q2"], maps["k2"], degeneracy_tol)
    factors = [_canonical_svd(value, rank, degeneracy_tol)
               for value in first+second]
    header = json.dumps({
        "codec": "product_attention_head_lowrank_quotient_v1",
        "semantic_id": semantic_id,
        "shape": list((first+second)[0].shape),
        "rank": rank,
        "quantization_step": quantization_step,
        "order": list(NAMES),
    }, sort_keys=True, separators=(",", ":")).encode()
    payload = []
    for triple in factors:
        for value in triple:
            quantized = torch.round(value/quantization_step).to(torch.int32).numpy()
            payload.append(np.asarray(quantized, dtype="<i4", order="C").tobytes())
    compressed = zlib.compress(b"".join(payload), level=9)
    return struct.pack("<I", len(header))+header+compressed


def canonical_lowrank_head_bytes(maps, rank, quantization_step, *,
                                 degeneracy_tol=1e-10,
                                 semantic_id="bilin18.product_attention.head"):
    """Canonical rank-aware bytes over the exact sign/swap/centralizer orbit."""
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
            # Select the finite orbit using canonical dense semantics first. This
            # avoids 16 redundant large SVDs per head while retaining invariance.
            key = _candidate_bytes(candidate, quantization_step,
                                   degeneracy_tol, semantic_id)
            candidates.append((key, candidate))
    _, selected = min(candidates, key=lambda item: item[0])
    return _lowrank_candidate_bytes(selected, rank, quantization_step,
                                    degeneracy_tol, semantic_id)


def decode_lowrank_head(encoded):
    """Decode canonical low-rank bytes to its four canonical dense representatives."""
    header_length = struct.unpack("<I", encoded[:4])[0]
    header = json.loads(encoded[4:4+header_length])
    if header["codec"] != "product_attention_head_lowrank_quotient_v1":
        raise ValueError("wrong codec")
    rows, columns = header["shape"]
    rank = header["rank"]
    step = header["quantization_step"]
    raw = zlib.decompress(encoded[4+header_length:])
    offset = 0

    def take(count, shape):
        nonlocal offset
        size = 4*count
        values = np.frombuffer(raw[offset:offset+size], dtype="<i4").copy()
        if values.size != count:
            raise ValueError("truncated payload")
        offset += size
        return torch.from_numpy(values.reshape(shape)).to(torch.float64)*step

    decoded = {}
    for name in NAMES:
        left = take(rows*rank, (rows, rank))
        singular = take(rank, (rank,))
        right = take(rank*columns, (rank, columns))
        decoded[name] = (left*singular) @ right
    if offset != len(raw):
        raise ValueError("trailing payload")
    return header, decoded
