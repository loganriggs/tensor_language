#!/usr/bin/env python3
"""Canonical byte codec for generic-stratum affine SVD replacements.

The affine semantics are ``y = x @ W + b``. Centering parameters are folded into
``b`` before pricing, and SVD column signs are fixed. Near-degenerate singular
spectra are rejected because their within-eigenspace rotation gauge needs a
different canonicalizer.
"""

from __future__ import annotations

import hashlib
import struct
import zlib

import numpy as np
import torch

MAGIC = b"BAFFINE1"
FORMAT_VERSION = 1
DEFAULT_SEMANTICS = "affine|inputs=ordered|output=vector"


def _validate_spectrum(s, rank, degeneracy_tolerance):
    if rank < 1 or rank > s.numel():
        raise ValueError("rank outside stored SVD")
    if not torch.isfinite(s).all() or not bool((s > 0).all()):
        raise ValueError("singular values must be finite and positive")
    if not bool((s[:-1] >= s[1:]).all()):
        raise ValueError("singular values must be descending")
    check_to = min(rank, s.numel() - 1)
    if check_to:
        gaps = (s[:check_to] - s[1:check_to + 1]).abs()
        scale = s[:check_to].abs().clamp_min(torch.finfo(s.dtype).tiny)
        if bool((gaps / scale <= degeneracy_tolerance).any()):
            raise ValueError("near-degenerate singular stratum is not canonicalized")


def canonical_affine_bytes(U, S, Vh, xm, ym, rank, step,
                           semantic_id=DEFAULT_SEMANTICS,
                           degeneracy_tolerance=1e-7):
    if not step > 0:
        raise ValueError("quantization step must be positive")
    U = U.detach().double().cpu()[:, :rank].clone()
    S = S.detach().double().cpu()
    Vh = Vh.detach().double().cpu()[:rank].clone()
    xm = xm.detach().double().cpu()
    ym = ym.detach().double().cpu()
    _validate_spectrum(S, rank, degeneracy_tolerance)
    if U.shape[1] != rank or Vh.shape[0] != rank or U.shape[0] != xm.numel() \
            or Vh.shape[1] != ym.numel():
        raise ValueError("incompatible affine SVD shapes")

    for column in range(rank):
        pivot = int(U[:, column].abs().argmax())
        if U[pivot, column] < 0:
            U[:, column] *= -1
            Vh[column] *= -1
    bias = ym - ((xm @ U) * S[:rank]) @ Vh
    arrays = (U, S[:rank], Vh, bias)
    quantized = []
    for value in arrays:
        q64 = torch.round(value / step).to(torch.int64)
        limit = torch.iinfo(torch.int32)
        if int(q64.min()) < limit.min or int(q64.max()) > limit.max:
            raise OverflowError("quantized coefficient exceeds int32")
        quantized.append(q64.to(torch.int32).contiguous().numpy())
    raw_parameters = b"".join(array.astype("<i4", copy=False).tobytes(order="C")
                              for array in quantized)
    parameter_bytes = zlib.compress(raw_parameters, level=9)
    semantic = semantic_id.encode("utf-8")
    graph = struct.pack(">8sBIIIdHQ", MAGIC, FORMAT_VERSION, U.shape[0],
                        Vh.shape[1], rank, float(step), len(semantic),
                        len(parameter_bytes)) + semantic
    encoded = graph + parameter_bytes
    return encoded, {
        "codec_id": "canonical-affine-svd",
        "codec_version": FORMAT_VERSION,
        "canonical_bytes_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "quotient_bits": 8 * len(encoded),
        "graph_bits": 8 * len(graph),
        "parameter_bits": 8 * len(parameter_bytes),
        "rank": rank,
        "step": step,
        "semantic_id": semantic_id,
    }


def decode_affine(encoded):
    fixed = struct.Struct(">8sBIIIdHQ")
    magic, version, din, dout, rank, step, semantic_length, parameter_length = \
        fixed.unpack(encoded[:fixed.size])
    if magic != MAGIC or version != FORMAT_VERSION:
        raise ValueError("unsupported affine encoding")
    offset = fixed.size + semantic_length
    if len(encoded) != offset + parameter_length:
        raise ValueError("encoded length mismatch")
    raw = zlib.decompress(encoded[offset:])
    counts = (din * rank, rank, rank * dout, dout)
    if len(raw) != 4 * sum(counts):
        raise ValueError("parameter length mismatch")
    values = np.frombuffer(raw, dtype="<i4")
    cuts = np.cumsum((0,) + counts)
    U = torch.from_numpy(values[cuts[0]:cuts[1]].copy()).reshape(din, rank).double() * step
    S = torch.from_numpy(values[cuts[1]:cuts[2]].copy()).double() * step
    Vh = torch.from_numpy(values[cuts[2]:cuts[3]].copy()).reshape(rank, dout).double() * step
    bias = torch.from_numpy(values[cuts[3]:cuts[4]].copy()).double() * step
    semantic = encoded[fixed.size:offset].decode("utf-8")
    return {"weight": (U * S) @ Vh, "bias": bias, "semantic_id": semantic,
            "rank": rank, "step": step}
