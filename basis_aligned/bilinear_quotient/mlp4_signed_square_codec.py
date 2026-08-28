#!/usr/bin/env python3
"""Conditional known-gauge codec using a difference of two square channels."""
from __future__ import annotations

import hashlib
import json
import struct
import zlib

import numpy as np
import torch

from . import mlp4_bilinear_residual_codec as product_codec

MAGIC = b"B4SQUAR1"
VERSION = 1
CODEC_ID = "mlp4-signed-square-known-gauge-v1"


def signed_square_factors(A, B, C):
    """Canonical product factors -> ``(u^Tz)^2-(v^Tz)^2`` factors."""
    A, B, C = product_codec.canonical_factors(A, B, C)
    U = (A+B)/2
    V = (A-B)/2
    # Square-vector signs are unobservable and are fixed independently.
    for factors in (U, V):
        for j in range(factors.shape[1]):
            pivot = int(factors[:, j].abs().argmax())
            if factors[pivot, j] < 0:
                factors[:, j] *= -1
    rows = [(U[:, j], V[:, j], C[j]) for j in range(U.shape[1])]
    rows.sort(key=lambda row: b"".join(
        value.contiguous().numpy().astype("<f8", copy=False).tobytes()
        for value in row))
    return (torch.stack([row[0] for row in rows], 1),
            torch.stack([row[1] for row in rows], 1),
            torch.stack([row[2] for row in rows], 0))


def encode(A, B, C, bias, step, source_ids, source_widths,
           output_id="mlp4.output"):
    if not step > 0:
        raise ValueError("quantization step must be positive")
    semantic = json.loads(product_codec._semantic(
        source_ids, source_widths, output_id))
    semantic.update({"codec": CODEC_ID,
                     "formula": "bias+sum_j c_j*((x@u_j)^2-(x@v_j)^2)",
                     "polarization_identity": True})
    semantic_bytes = json.dumps(
        semantic, sort_keys=True, separators=(",", ":")).encode()
    U, V, C = signed_square_factors(A, B, C)
    bias = bias.detach().double().cpu().clone()
    din, components = U.shape; dout = C.shape[1]
    if din != sum(source_widths) or bias.shape != (dout,):
        raise ValueError("factor dimensions do not match interface")
    quantized = []
    for value in (U, V, C, bias):
        q64 = torch.round(value/step).to(torch.int64)
        limits = torch.iinfo(torch.int32)
        if int(q64.min()) < limits.min or int(q64.max()) > limits.max:
            raise OverflowError("quantized coefficient exceeds int32")
        quantized.append(q64.to(torch.int32).contiguous().numpy())
    qU, qV, qC, _ = quantized
    for j in range(components):
        if not qU[:, j].any() or not qV[:, j].any() or not qC[j].any():
            raise ValueError("quantization collapsed a signed-square component")
    raw = b"".join(x.astype("<i4", copy=False).tobytes(order="C")
                   for x in quantized)
    payload = zlib.compress(raw, level=9)
    fixed = struct.pack(">8sBIIIdHQ", MAGIC, VERSION, din, dout, components,
                        float(step), len(semantic_bytes), len(payload))
    encoded = fixed+semantic_bytes+payload
    return encoded, {
        "codec_id": CODEC_ID, "codec_version": VERSION,
        "canonical_bytes_hash": "sha256:"+hashlib.sha256(encoded).hexdigest(),
        "conditional_known_gauge_bits": 8*len(encoded),
        "graph_bits": 8*(len(fixed)+len(semantic_bytes)),
        "parameter_bits": 8*len(payload), "component_count": components,
        "step": step,
        "quotiented_gauges": ["component scale", "component signs",
                              "input-leg swap", "square-vector signs",
                              "component permutation"],
        "global_cp_nonuniqueness_quotiented": False,
        "eligible_as_unconditional_quotient_price": False,
        "behavioral_roster_member": False,
    }


def decode(encoded):
    fixed = struct.Struct(">8sBIIIdHQ")
    magic, version, din, dout, components, step, semantic_size, payload_size = \
        fixed.unpack(encoded[:fixed.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported signed-square encoding")
    offset = fixed.size+semantic_size
    if len(encoded) != offset+payload_size:
        raise ValueError("encoded length mismatch")
    semantic = json.loads(encoded[fixed.size:offset].decode())
    raw = zlib.decompress(encoded[offset:])
    counts = (din*components, din*components, components*dout, dout)
    if len(raw) != 4*sum(counts):
        raise ValueError("parameter payload length mismatch")
    values = np.frombuffer(raw, dtype="<i4")
    cuts = np.cumsum((0,)+counts)
    arrays = [torch.from_numpy(values[cuts[i]:cuts[i+1]].copy()).double()*step
              for i in range(4)]
    return {"U": arrays[0].reshape(din, components),
            "V": arrays[1].reshape(din, components),
            "C": arrays[2].reshape(components, dout), "bias": arrays[3],
            "semantic": semantic, "step": step}


def execute_decoded(program, x):
    x = x.double()
    return program["bias"] + ((x@program["U"]).square()
                              -(x@program["V"]).square())@program["C"]
