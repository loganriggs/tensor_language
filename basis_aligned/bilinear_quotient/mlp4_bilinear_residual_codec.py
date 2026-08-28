#!/usr/bin/env python3
"""Known-gauge canonical codec for a labeled-source bilinear residual program.

The executable semantics are

    y = bias + sum_j c_j (x @ a_j) (x @ b_j).

This removes the explicit per-component scale/sign, input-leg swap, and component
permutation gauges. It does *not* claim to identify globally equivalent CP
decompositions, so its price is conditional on this factor-program language.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib

import numpy as np
import torch

from .mlp4_local_source_codec import FORBIDDEN_SOURCE_IDS

MAGIC = b"B4BILIN1"
VERSION = 1
CODEC_ID = "mlp4-bilinear-residual-known-gauge-v1"


def _pivot_positive(vector, output):
    pivot = int(vector.abs().argmax())
    if vector[pivot] < 0:
        vector = -vector
        output = -output
    return vector, output


def _key(*vectors):
    return b"".join(value.contiguous().numpy().astype("<f8", copy=False).tobytes()
                    for value in vectors)


def canonical_factors(A, B, C, zero_tolerance=1e-14):
    A = A.detach().double().cpu().clone()
    B = B.detach().double().cpu().clone()
    C = C.detach().double().cpu().clone()
    if A.ndim != 2 or B.shape != A.shape or C.ndim != 2 or C.shape[0] != A.shape[1]:
        raise ValueError("bilinear factor shapes are incompatible")
    if not all(torch.isfinite(value).all() for value in (A, B, C)):
        raise ValueError("bilinear factors must be finite")
    rows = []
    for j in range(A.shape[1]):
        a, b, c = A[:, j], B[:, j], C[j]
        na, nb, nc = a.norm(), b.norm(), c.norm()
        if min(float(na), float(nb), float(nc)) <= zero_tolerance:
            raise ValueError("zero or near-zero component has a singular gauge orbit")
        a = a / na; b = b / nb; c = c * na * nb
        a, c = _pivot_positive(a, c)
        b, c = _pivot_positive(b, c)
        if _key(b) < _key(a):
            a, b = b, a
        rows.append((a, b, c))
    rows.sort(key=lambda row: _key(*row))
    if any(_key(*left) == _key(*right) for left, right in zip(rows, rows[1:])):
        raise ValueError("duplicate canonical components require a merge stratum")
    return (torch.stack([row[0] for row in rows], 1),
            torch.stack([row[1] for row in rows], 1),
            torch.stack([row[2] for row in rows], 0))


def _semantic(source_ids, source_widths, output_id):
    if not source_ids or len(source_ids) != len(source_widths):
        raise ValueError("source IDs and widths must be nonempty and aligned")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique")
    if any(source in FORBIDDEN_SOURCE_IDS for source in source_ids):
        raise ValueError("candidate may not read native MLP4 parameters or output")
    if any(not isinstance(width, int) or width <= 0 for width in source_widths):
        raise ValueError("source widths must be positive integers")
    return json.dumps({
        "codec": CODEC_ID,
        "formula": "bias+sum_j c_j*(x@a_j)*(x@b_j)",
        "global_cp_nonuniqueness_quotiented": False,
        "output": output_id,
        "sources": [{"id": source, "width": width}
                    for source, width in zip(source_ids, source_widths)],
    }, sort_keys=True, separators=(",", ":"))


def encode(A, B, C, bias, step, source_ids, source_widths,
           output_id="mlp4.output", zero_tolerance=1e-14):
    if not step > 0:
        raise ValueError("quantization step must be positive")
    A, B, C = canonical_factors(A, B, C, zero_tolerance)
    bias = bias.detach().double().cpu().clone()
    din, components = A.shape
    dout = C.shape[1]
    if din != sum(source_widths) or bias.shape != (dout,):
        raise ValueError("factor dimensions do not match source widths or bias")
    semantic = _semantic(source_ids, source_widths, output_id).encode()
    quantized = []
    for value in (A, B, C, bias):
        q64 = torch.round(value / step).to(torch.int64)
        limits = torch.iinfo(torch.int32)
        if int(q64.min()) < limits.min or int(q64.max()) > limits.max:
            raise OverflowError("quantized coefficient exceeds int32")
        quantized.append(q64.to(torch.int32).contiguous().numpy())
    qA, qB, qC, _ = quantized
    for j in range(components):
        if not qA[:, j].any() or not qB[:, j].any() or not qC[j].any():
            raise ValueError("quantization collapsed a component to zero")
    component_keys = [qA[:, j].tobytes() + qB[:, j].tobytes() + qC[j].tobytes()
                      for j in range(components)]
    if len(set(component_keys)) != components:
        raise ValueError("quantization collapsed distinct components together")
    raw = b"".join(value.astype("<i4", copy=False).tobytes(order="C")
                   for value in quantized)
    payload = zlib.compress(raw, level=9)
    fixed = struct.pack(">8sBIIIdHQ", MAGIC, VERSION, din, dout, components,
                        float(step), len(semantic), len(payload))
    encoded = fixed + semantic + payload
    graph_bits = 8 * (len(fixed) + len(semantic))
    return encoded, {
        "codec_id": CODEC_ID,
        "codec_version": VERSION,
        "canonical_bytes_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "conditional_known_gauge_bits": 8 * len(encoded),
        "graph_bits": graph_bits,
        "parameter_bits": 8 * len(payload),
        "component_count": components,
        "step": step,
        "quotiented_gauges": [
            "component scale", "component signs", "input-leg swap",
            "component permutation",
        ],
        "global_cp_nonuniqueness_quotiented": False,
        "eligible_as_unconditional_quotient_price": False,
        "source_producer_payloads_included": False,
        "whole_program_mdl_eligible": False,
    }


def decode(encoded):
    fixed = struct.Struct(">8sBIIIdHQ")
    magic, version, din, dout, components, step, semantic_size, payload_size = \
        fixed.unpack(encoded[:fixed.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported bilinear residual encoding")
    offset = fixed.size + semantic_size
    if len(encoded) != offset + payload_size:
        raise ValueError("encoded length mismatch")
    semantic = json.loads(encoded[fixed.size:offset].decode())
    raw = zlib.decompress(encoded[offset:])
    counts = (din*components, din*components, components*dout, dout)
    if len(raw) != 4 * sum(counts):
        raise ValueError("parameter payload length mismatch")
    values = np.frombuffer(raw, dtype="<i4")
    cuts = np.cumsum((0,) + counts)
    arrays = [torch.from_numpy(values[cuts[i]:cuts[i+1]].copy()).double() * step
              for i in range(4)]
    A = arrays[0].reshape(din, components)
    B = arrays[1].reshape(din, components)
    C = arrays[2].reshape(components, dout)
    return {"A": A, "B": B, "C": C, "bias": arrays[3],
            "semantic": semantic, "step": step}


def execute(encoded, sources, source_ids):
    program = decode(encoded)
    expected = program["semantic"]["sources"]
    if list(source_ids) != [source["id"] for source in expected]:
        raise ValueError("runtime source order differs from encoded graph")
    if len(sources) != len(expected) or any(
            value.shape[-1] != source["width"]
            for value, source in zip(sources, expected)):
        raise ValueError("runtime source widths differ from encoded graph")
    prefix = sources[0].shape[:-1]
    if any(value.shape[:-1] != prefix for value in sources):
        raise ValueError("runtime source batch shapes differ")
    x = torch.cat([value.double() for value in sources], -1)
    products = (x @ program["A"]) * (x @ program["B"])
    return program["bias"] + products @ program["C"]
