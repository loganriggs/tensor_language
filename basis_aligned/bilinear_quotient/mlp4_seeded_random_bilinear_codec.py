#!/usr/bin/env python3
"""Portable seeded random-bilinear control with exact serialized-budget selection."""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib

import numpy as np
import torch

from .mlp4_local_source_codec import FORBIDDEN_SOURCE_IDS

MAGIC = b"B4RANDB1"
VERSION = 1
CODEC_ID = "mlp4-shake256-rademacher-bilinear-v1"
ALGORITHM = "SHAKE256(component,leg)->Rademacher/sqrt(din)"


def _validate_sources(source_ids, source_widths):
    if not source_ids or len(source_ids) != len(source_widths):
        raise ValueError("source IDs and widths must be nonempty and aligned")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique")
    if any(source in FORBIDDEN_SOURCE_IDS for source in source_ids):
        raise ValueError("candidate may not read native MLP4 parameters or output")
    if any(not isinstance(width, int) or width <= 0 for width in source_widths):
        raise ValueError("source widths must be positive integers")


def _rademacher_vector(seed, component, leg, din):
    label = f"{CODEC_ID}|{seed}|{component}|{leg}|{din}".encode()
    packed = hashlib.shake_256(label).digest((din + 7) // 8)
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:din]
    values = bits.astype(np.float64) * 2.0 - 1.0
    return torch.from_numpy(values) / math.sqrt(din)


def feature_factors(seed, din, components):
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a nonempty semantic string")
    if din <= 0 or components <= 0:
        raise ValueError("feature dimensions must be positive")
    A = torch.stack([_rademacher_vector(seed, j, "a", din)
                     for j in range(components)], 1)
    B = torch.stack([_rademacher_vector(seed, j, "b", din)
                     for j in range(components)], 1)
    return A, B


def encode(C, bias, step, seed, source_ids, source_widths,
           output_id="mlp4.output"):
    _validate_sources(source_ids, source_widths)
    if not step > 0:
        raise ValueError("quantization step must be positive")
    C = C.detach().double().cpu().clone()
    bias = bias.detach().double().cpu().clone()
    if C.ndim != 2 or bias.shape != (C.shape[1],) or C.shape[0] <= 0:
        raise ValueError("output factors and bias have incompatible shapes")
    if not torch.isfinite(C).all() or not torch.isfinite(bias).all():
        raise ValueError("output factors and bias must be finite")
    din = sum(source_widths); components, dout = C.shape
    semantic = json.dumps({
        "algorithm": ALGORITHM,
        "codec": CODEC_ID,
        "formula": "bias+sum_j c_j*(x@random_a_j)*(x@random_b_j)",
        "output": output_id,
        "seed": seed,
        "sources": [{"id": source, "width": width}
                    for source, width in zip(source_ids, source_widths)],
    }, sort_keys=True, separators=(",", ":")).encode()
    quantized = []
    for value in (C, bias):
        q64 = torch.round(value / step).to(torch.int64)
        limits = torch.iinfo(torch.int32)
        if int(q64.min()) < limits.min or int(q64.max()) > limits.max:
            raise OverflowError("quantized coefficient exceeds int32")
        quantized.append(q64.to(torch.int32).contiguous().numpy())
    qC, qbias = quantized
    if any(not qC[j].any() for j in range(components)):
        raise ValueError("quantization collapsed a learned output component to zero")
    raw = qC.astype("<i4", copy=False).tobytes(order="C") + \
        qbias.astype("<i4", copy=False).tobytes(order="C")
    payload = zlib.compress(raw, level=9)
    fixed = struct.pack(">8sBIIIdHQ", MAGIC, VERSION, din, dout, components,
                        float(step), len(semantic), len(payload))
    encoded = fixed + semantic + payload
    graph_bits = 8 * (len(fixed) + len(semantic))
    return encoded, {
        "codec_id": CODEC_ID,
        "codec_version": VERSION,
        "canonical_bytes_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "canonical_program_bits": 8 * len(encoded),
        "graph_bits": graph_bits,
        "learned_parameter_bits": 8 * len(payload),
        "component_count": components,
        "step": step,
        "seed": seed,
        "random_input_factor_bits": 0,
        "random_input_factors_generated_by_priced_graph": True,
        "source_producer_payloads_included": False,
        "minimal_program_claim": False,
        "whole_program_mdl_eligible": False,
    }


def decode(encoded):
    fixed = struct.Struct(">8sBIIIdHQ")
    magic, version, din, dout, components, step, semantic_size, payload_size = \
        fixed.unpack(encoded[:fixed.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported random bilinear encoding")
    offset = fixed.size + semantic_size
    if len(encoded) != offset + payload_size:
        raise ValueError("encoded length mismatch")
    semantic = json.loads(encoded[fixed.size:offset].decode())
    raw = zlib.decompress(encoded[offset:])
    count = components*dout + dout
    if len(raw) != 4*count:
        raise ValueError("parameter payload length mismatch")
    values = torch.from_numpy(np.frombuffer(raw, dtype="<i4").copy()).double()*step
    return {"C": values[:components*dout].reshape(components, dout),
            "bias": values[components*dout:], "din": din,
            "components": components, "semantic": semantic, "step": step}


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
    A, B = feature_factors(program["semantic"]["seed"], program["din"],
                           program["components"])
    return program["bias"] + ((x@A)*(x@B)) @ program["C"]


def select_prefix_at_or_below_bits(C, bias, step, seed, source_ids,
                                   source_widths, component_counts,
                                   target_bits):
    counts = sorted(set(component_counts))
    if not counts or counts[0] <= 0 or counts[-1] > C.shape[0]:
        raise ValueError("component-count grid is invalid for learned factors")
    candidates = []
    for count in counts:
        encoded, price = encode(C[:count], bias, step, seed, source_ids,
                                source_widths)
        candidates.append((count, encoded, price))
    eligible = [candidate for candidate in candidates
                if candidate[2]["canonical_program_bits"] <= target_bits]
    if not eligible:
        raise ValueError("no preregistered random-feature prefix fits target bits")
    selected = max(eligible, key=lambda candidate: (
        candidate[2]["canonical_program_bits"], candidate[0]))
    return selected[1], {
        **selected[2],
        "target_bits": target_bits,
        "unused_budget_bits": target_bits-selected[2]["canonical_program_bits"],
        "selection_grid": counts,
        "selection_uses_behavioral_results": False,
    }
