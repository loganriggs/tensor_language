"""Deterministic prototype codec for one shared-depth V/O head orbit.

This is a CPU fixture, not a promoted checkpoint price.  It first chooses the
canonical section of the exact shared GL(head_dim) gauge and only then quantizes.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch

from .shared_value_gauge import canonicalize_shared_value_bus


MAGIC = b"SVOC1"


def _quantize(matrix, exponent):
    scale = 2.0 ** exponent
    integers = torch.round(matrix * scale)
    limit = torch.iinfo(torch.int32)
    if not torch.isfinite(integers).all() or integers.min() < limit.min or integers.max() > limit.max:
        raise ValueError("quantized coefficient exceeds signed int32 range")
    return integers.to(torch.int32).contiguous()


def encode_shared_head(value_maps, output_maps, quantization_exponent=16,
                       relative_tolerance=1e-9):
    """Canonicalize and serialize one head shared across every depth."""
    if not isinstance(quantization_exponent, int) or not 0 <= quantization_exponent <= 24:
        raise ValueError("quantization_exponent must be an integer in [0, 24]")
    values, outputs = canonicalize_shared_value_bus(
        value_maps, output_maps, relative_tolerance=relative_tolerance)
    if any(matrix.shape != values[0].shape for matrix in values):
        raise ValueError("all value maps must have one shared shape")
    expected_output_shape = (values[0].shape[1], values[0].shape[0])
    if any(matrix.shape != expected_output_shape for matrix in outputs):
        raise ValueError("all output head blocks must transpose the value-map shape")
    arrays = [_quantize(matrix, quantization_exponent) for matrix in values + outputs]
    header = {"schema_version": 1, "layers": len(values),
              "head_dimension": values[0].shape[0],
              "model_dimension": values[0].shape[1],
              "quantization_exponent": quantization_exponent,
              "tensor_order": [f"V{layer}" for layer in range(len(values))] +
                              [f"O{layer}" for layer in range(len(outputs))],
              "integer_dtype": "int32_le"}
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = b"".join(np.asarray(array.numpy(), dtype="<i4").tobytes(order="C")
                       for array in arrays)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload


def decode_shared_head(encoded):
    """Decode a prototype container into quantized canonical V/O maps."""
    if not isinstance(encoded, (bytes, bytearray)) or encoded[:len(MAGIC)] != MAGIC:
        raise ValueError("invalid shared V/O container magic")
    offset = len(MAGIC)
    if len(encoded) < offset + 4:
        raise ValueError("truncated shared V/O header")
    header_size = struct.unpack("<I", encoded[offset:offset+4])[0]
    offset += 4
    try:
        header = json.loads(encoded[offset:offset+header_size])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid shared V/O JSON header") from exc
    offset += header_size
    layers = header["layers"]; head = header["head_dimension"]; model = header["model_dimension"]
    count = layers * head * model
    expected_bytes = 2 * count * 4
    if len(encoded) - offset != expected_bytes:
        raise ValueError("shared V/O payload length does not match header")
    flat = np.frombuffer(encoded, dtype="<i4", offset=offset).astype(np.int32, copy=True)
    scale = 2.0 ** header["quantization_exponent"]
    tensors = torch.from_numpy(flat).double() / scale
    values = [row.clone() for row in tensors[:count].reshape(layers, head, model)]
    outputs = [row.clone() for row in tensors[count:].reshape(layers, model, head)]
    return header, values, outputs


def descriptive_bits(encoded):
    """Literal container length; not an operationally promoted MDL price."""
    return 8 * len(encoded)
