#!/usr/bin/env python3
"""Global-residual canonical wrapper around production Q/K and V/O codecs."""
from __future__ import annotations

from .attention_head_codec import canonical_head_bytes, canonical_lowrank_head_bytes
from .residual_basis_gauge import canonical_frame, transform_read_weight, \
    transform_write_weight
from .shared_value_output_codec import encode_qk_keyed_heads


QK_NAMES = ("q", "k", "q2", "k2")


def canonicalize_qk_maps(anchor, maps_by_head, residual_tolerance=1e-10):
    if not maps_by_head:
        raise ValueError("at least one Q/K head is required")
    residual = canonical_frame(anchor, residual_tolerance)
    frame = residual["frame"]
    transformed = []
    for maps in maps_by_head:
        if set(maps) != set(QK_NAMES):
            raise ValueError(f"each head requires {QK_NAMES}")
        transformed.append({name: transform_read_weight(maps[name], frame)
                            for name in QK_NAMES})
    return residual, transformed


def encode_nested_qk_routes(anchor, maps_by_head, quantization_step, *, rank=None,
                            residual_tolerance=1e-10, qk_tolerance=1e-10,
                            semantic_id="bilin18.product_attention.head"):
    """Encode route bytes only after fixing the shared residual frame."""
    residual, transformed = canonicalize_qk_maps(
        anchor, maps_by_head, residual_tolerance)
    if rank is None:
        routes = [canonical_head_bytes(
            maps, quantization_step, degeneracy_tol=qk_tolerance,
            semantic_id=semantic_id) for maps in transformed]
    else:
        routes = [canonical_lowrank_head_bytes(
            maps, rank, quantization_step, degeneracy_tol=qk_tolerance,
            semantic_id=semantic_id) for maps in transformed]
    return residual, routes


def encode_nested_attention_heads(anchor, qk_maps_by_head, value_maps_by_head,
                                  output_maps_by_head, qk_quantization_step,
                                  vo_quantization_exponent=16, *, rank=None,
                                  residual_tolerance=1e-10, qk_tolerance=1e-10,
                                  value_tolerance=1e-9):
    """Encode route-bound V/O bytes under global, local, and common-head gauges."""
    if not (len(qk_maps_by_head) == len(value_maps_by_head)
            == len(output_maps_by_head)):
        raise ValueError("one Q/K and V/O program per head required")
    residual, routes = encode_nested_qk_routes(
        anchor, qk_maps_by_head, qk_quantization_step, rank=rank,
        residual_tolerance=residual_tolerance, qk_tolerance=qk_tolerance)
    frame = residual["frame"]
    values = [[transform_read_weight(value, frame) for value in head]
              for head in value_maps_by_head]
    outputs = [[transform_write_weight(output, frame) for output in head]
               for head in output_maps_by_head]
    bundle = encode_qk_keyed_heads(values, outputs, routes,
                                   vo_quantization_exponent, value_tolerance)
    return {"canonical_anchor": residual["canonical_anchor"],
            "routes": routes, "route_bound_vo_bundle": bundle}
