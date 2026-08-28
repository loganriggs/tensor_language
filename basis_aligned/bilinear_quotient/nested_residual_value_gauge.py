#!/usr/bin/env python3
"""Ordered canonicalization of global residual and shared-value gauges."""
from __future__ import annotations

import torch

from .residual_basis_gauge import canonical_frame, transform_read_weight, \
    transform_write_weight
from .shared_value_gauge import canonicalize_shared_value_bus
from .shared_value_output_codec import encode_qk_keyed_heads


def canonicalize_nested(anchor, value_maps_by_head, output_maps_by_head,
                        residual_tolerance=1e-10, value_tolerance=1e-9):
    """Fix global ``O(D)`` first, then one shared-depth ``GL(h)`` per head."""
    if not value_maps_by_head or len(value_maps_by_head) != len(output_maps_by_head):
        raise ValueError("matched nonempty head collections required")
    residual = canonical_frame(anchor, residual_tolerance)
    frame = residual["frame"]
    canonical_values = []
    canonical_outputs = []
    layers = None
    for values, outputs in zip(value_maps_by_head, output_maps_by_head):
        if not values or len(values) != len(outputs):
            raise ValueError("each head needs matched nonempty depth maps")
        if layers is None:
            layers = len(values)
        elif len(values) != layers:
            raise ValueError("all heads must span the same layers")
        globally_rotated_values = [transform_read_weight(value, frame)
                                   for value in values]
        globally_rotated_outputs = [transform_write_weight(output, frame)
                                    for output in outputs]
        values_fixed, outputs_fixed = canonicalize_shared_value_bus(
            globally_rotated_values, globally_rotated_outputs, value_tolerance)
        canonical_values.append(values_fixed)
        canonical_outputs.append(outputs_fixed)
    return {"residual_frame": frame,
            "canonical_anchor": residual["canonical_anchor"],
            "value_maps_by_head": canonical_values,
            "output_maps_by_head": canonical_outputs}


def encode_nested_qk_keyed_heads(anchor, value_maps_by_head, output_maps_by_head,
                                 routing_keys, quantization_exponent=16,
                                 residual_tolerance=1e-10, value_tolerance=1e-9):
    """Canonical nested-gauge V/O bytes, with Q/K keys fixing common head order."""
    nested = canonicalize_nested(
        anchor, value_maps_by_head, output_maps_by_head,
        residual_tolerance, value_tolerance)
    return encode_qk_keyed_heads(
        nested["value_maps_by_head"], nested["output_maps_by_head"], routing_keys,
        quantization_exponent, value_tolerance)


def generic_combined_gauge_dimension(model_dimension, head_dimension, heads):
    if min(model_dimension, head_dimension, heads) <= 0 \
            or head_dimension > model_dimension:
        raise ValueError("invalid gauge dimensions")
    residual = model_dimension*(model_dimension-1)//2
    values = heads*head_dimension*head_dimension
    return {"global_residual_orthogonal": residual,
            "shared_value_general_linear": values,
            "generic_combined": residual+values}
