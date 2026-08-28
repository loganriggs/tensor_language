#!/usr/bin/env python3
"""Canonical reduced-rank affine program over explicitly labeled DAG sources."""
from __future__ import annotations

import json

import torch

from . import affine_codec

CODEC_ID = "mlp4-source-labeled-affine-svd-v1"
FORBIDDEN_SOURCE_IDS = {
    "mlp4.Left", "mlp4.Right", "mlp4.Down", "mlp4.forward", "mlp4.output",
}


def semantic_id(source_ids, source_widths, output_id="mlp4.output"):
    if not source_ids or len(source_ids) != len(source_widths):
        raise ValueError("source IDs and widths must be nonempty and aligned")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique")
    if any(source in FORBIDDEN_SOURCE_IDS for source in source_ids):
        raise ValueError("candidate may not read native MLP4 parameters or output")
    if any(not isinstance(width, int) or width <= 0 for width in source_widths):
        raise ValueError("source widths must be positive integers")
    graph = {
        "codec": CODEC_ID,
        "formula": "concat(labeled_sources)@W+b",
        "output": output_id,
        "sources": [{"id": source, "width": width}
                    for source, width in zip(source_ids, source_widths)],
    }
    return json.dumps(graph, sort_keys=True, separators=(",", ":"))


def encode(U, S, Vh, xm, ym, rank, step, source_ids, source_widths,
           degeneracy_tolerance=1e-7):
    if U.shape[0] != sum(source_widths) or xm.numel() != sum(source_widths):
        raise ValueError("factor input dimension does not match source widths")
    semantics = semantic_id(source_ids, source_widths)
    encoded, price = affine_codec.canonical_affine_bytes(
        U, S, Vh, xm, ym, rank, step, semantic_id=semantics,
        degeneracy_tolerance=degeneracy_tolerance,
    )
    return encoded, {
        **price,
        "wrapper_codec_id": CODEC_ID,
        "source_ids": list(source_ids),
        "source_widths": list(source_widths),
        "source_producer_payloads_included": False,
        "source_edge_schema_included": True,
        "whole_program_mdl_eligible": False,
    }


def execute(encoded, sources, source_ids):
    decoded = affine_codec.decode_affine(encoded)
    graph = json.loads(decoded["semantic_id"])
    expected_ids = [source["id"] for source in graph["sources"]]
    expected_widths = [source["width"] for source in graph["sources"]]
    if list(source_ids) != expected_ids or len(sources) != len(expected_ids):
        raise ValueError("runtime source order differs from encoded graph")
    if any(value.shape[-1] != width
           for value, width in zip(sources, expected_widths)):
        raise ValueError("runtime source width differs from encoded graph")
    prefix = sources[0].shape[:-1]
    if any(value.shape[:-1] != prefix for value in sources):
        raise ValueError("runtime source batch shapes differ")
    x = torch.cat([value.double() for value in sources], dim=-1)
    return x @ decoded["weight"] + decoded["bias"]
