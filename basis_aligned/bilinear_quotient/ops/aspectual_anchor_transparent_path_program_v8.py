#!/usr/bin/env python3
"""Prospectively transferred scope release of executable aspectual program v7."""

from __future__ import annotations

import aspectual_anchor_transparent_path_program_v7 as v7


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v8"
TRANSFER_CONSTRUCTIONS = (
    "archive_evidential_prefix_anchor",
    "explanatory_subordinate_prefix_anchor",
)

ProgramInputError = v7.ProgramInputError
MLP_FACTORS = v7.MLP_FACTORS
SUFFIX_BOUNDARIES = v7.SUFFIX_BOUNDARIES
SUFFIX_SOURCE_BOUNDARIES = v7.SUFFIX_SOURCE_BOUNDARIES
SUFFIX_MLP_FACTORS_BY_BOUNDARY = v7.SUFFIX_MLP_FACTORS_BY_BOUNDARY
compiled_sparse_suffix_delta = v7.compiled_sparse_suffix_delta
selected_suffix_mlp_write_delta = v7.selected_suffix_mlp_write_delta
bilinear_hidden_delta = v7.bilinear_hidden_delta
exact_final_logits = v7.exact_final_logits
exact_scored_pair = v7.exact_scored_pair
compiled_sparse_suffix_scored_pair = v7.compiled_sparse_suffix_scored_pair


def program_manifest() -> dict[str, object]:
    """Return v7 unchanged with its prospectively validated construction scope."""
    manifest = dict(v7.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "prospective_transfer_constructions": TRANSFER_CONSTRUCTIONS,
        "prospective_transfer_target_rows": 32,
        "prospective_transfer_direction_fraction": {"A1": 1.0, "A2": 1.0},
        "prospective_transfer_writer_retention": {
            "archive_evidential_prefix_anchor": 0.9372244190381819,
            "explanatory_subordinate_prefix_anchor": 0.8678597057234848,
            "pooled": 0.8999465787382264,
        },
    })
    return manifest
