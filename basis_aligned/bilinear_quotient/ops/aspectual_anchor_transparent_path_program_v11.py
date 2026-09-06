#!/usr/bin/env python3
"""Operational-quotient extension of donor-free aspectual program v10."""

from __future__ import annotations

import aspectual_anchor_transparent_path_program_v10 as v10


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v11"
CONSTRUCTION_EQUIVALENCE_RESULT_SHA256 = "951044ca2b195d1128bebfabe2b3aa815407b20c02a20747e5bf18b8d6d13286"
FAMILY_EQUIVALENCE_RESULT_SHA256 = "b10b919f4c3cf9d53d1a397326f5d46a195ecd710f3df24fb8acff35966ea031"
LEXICON_EQUIVALENCE_RESULT_SHA256 = "a4f719b10b42d6452a51fb4d472439d6e19bba56ea92eed309f9915e049c0ac9"

ProgramInputError = v10.ProgramInputError
carrier_amplitude = v10.carrier_amplitude
rank1_carrier_projection = v10.rank1_carrier_projection
compiled_sparse_suffix_delta = v10.compiled_sparse_suffix_delta
exact_final_logits = v10.exact_final_logits
exact_scored_pair = v10.exact_scored_pair
exact_selected_margin = v10.exact_selected_margin
donor_free_margin_reflection = v10.donor_free_margin_reflection


def operational_quotient_manifest() -> dict[str, object]:
    """Declare the tested equivalence class of explicit v8 variable groups."""
    return {
        "variable_groups": {
            "initial": "resid10 donor-minus-base query displacement",
            "attention": "role-projected block11 and block15 source-attention deltas",
            "mlp": "explicit block11/12/14/15 bilinear-factor state tuples",
        },
        "downstream_reader": "blocks10-17 sparse recurrence plus exact final readout",
        "equivalences": {
            "construction": {
                "result_sha256": CONSTRUCTION_EQUIVALENCE_RESULT_SHA256,
                "whole_recovery_fraction_range": (0.7614164364576264, 1.328088119723955),
                "single_group_recovery_fraction_range": (0.8770960957982675, 1.152446697991182),
            },
            "A1_A2_family": {
                "result_sha256": FAMILY_EQUIVALENCE_RESULT_SHA256,
                "whole_recovery_fraction_range": (0.9024155783815451, 1.1236555332325653),
                "single_group_recovery_fraction_range": (0.9261138664226968, 1.0881374331929328),
            },
            "reporter_period_lexicon": {
                "result_sha256": LEXICON_EQUIVALENCE_RESULT_SHA256,
                "whole_recovery_fraction_range": (1.0173329613977797, 1.0800782615361804),
                "single_group_recovery_fraction_range": (0.9994659296116863, 1.046187532642156),
                "source_permutation": "same-direction group offset +6 modulo 16",
            },
        },
        "direction_fraction": 1.0,
        "claim": "operational equivalence on the registered lexical/fresh corpus under the named downstream reader",
        "not_claimed": ("unrestricted corpus invariance", "opposite-direction signed equivalence", "gauge-independent coordinate identity"),
    }


def program_manifest() -> dict[str, object]:
    """Return v10 plus construction/family/lexicon operational grouping evidence."""
    manifest = dict(v10.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "operational_quotient": operational_quotient_manifest(),
        "quotient_variable_count": 3,
        "native_module_boundaries_are_semantic_units": False,
    })
    return manifest
