#!/usr/bin/env python3
"""Native-dtype instrument repair for the sealed will/had block-8 component cube."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_cube pred_b_boundary9_effect_recurrence pred_c_attention8_is_dominant_writer pred_d_attention8_is_material pred_e_exact_zero_fit_price
from __future__ import annotations

import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import run_temporal_auxiliary_will_had_subject_onset_block8_component_cube_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_subject_onset_block8_component_cube_v2.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_subject_onset_block8_component_cube_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.subject_onset_block8_component_cube_v2"
EXPECTED_PRIOR_SHA256 = "fcba0ecc439a2f076133a12f0b068bcb133263a38cb5439bfbc92d254fdbde4b"
NATIVE_TOLERANCE = 1.0e-7
PREDICATES = (
    "pred_a_authority_capability_exact_cube",
    "pred_b_boundary9_effect_recurrence",
    "pred_c_attention8_is_dominant_writer",
    "pred_d_attention8_is_material",
    "pred_e_exact_zero_fit_price",
)


def native_component_states(raw):
    return {
        v1.BRANCHES[0]: raw["z8"].detach().clone(),
        v1.BRANCHES[1]: raw["attention8"].detach().clone(),
        v1.BRANCHES[2]: raw["mlp8"].detach().clone(),
    }


def native_sum(values, _start=None):
    pieces = tuple(values)
    if len(pieces) != 3:
        raise v1.ExperimentError("native block sum must contain three ordered components")
    return (pieces[0] + pieces[1]) + pieces[2]


def native_assembled_state(base_components, donor_components, subset):
    selected = set(subset)
    pieces = tuple(
        donor_components[branch] if branch in selected else base_components[branch]
        for branch in v1.BRANCHES
    )
    return native_sum(pieces)


def configure_repair():
    if v1.sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise v1.ExperimentError("v2 prior-art hash changed")
    v1.PRIOR = PRIOR
    v1.OUT = OUT
    v1.CANDIDATE_ID = CANDIDATE_ID
    v1.EXPECTED = {**v1.EXPECTED, "prior": EXPECTED_PRIOR_SHA256}
    v1.component_states = native_component_states
    v1.assembled_state = native_assembled_state
    v1.sum = native_sum

    native_error = {"maximum": 0.0}
    original_capture = v1.block_cube.capture_components

    def capture_with_native_closure(backend, batch):
        result = original_capture(backend, batch)
        raw = result[3]
        x9 = native_sum((raw["z8"], raw["attention8"], raw["mlp8"]))
        block9 = backend.model.transformer.h[9]
        live9 = block9.lambdas[0] * x9 + block9.lambdas[1] * raw["x0"]
        native_error["maximum"] = max(
            native_error["maximum"], float((live9 - raw["z9"]).abs().max())
        )
        return result

    original_write = managed.atomic_create_json

    def write_v2(_path, result):
        result["schema"] = "temporal_auxiliary_will_had_subject_onset_block8_component_cube_result_v2"
        result["candidate_id"] = CANDIDATE_ID
        result["instrument"]["float32_reassembly_diagnostic_max_abs"] = result["instrument"].pop(
            "block8_component_recombination_max_abs"
        )
        result["instrument"]["native_order_block8_recombination_max_abs"] = native_error["maximum"]
        exact = max(
            result["instrument"][name]
            for name in (
                "native_order_block8_recombination_max_abs",
                "full_state_closure_max_abs",
                "empty_scored_logit_closure_max_abs",
                "shapley_efficiency_max_abs",
            )
        ) <= NATIVE_TOLERANCE
        result["predictions"][PREDICATES[0]] = bool(
            result["predictions"][PREDICATES[0]]
            and result["instrument"]["manual_scored_logit_max_abs"] <= 1.0e-4
            and exact
        )
        predictions = result["predictions"]
        result["terminal"] = "screen" if all(predictions.values()) else (
            "null" if predictions[PREDICATES[0]] and predictions[PREDICATES[1]] and predictions[PREDICATES[4]]
            else "invalid"
        )
        result["reason"] = {
            "screen": "block8_attention_is_dominant_material_subject_onset_writer",
            "null": "registered_block8_attention_writer_prediction_missed",
            "invalid": "native_order_instrument_authority_recurrence_coverage_or_price_invalid",
        }[result["terminal"]]
        result["next_action"] = (
            "partition block8 attention by head and source group"
            if result["terminal"] == "screen"
            else "retain the valid component cube and localize why carried entry state has the largest Shapley value"
        )
        original_write(OUT, result)

    v1.block_cube.capture_components = capture_with_native_closure
    v1.atomic_create_json = write_v2
    # The obsolete float32 diagnostic is deliberately non-terminal in v2.
    v1.EXACT_TOLERANCE = 1.0


def main():
    configure_repair()
    v1.main()


if __name__ == "__main__":
    main()
