#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin pending-quote-parity wrapper for the generic managed FIT screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_possessive_number as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_possessive_number_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/possessive_number_their_vs_his_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "possessive-number-their-vs-his-v1"
EXPERIMENT_ID = "fast-screen-possessive-number-their-vs-his-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "e86317de4deb750de82e1ed5aa4deff4172ff82989a38713ae80f66f08cdbf62"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "310a2152d4575f2fd809340c06c2e963e4f86efa7068fdd4c3c7dae6c1c64b8d"
)
REGISTERED_PREDICTIONS = (
    ("pred_a_native_capability", "Every ordered native capability cell passes."),
    ("pred_b_cross_construction_transfer", "One site transfers both pending-quote constructions."),
    ("pred_c_controls_selective", "The selected site spares writer and endpoint controls."),
)

CONFIG = managed.CandidateRunConfig(
    request_id=REQUEST_ID,
    experiment_id=EXPERIMENT_ID,
    prior_art_relative=PRIOR_ART.relative_to(ROOT).as_posix(),
    result_relative=RESULT_RELATIVE.as_posix(),
    ledger_relative=LEDGER.relative_to(ROOT).as_posix(),
    expected_prior_art_sha256=EXPECTED_PRIOR_ART_SHA256,
    expected_authority_sha256=EXPECTED_AUTHORITY_SHA256,
    information_read="the number of the antecedent noun",
    proposed_operation="carry antecedent number to a possessive pronoun in two constructions",
    proposed_write="evidence for their rather than his",
    alternative_explanation=(
        "a construction-local inflection cue rather than a carried temporal frame"
    ),
    circuit_prediction=(
        "one site transfers the was/is decision in both constructions while sparing the subject rewrite"
    ),
    opposing_null_prediction=(
        "native capability fails, or no site transfers both constructions"
    ),
    semantic_position_role="final input token before the determiner",
    batch_size=32,
    max_price=screen.battery.ExactPhasePrice(
        phase="FIT",
        forward_calls=264,
        example_evaluations=8448,
        backward_calls=0,
        model_updates=0,
        evidence_bytes=67584,
    ),
)

utc_now = managed.utc_now
utc_text = managed.utc_text
literal_json = managed.literal_json
atomic_create_json = managed.atomic_create_json
selected_controls_pass = managed.selected_controls_pass


def build_spec(rows: list[dict[str, object]]) -> screen.CircuitFastScreenSpec:
    return managed.build_spec(CONFIG, candidate, rows)


def main() -> None:
    managed.run_managed(CONFIG, candidate, root=ROOT)


if __name__ == "__main__":
    main()
