#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin pending-quote-parity wrapper for the generic managed FIT screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_existential as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_existential_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/existential_agreement_were_vs_was_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "existential-agreement-were-vs-was-v1"
EXPERIMENT_ID = "fast-screen-existential-agreement-were-vs-was-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "3b2644bc0c46857ba4878ecc9f134d8a7487f224b14931714c0079dc0db8e3be"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "aabab4b0c3a9ad6cc7aba51692f70b68d95ee5d37a21df6642984d6add268b50"
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
    information_read="the number of the notional subject",
    proposed_operation="carry notional number across an expletive in two constructions",
    proposed_write="evidence for were rather than was",
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
