#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin pronoun-antecedent wrapper for the generic managed FIT screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_pronoun as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_pronoun_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/pronoun_antecedent_gender_reference_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "pronoun-antecedent-gender-reference-v1"
EXPERIMENT_ID = "fast-screen-pronoun-antecedent-gender-reference-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "a4acf288af74f6e6787f01e06818a55d03174370323d07b6940cf85df964ab5b"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "0f6b9fb585ecf688109b9feabb86e5c08bf95b906ed9e3876f89cf95ce84d711"
)
REGISTERED_PREDICTIONS = (
    ("pred_a_native_capability", "Every ordered native capability cell passes."),
    ("pred_b_cross_construction_transfer", "One site transfers active and passive targets."),
    ("pred_c_controls_selective", "The selected site spares location and pronoun controls."),
)

CONFIG = managed.CandidateRunConfig(
    request_id=REQUEST_ID,
    experiment_id=EXPERIMENT_ID,
    prior_art_relative=PRIOR_ART.relative_to(ROOT).as_posix(),
    result_relative=RESULT_RELATIVE.as_posix(),
    ledger_relative=LEDGER.relative_to(ROOT).as_posix(),
    expected_prior_art_sha256=EXPECTED_PRIOR_ART_SHA256,
    expected_authority_sha256=EXPECTED_AUTHORITY_SHA256,
    information_read="which explicitly gendered person performed the action",
    proposed_operation=(
        "bind the selected event participant to the correct gendered pronoun across "
        "active and passive voice"
    ),
    proposed_write="evidence for he versus she",
    alternative_explanation=(
        "construction-specific actor-position cue or generic he/she output service"
    ),
    circuit_prediction=(
        "one site transfers A1 and A2 while sparing location P and visible-pronoun C"
    ),
    opposing_null_prediction=(
        "native capability fails or no site transfers both constructions selectively"
    ),
    semantic_position_role="final input token before the predicted pronoun",
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
