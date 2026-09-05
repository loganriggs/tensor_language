#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin pending-quote-parity wrapper for the generic managed FIT screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_narrative_tense_short_cue as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_narrative_tense_short_cue_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/narrative_tense_short_cue_distance_v2_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "narrative-tense-short-cue-distance-v2"
EXPERIMENT_ID = "fast-screen-narrative-tense-short-cue-distance-v2"
EXPECTED_AUTHORITY_SHA256 = (
    "c345d553ff43f9a11981d8ceab35433c84ae52a66c7d49b68c021f1dade8a730"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "1bff2deb37351390350f68aedee45f47001c40c48ddbf7227821460f32534c24"
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
    information_read="whether the narrative frame is past or present, with the cue adjacent",
    proposed_operation="read the tense cue two to four tokens back in two constructions",
    proposed_write="evidence for a past copula rather than a present one",
    alternative_explanation=(
        "a construction-local inflection cue rather than a carried temporal frame"
    ),
    circuit_prediction=(
        "one site transfers the was/is decision in both constructions while sparing the subject rewrite"
    ),
    opposing_null_prediction=(
        "native capability fails, or no site transfers both constructions"
    ),
    semantic_position_role="final input token before the predicted copula",
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
