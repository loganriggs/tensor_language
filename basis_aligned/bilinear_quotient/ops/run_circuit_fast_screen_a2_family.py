#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin numbered-list cached-value SUFFICIENCY wrapper for the generic managed FIT screen.

Complement of r576's removal null at `final_label_l0_value_through_l8h3_h7`: the cached-value weights are
held there and removal did NOT damage the behaviour (audited r579), so the path is not necessary. This asks
whether it is SUFFICIENT under donor interchange, and whether that holds across two list constructions.
"""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_a2_family as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_a2_family_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/numeric_sequence_a2_family_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "numeric-sequence-a2-family-v1"
EXPERIMENT_ID = "fast-screen-numeric-sequence-a2-family-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "c830386d049a3894e30f9b900cef8d26d4863d2b919371193f6a5cbcdfe39872"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "e3c41eb7af3ebf65824dda0fe0fd998dc179daf7a7aa675964340c06e7899ee0"
)
REGISTERED_PREDICTIONS = (
    ("pred_a_native_capability", "Every ordered native capability cell passes."),
    ("pred_b_cached_value_sufficiency",
     "One site transfers the cached label index in both the two-line and three-line constructions."),
    ("pred_c_controls_selective",
     "The selected site spares the surface-preserved P rows and the repeated-index C rows."),
)

CONFIG = managed.CandidateRunConfig(
    request_id=REQUEST_ID,
    experiment_id=EXPERIMENT_ID,
    prior_art_relative=PRIOR_ART.relative_to(ROOT).as_posix(),
    result_relative=RESULT_RELATIVE.as_posix(),
    ledger_relative=LEDGER.relative_to(ROOT).as_posix(),
    expected_prior_art_sha256=EXPECTED_PRIOR_ART_SHA256,
    expected_authority_sha256=EXPECTED_AUTHORITY_SHA256,
    information_read="the numeric-sequence continuation state at the final input token",
    proposed_operation=(
        "carry the sequence state forward and continue it in whichever numeral format the context uses"
    ),
    proposed_write="evidence for the next sequence value against the previous one",
    alternative_explanation=(
        "a construction-specific surface cue, or a generic numeral-output service that is "
        "insensitive to which index the list actually reached"
    ),
    circuit_prediction=(
        "one site transfers the sequence state in BOTH the digit and word constructions "
        "while sparing digit-surface-preserved P and the unrelated numbered-list C"
    ),
    opposing_null_prediction=(
        "native capability fails, or no site transfers both constructions selectively -- which "
        "would localise r576's redundancy rather than restate it"
    ),
    semantic_position_role="final input token before the predicted sequence value",
    batch_size=32,
    max_price=screen.battery.ExactPhasePrice(
        phase="FIT",
        forward_calls=528,
        example_evaluations=16896,
        backward_calls=0,
        model_updates=0,
        evidence_bytes=135168,
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
