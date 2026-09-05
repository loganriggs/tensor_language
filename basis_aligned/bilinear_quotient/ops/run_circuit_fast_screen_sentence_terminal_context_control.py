#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin sentence-terminal wrapper with a context-driven control, for the generic managed FIT screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_sentence_terminal_context_control as candidates
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_sentence_terminal_context_control_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/sentence_terminal_semantic_choice_context_control_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "sentence-terminal-semantic-choice-context-control-v1"
EXPERIMENT_ID = "fast-screen-sentence-terminal-semantic-choice-context-control-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "ee62cf7ee83891e84ed8f75943a534b8dd606d00607cc800deb03d1d0077e289"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "0c955b4c5e6004be56228be26ce86b06892218266fba9ad78bde4564cb5e9f4e"
)
REGISTERED_PREDICTIONS = (
    ("pred_a_native_capability", "Every ordered native capability cell passes."),
    ("pred_b_cross_construction_transfer", "One site transfers both target constructions."),
    ("pred_c_controls_selective", "The selected site spares both registered controls."),
)

CONFIG = managed.CandidateRunConfig(
    request_id=REQUEST_ID,
    experiment_id=EXPERIMENT_ID,
    prior_art_relative=PRIOR_ART.relative_to(ROOT).as_posix(),
    result_relative=RESULT_RELATIVE.as_posix(),
    ledger_relative=LEDGER.relative_to(ROOT).as_posix(),
    expected_prior_art_sha256=EXPECTED_PRIOR_ART_SHA256,
    expected_authority_sha256=EXPECTED_AUTHORITY_SHA256,
    information_read="whether the unfinished sentence is declarative or interrogative",
    proposed_operation="carry that sentence-mode state across two syntactic constructions",
    proposed_write="evidence for a period or question mark",
    alternative_explanation=(
        "a construction-specific word cue or generic punctuation-token service"
    ),
    circuit_prediction=(
        "one site transfers both target constructions while sparing both controls"
    ),
    opposing_null_prediction=(
        "native capability fails or no site transfers both constructions selectively"
    ),
    semantic_position_role="final input token before the predicted punctuation",
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

# Compatibility aliases keep the checked-in sentence tests and callers stable.
utc_now = managed.utc_now
utc_text = managed.utc_text
literal_json = managed.literal_json
atomic_create_json = managed.atomic_create_json
selected_controls_pass = managed.selected_controls_pass


def build_spec(rows: list[dict[str, object]]) -> screen.CircuitFastScreenSpec:
    return managed.build_spec(CONFIG, candidates, rows)


def main() -> None:
    managed.run_managed(CONFIG, candidates, root=ROOT)


if __name__ == "__main__":
    main()
