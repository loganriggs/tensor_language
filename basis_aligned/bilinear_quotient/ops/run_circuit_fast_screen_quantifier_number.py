#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin quantifier-number wrapper for the generic managed FIT screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_quantifier_number as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_quantifier_number_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/quantifier_number_each_vs_all_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "quantifier-number-each-vs-all-v1"
EXPERIMENT_ID = "fast-screen-quantifier-number-each-vs-all-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "bd4567a2e62fcd93c19ad69d7ffeea3c9df8b8c88c73fae366c8477c2d233236"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "6aaddc44f6e6bb25322079905529b96758b5a9a9cdd2a1b2c0d06b1b89734dc8"
)
REGISTERED_PREDICTIONS = (
    ("pred_a_native_capability", "Every ordered native capability cell passes."),
    ("pred_b_cross_construction_transfer", "One site transfers both quantifier constructions."),
    ("pred_c_controls_selective", "The selected site spares the agent rewrite and the endpoint control."),
)

CONFIG = managed.CandidateRunConfig(
    request_id=REQUEST_ID,
    experiment_id=EXPERIMENT_ID,
    prior_art_relative=PRIOR_ART.relative_to(ROOT).as_posix(),
    result_relative=RESULT_RELATIVE.as_posix(),
    ledger_relative=LEDGER.relative_to(ROOT).as_posix(),
    expected_prior_art_sha256=EXPECTED_PRIOR_ART_SHA256,
    expected_authority_sha256=EXPECTED_AUTHORITY_SHA256,
    information_read="which number the quantifier imposes",
    proposed_operation="carry the quantifier-imposed number across the plural noun in two constructions",
    proposed_write="evidence for were rather than was",
    alternative_explanation=(
        "the plural noun local cue rather than the quantifier-imposed number"
    ),
    circuit_prediction=(
        "one site transfers the were/was decision in both constructions while sparing the agent rewrite"
    ),
    opposing_null_prediction=(
        "native capability fails, or no site transfers both constructions"
    ),
    semantic_position_role="final input token before the auxiliary",
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
