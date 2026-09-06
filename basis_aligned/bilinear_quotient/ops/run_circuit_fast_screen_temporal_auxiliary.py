#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin temporal-auxiliary wrapper for the generic managed FIT screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_temporal_auxiliary_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "temporal-auxiliary-will-vs-had-v1"
EXPERIMENT_ID = "fast-screen-temporal-auxiliary-will-vs-had-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "e19829f0e214f72892ee0307c6a2a123cc4ad3f1e932bd75f2ed036ecb53ab49"
)
REGISTERED_PREDICTIONS = (
    ("pred_a_native_capability", "Every ordered native capability cell passes."),
    ("pred_b_cross_construction_transfer", "One site transfers both temporal constructions."),
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
    information_read="which temporal frame the fronted adverb sets",
    proposed_operation="carry the temporal frame to the auxiliary in two constructions",
    proposed_write="evidence for will rather than had",
    alternative_explanation=(
        "a local cue at the prediction site rather than a carried temporal frame"
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
