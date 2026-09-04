#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Thin Task 14 wrapper for the generic managed FIT causal screen."""

from __future__ import annotations

from pathlib import Path

import circuit_fast_screen_candidate_task14_agreement as candidate
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_task14_agreement_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/task14_subject_verb_agreement_full_state_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "task14-subject-verb-agreement-full-state-v1"
EXPERIMENT_ID = "fast-screen-task14-subject-verb-agreement-full-state-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "9b8ede7d17b0358467438b7f8fda7703bba1c93c9c594d55454404c1bb6e21cc"
)
EXPECTED_PRIOR_ART_SHA256 = (
    "fa6a1c53136601d527c9efa2c667fc70b624e8f7b8ce3544bf19342615af649a"
)
REGISTERED_PREDICTIONS = (
    ("pred_a_native_capability", "Every ordered native capability cell passes."),
    (
        "pred_b_cross_construction_transfer",
        "One exact state site transfers both agreement constructions.",
    ),
    (
        "pred_c_controls_selective",
        "The selected site spares noun-identity and attractor-number controls.",
    ),
)

CONFIG = managed.CandidateRunConfig(
    request_id=REQUEST_ID,
    experiment_id=EXPERIMENT_ID,
    prior_art_relative=PRIOR_ART.relative_to(ROOT).as_posix(),
    result_relative=RESULT_RELATIVE.as_posix(),
    ledger_relative=LEDGER.relative_to(ROOT).as_posix(),
    expected_prior_art_sha256=EXPECTED_PRIOR_ART_SHA256,
    expected_authority_sha256=EXPECTED_AUTHORITY_SHA256,
    information_read="complete grammatical subject number rather than nearest-noun number",
    proposed_operation=(
        "carry subject number across prepositional phrases and relative clauses"
    ),
    proposed_write="evidence for the next-token choice between is and are",
    alternative_explanation=(
        "nearest-noun number, construction-specific state, or generic output state"
    ),
    circuit_prediction=(
        "one site transfers both answer-changing subject-number families while "
        "remaining small for noun-identity and attractor-number controls"
    ),
    opposing_null_prediction=(
        "native capability fails or no common selective full-state site exists"
    ),
    semantic_position_role="final input token immediately before the predicted verb",
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


def build_spec(rows: list[dict[str, object]]) -> screen.CircuitFastScreenSpec:
    return managed.build_spec(CONFIG, candidate, rows)


def main() -> None:
    managed.run_managed(CONFIG, candidate, root=ROOT)


if __name__ == "__main__":
    main()
