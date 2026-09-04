#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the managed harness.
"""Task 14 rerun after fixing conditional attention-head expansion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import circuit_fast_screen_candidate_task14_agreement as candidate
import circuit_fast_screen_managed_runner as managed
import run_circuit_fast_screen_task14_agreement as v1


ROOT = Path(__file__).resolve().parent.parent
RESULT_RELATIVE = Path(
    "circuits/fast_screens/task14_subject_verb_agreement_full_state_v2_result.json"
)
CONFIG = replace(
    v1.CONFIG,
    request_id="task14-subject-verb-agreement-full-state-v2",
    experiment_id="fast-screen-task14-subject-verb-agreement-full-state-v2",
    result_relative=RESULT_RELATIVE.as_posix(),
)
RESULT = ROOT / RESULT_RELATIVE
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


def main() -> None:
    managed.run_managed(CONFIG, candidate, root=ROOT)


if __name__ == "__main__":
    main()
