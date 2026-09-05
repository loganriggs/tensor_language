#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the targeted shared runner.
"""Run the Task14 SELECT cross-syntax confirmation through the shared engine."""

from pathlib import Path

import circuit_fast_screen_candidate_task14_select_cross_syntax as candidate
import run_circuit_fast_screen_task14_cross_syntax as shared


REGISTERED_PREDICTIONS = (
    "pred_a_native_capability",
    "pred_b_attention11_cross_syntax",
    "pred_c_head11_3_cross_syntax",
)


PROTOCOL = shared.TargetedCrossSyntaxProtocol(
    candidate=candidate,
    request_id="task14-subject-verb-agreement-select-cross-syntax-v1",
    experiment_id="fast-screen-task14-subject-verb-agreement-select-cross-syntax-v1",
    result_relative=Path(
        "circuits/fast_screens/task14_subject_verb_agreement_select_cross_syntax_v1_result.json"
    ),
    prior_art_sha256="544aa139689b11e9d8397794964ea0bfc82cadfb88e4dd522610d825e0c75f84",
    expected_authority_sha256=(
        "ecaae3b5e7baddcc3e9d7b888133ad78f8f6185656bbb439a044248bd58157c1"
    ),
    result_schema="task14_targeted_cross_syntax_result_v1",
    phase="SELECT",
    partition="HELD_OUT",
    validation_scope="unseen_nouns_and_prompt_templates_after_fit_site_selection",
    expected_cell_count=16,
    limits=(
        "This is held-out cross-syntax transfer at two FIT-preselected sites. "
        "It has no unrelated endpoint control, so it does not establish held-out selectivity. "
        "The historical FIT selection results did not record checkpoint hashes; this run binds "
        "the current canonical checkpoint but cannot retroactively prove same-checkpoint continuity."
    ),
    novelty=(
        "Test attention 11 and head 11.3 on the frozen SELECT noun and prompt-template pool, "
        "which is disjoint from FIT. Each relation holds its noun group and attractor number "
        "fixed while swapping between PP and relative-clause syntax and reversing subject number."
    ),
    checkpoint_sha256="680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    config_sha256="428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
)


if __name__ == "__main__":
    shared.cli(PROTOCOL)
