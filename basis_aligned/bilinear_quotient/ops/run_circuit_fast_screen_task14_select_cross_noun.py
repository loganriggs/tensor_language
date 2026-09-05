#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by the targeted shared runner.
"""Run the Task14 SELECT cross-noun donor profile through the shared engine."""

from pathlib import Path

import circuit_fast_screen_candidate_task14_select_cross_noun as candidate
import run_circuit_fast_screen_task14_cross_syntax as shared


REGISTERED_PREDICTIONS = (
    "pred_a_native_capability",
    "pred_b_attention11_cross_syntax",
    "pred_c_head11_3_cross_syntax",
)


PROTOCOL = shared.TargetedCrossSyntaxProtocol(
    candidate=candidate,
    request_id="task14-subject-verb-agreement-select-cross-noun-managed-v2",
    experiment_id="fast-screen-task14-subject-verb-agreement-select-cross-noun-managed-v2",
    result_relative=Path(
        "circuits/fast_screens/task14_subject_verb_agreement_select_cross_noun_managed_v2_result.json"
    ),
    prior_art_sha256="80d1cb2cfc53e41162384c4a0dc4caa8385997aff617bfe28fb81028e75ec7f8",
    expected_authority_sha256=(
        "9d5151f9e297788c0c8799cc60cc4c9bf1e6196e10df93793fb53094566091ae"
    ),
    result_schema="task14_targeted_cross_syntax_result_v1",
    phase="SELECT",
    partition="HELD_OUT",
    validation_scope=(
        "unseen_nouns_templates_and_cross_noun_donors_after_fit_site_selection"
    ),
    expected_cell_count=16,
    limits=(
        "This changes only the donor-matching rule relative to the matched-noun SELECT run. "
        "It tests counterfactual robustness at two fixed sites and has no unrelated endpoint "
        "control, so it does not establish held-out selectivity."
    ),
    novelty=(
        "Within each SELECT target-subject-number and attractor-plurality stratum, use a "
        "deterministic different-noun donor while retaining the A1/A2 syntax swap and fixed sites."
    ),
    checkpoint_sha256="680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    config_sha256="428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
)


if __name__ == "__main__":
    shared.cli(PROTOCOL)
