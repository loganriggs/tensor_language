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
    request_id="task14-subject-verb-agreement-select-cross-noun-managed-replication-v3",
    experiment_id="fast-screen-task14-subject-verb-agreement-select-cross-noun-managed-replication-v3",
    result_relative=Path(
        "circuits/fast_screens/task14_subject_verb_agreement_select_cross_noun_managed_replication_v3_result.json"
    ),
    prior_art_sha256="c34890a24dbc5c640612257fd2880386a13bb6e81d15af9609b1ddf8ddce2364",
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
        "Process-compliant managed replication of the frozen SELECT cross-noun profile after "
        "the direct v1 and incomplete-publication v2 runs were excluded from canonical evidence."
    ),
    relation="replication",
    checkpoint_sha256="680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    config_sha256="428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
)


if __name__ == "__main__":
    shared.cli(PROTOCOL)
