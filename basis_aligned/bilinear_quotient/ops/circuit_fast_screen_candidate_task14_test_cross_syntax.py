#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen TEST facade for the outcome-blind Task14 cross-syntax factory."""

from circuit_fast_screen_candidate_task14_phase_cross_syntax import (
    PhaseCrossSyntaxAuthorityError,
    PhaseCrossSyntaxConfig,
    make_candidate,
)


_CANDIDATE = make_candidate(PhaseCrossSyntaxConfig(
    phase="TEST",
    schema="task14_test_cross_syntax_authority_v1",
    validation_scope="unseen_nouns_and_prompt_templates_after_fit_and_select",
    expected_phase_records_sha256=(
        "d62dae278f66ae5a2e77aadf8b841fe9aecf4bf2fa7bb9378b8d59e9f5829b27"
    ),
    correction=(
        "FIT selected the sites and SELECT confirmed them; TEST uses a third "
        "disjoint noun and prompt-template pool without reading phase outcomes."
    ),
    donor_rule="cyclic_cross_noun_by_subject_and_attractor",
    site_ids=("attn:11:head:03",),
))

TASK_ID = _CANDIDATE.TASK_ID
PHASE = _CANDIDATE.PHASE
PARTITION = _CANDIDATE.PARTITION
VALIDATION_SCOPE = _CANDIDATE.VALIDATION_SCOPE
SCHEMA = _CANDIDATE.SCHEMA
SITE_IDS = _CANDIDATE.SITE_IDS
BATCH_SIZE = _CANDIDATE.BATCH_SIZE
MIN_NATIVE_CELL_ACCURACY = _CANDIDATE.MIN_NATIVE_CELL_ACCURACY
MIN_CELL_DIRECTION_FRACTION = _CANDIDATE.MIN_CELL_DIRECTION_FRACTION
MIN_CELL_MEAN_RECOVERY = _CANDIDATE.MIN_CELL_MEAN_RECOVERY
MIN_DONOR_DENOMINATOR = _CANDIDATE.MIN_DONOR_DENOMINATOR
EXPECTED_SOURCE_SHA256 = _CANDIDATE.EXPECTED_SOURCE_SHA256

build_rows = _CANDIDATE.build_rows
validate_rows = _CANDIDATE.validate_rows
authority_sha256 = _CANDIDATE.authority_sha256
compile_plan = _CANDIDATE.compile_plan


if __name__ == "__main__":
    import json
    print(json.dumps(compile_plan(), sort_keys=True))
