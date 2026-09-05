#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen unopened OOD facade for the outcome-blind Task14 phase factory."""

from circuit_fast_screen_candidate_task14_phase_cross_syntax import (
    PhaseCrossSyntaxAuthorityError,
    PhaseCrossSyntaxConfig,
    make_candidate,
)


_CANDIDATE = make_candidate(PhaseCrossSyntaxConfig(
    phase="OOD",
    schema="task14_ood_cross_syntax_authority_v1",
    validation_scope="fronted_or_two_attractor_syntax_after_test_gate",
    expected_phase_records_sha256=(
        "f2e4a6fc68be3ff8a87efde056780996106b9fb10a532381588d3d47d9da40b6"
    ),
    correction=(
        "OOD is compiled but unopened; it adds fronting or two attractors and "
        "must not be executed until the separately frozen TEST gate permits it."
    ),
    donor_rule="cyclic_cross_noun_by_subject_and_two_attractors",
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
