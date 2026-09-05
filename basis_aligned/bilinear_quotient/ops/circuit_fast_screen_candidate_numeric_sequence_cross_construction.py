"""Frozen numeric-sequence screen using a related numbered-list C hypothesis."""
# BQGATE: ANALYSIS  pred_a_rows_build_and_validate

from __future__ import annotations

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_numeric_sequence_shared as shared


CONFIG = shared.NumericSequenceCandidateConfig(
    task_id="numeric_sequence.cross_construction_sufficiency",
    prior_art_name="fast_screen_numeric_sequence_cross_construction_prior_art.json",
    family_map={
        "sequence_digit_state_shift": ("A1", "digit_format"),
        "sequence_word_state_shift": ("A2", "word_format"),
        "sequence_digit_surface_preserved": ("P", "digit_surface_preserved"),
        "list_two_line_state_shift": ("C", "numbered_list_control"),
    },
    transforms=(
        battery.TransformSpec("A1", "digit_sequence_state_shift", True, "toward_donor"),
        battery.TransformSpec("A2", "word_sequence_state_shift", True, "toward_donor"),
        battery.TransformSpec("P", "digit_surface_preserved_rewrite", False, "invariant"),
        battery.TransformSpec("C", "numbered_list_active_control", True, "registered_active"),
    ),
)

ROOT = shared.ROOT
ROWS_PATH = shared.ROWS_PATH
ROWS_SHA256 = shared.ROWS_SHA256
PRIOR_ART = CONFIG.prior_art
SCHEMA = shared.SCHEMA
TASK_ID = CONFIG.task_id
HYPOTHESIS = shared.HYPOTHESIS
DEFAULT_GROUPS = shared.DEFAULT_GROUPS
DEFAULT_SEED = shared.DEFAULT_SEED
SPLIT = shared.SPLIT
ENCODING = shared.ENCODING
FAMILY_MAP = dict(CONFIG.family_map)
TASK_SPEC = CONFIG.task_spec
NumberedListSufficiencyError = shared.NumericSequenceCandidateError


def build_rows(task_id=TASK_ID, groups=DEFAULT_GROUPS, seed=DEFAULT_SEED):
    return shared.build_rows(CONFIG, task_id, groups, seed)


def validate_rows(rows, *, task_id=TASK_ID, groups=DEFAULT_GROUPS, seed=DEFAULT_SEED):
    return shared.validate_rows(CONFIG, rows, task_id=task_id, groups=groups, seed=seed)


def authority_sha256(task_id=TASK_ID, groups=DEFAULT_GROUPS, seed=DEFAULT_SEED):
    return shared.authority_sha256(CONFIG, task_id, groups, seed)


if __name__ == "__main__":
    print(f"{len(build_rows())} rows, authority {authority_sha256()[:16]}")
