#!/usr/bin/env python3
"""V15 targets with per-row token-aligned paraphrase and unrelated controls."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as v15
from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract


SCHEMA, SPLIT, GROUPS = v15.SCHEMA, v15.SPLIT, v15.GROUPS
SEED, TASK_ID, TASK_SPEC, READOUT = v15.SEED, v15.TASK_ID, v15.TASK_SPEC, v15.READOUT
canonical_sha256, CandidateBankError = v15.canonical_sha256, v15.CandidateBankError
_PLACES = (
    "harbor", "canyon", "valley", "garden", "forest", "island", "castle", "station",
    "market", "village", "museum", "office", "school", "river", "desert", "temple",
)
_ROLES = (
    "pilot", "sailor", "teacher", "doctor", "artist", "farmer", "writer", "driver",
    "baker", "nurse", "judge", "guard", "clerk", "guide", "cook", "miner",
)


def _controls(group_number: int, target_rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    a1 = next(row for row in target_rows if row["group_number"] == group_number and row["transform_id"] == "A1")
    a2 = next(row for row in target_rows if row["group_number"] == group_number and row["transform_id"] == "A2")
    builder = v15.v14.v13.v12.v11.v10.v9.v8.v1.builder
    common = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=str(a1["group_id"]),
        reporter=str(a1["reporter"]), alternate_reporter=str(a1["alternate_reporter"]),
        adjective="v15_aligned_control_repair", object_name="temporal_adverb", spec=TASK_SPEC,
    )
    answer = str(a1["base_answer"])
    sentence_type = "present_progressive" if answer == READOUT[0] else "past_progressive"
    p = builder._row(
        **common, transform_id="P", construction_id="cross_cue_same_tense_aligned_v1",
        direction_id="today_to_currently" if answer == READOUT[0] else "yesterday_to_previously",
        base_text=str(a1["base_text"]), donor_text=str(a2["base_text"]),
        base_answer=answer, donor_answer=answer, vocabulary=READOUT,
        matched_suffix=f", the {a1['reporter']}", sentence_types=(sentence_type, sentence_type),
    )
    left, right = group_number, (group_number + 7) % GROUPS
    suffix = " in the middle of the"
    base_text = f"Near the {_PLACES[left]} the {_ROLES[left]} finished the work{suffix}"
    donor_text = f"Near the {_PLACES[right]} the {_ROLES[right]} finished the work{suffix}"
    c = builder._row(
        **common, transform_id="C", construction_id="aligned_same_answer_nocturnal_completion_v1",
        direction_id="lexical_pair_forward", base_text=base_text, donor_text=donor_text,
        base_answer=" night", donor_answer=" night", vocabulary=(" night", " day"),
        matched_suffix=suffix, sentence_types=("canonical_control_v3", "canonical_control_v3"),
    )
    return [p, c]


def _build() -> list[dict[str, Any]]:
    old = v15.build_rows()
    targets = [row for row in old if row["transform_id"] in {"A1", "A2"}]
    controls = [row for group in range(GROUPS) for row in _controls(group, targets)]
    by_group = {(row["group_number"], row["transform_id"]): row for row in targets + controls}
    return [by_group[(group, panel)] for group in range(GROUPS) for panel in ("A1", "A2", "P", "C")]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed aligned-control authority")
    builder = v15.v14.v13.v12.v11.v10.v9.v8.v1.builder
    try:
        digest = builder.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
        contract = derive_full_sequence_alignment_contract(
            materialized, required_panels=("A1", "A2", "P", "C")
        )
    except (builder.battery.BatteryContractError, ValueError) as error:
        raise CandidateBankError(str(error)) from error
    old_targets = [row for row in v15.build_rows() if row["transform_id"] in {"A1", "A2"}]
    new_targets = [row for row in materialized if row["transform_id"] in {"A1", "A2"}]
    if (new_targets != old_targets or contract["panel_counts"] != {panel: 16 for panel in ("A1", "A2", "P", "C")}
            or len({row["row_id"] for row in materialized}) != 64):
        raise CandidateBankError("target preservation, panel count, or uniqueness failed")
    return digest


def build_rows(task_id: str = TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build()
    _validate(rows)
    return rows


def validate_rows(rows, *, task_id: str = TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return _validate(rows)


def authority_sha256(task_id: str = TASK_ID):
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    print("authority:", authority_sha256())
