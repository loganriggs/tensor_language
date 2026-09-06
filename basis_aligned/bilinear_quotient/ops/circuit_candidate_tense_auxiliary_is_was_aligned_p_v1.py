#!/usr/bin/env python3
"""Alignment-preserving same-tense moment/instant P authority for is/was."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_candidate_aspectual_different_readout_is_was_v1 as v1
import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as matched


SEED, GROUPS = 20261019, 16
TASK_ID, TASK_SPEC, READOUT = v1.TASK_ID, v1.TASK_SPEC, v1.READOUT
_AGENTS = matched._AGENTS


class AlignedPBankError(RuntimeError):
    pass


def _row(group_number: int) -> dict[str, Any]:
    agent = _AGENTS[group_number]
    alternate = _AGENTS[(group_number + 5) % GROUPS]
    present = group_number % 2 == 0
    cue = "this" if present else "that"
    answer = READOUT[0] if present else READOUT[1]
    group_id = f"FIT:{v1.canonical_sha256([v1.SCHEMA, TASK_ID, 'aligned_p_moment_instant_v1', SEED, group_number])[:24]}"
    sentence_type = "present_progressive" if present else "past_progressive"
    common = dict(seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id, reporter=agent, alternate_reporter=alternate, adjective="aligned_p_moment_instant_v1", object_name="moment", spec=TASK_SPEC, vocabulary=READOUT)
    return v1.builder._row(
        **common, transform_id="P", construction_id="aligned_same_tense_moment_to_instant",
        direction_id="primary_to_alternative" if present else "alternative_to_primary",
        matched_suffix=f" the {agent}", base_text=f"At {cue} moment the {agent}", donor_text=f"At {cue} instant the {agent}",
        base_answer=answer, donor_answer=answer, sentence_types=(sentence_type, sentence_type),
    )


def build_rows(task_id: str = TASK_ID) -> list[dict[str, Any]]:
    if task_id != TASK_ID:
        raise AlignedPBankError("task ID changed")
    return [_row(group_number) for group_number in range(GROUPS)]


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID) -> str:
    materialized = [dict(row) for row in rows]
    if task_id != TASK_ID or materialized != build_rows(task_id):
        raise AlignedPBankError("rows differ from sealed aligned P authority")
    ids = {str(row["row_id"]) for row in materialized}
    prior_ids = {str(row["row_id"]) for row in matched.build_rows_by_bank()["is_was"]}
    structural = all(
        row["family"] == "P" and row["base_answer_id"] == row["donor_answer_id"]
        and len(row["base_ids"]) == len(row["donor_ids"])
        and sum(base != donor for base, donor in zip(row["base_ids"], row["donor_ids"])) == 1
        and all(row["construction_checks"].values())
        for row in materialized
    )
    if len(materialized) != 16 or len(ids) != 16 or ids & prior_ids or not structural:
        raise AlignedPBankError("count, uniqueness, novelty, alignment, answer identity, or construction check failed")
    return v1.canonical_sha256(materialized)


def authority_sha256(task_id: str = TASK_ID) -> str:
    rows = build_rows(task_id)
    return validate_rows(rows, task_id=task_id)


if __name__ == "__main__":
    print("authority:", authority_sha256())
