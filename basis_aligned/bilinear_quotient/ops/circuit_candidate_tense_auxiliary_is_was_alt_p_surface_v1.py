"""Alternative same-tense P family for the frozen donor-free is/was actuator."""
from __future__ import annotations

from typing import Mapping, Sequence

import circuit_candidate_aspectual_different_readout_is_was_v2 as parent


SCHEMA = parent.SCHEMA
SPLIT = parent.SPLIT
TASK_ID = parent.TASK_ID
TASK_SPEC = parent.TASK_SPEC
READOUT = parent.READOUT
GROUPS = parent.GROUPS
CandidateBankError = parent.CandidateBankError


def _alternative_p(old: Mapping[str, object]) -> dict:
    group_number = int(old["group_number"])
    agent = str(old["reporter"])
    base_present = old["base_answer"] == READOUT[0]
    donor_text = f"At {'this' if base_present else 'that'} exact moment the {agent}"
    details = old["semantic_details"]
    return parent.v1.builder._row(
        seed=int(old["seed"]), task_id=TASK_ID, group_number=group_number,
        group_id=str(old["group_id"]), reporter=agent,
        alternate_reporter=str(old["alternate_reporter"]),
        adjective=str(old["adjective"]), object_name=str(old["object_name"]),
        spec=TASK_SPEC, vocabulary=READOUT, transform_id="P",
        construction_id="exact_word_insertion_same_auxiliary",
        direction_id=str(old["direction_id"]), matched_suffix=f" moment the {agent}",
        base_text=str(old["base_text"]), donor_text=donor_text,
        base_answer=str(old["base_answer"]), donor_answer=str(old["base_answer"]),
        sentence_types=(str(details["base_sentence_type"]), str(details["donor_sentence_type"])),
    )


def _build() -> list[dict]:
    rows = []
    for old in parent.build_rows():
        rows.append(_alternative_p(old) if old["family"] == "P" else dict(old))
    return rows


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed alternative-P authority")
    try:
        digest = parent.v1.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except parent.v1.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    original = parent.build_rows()
    old_non_p = [row for row in original if row["family"] != "P"]
    new_non_p = [row for row in materialized if row["family"] != "P"]
    old_p_ids = {str(row["row_id"]) for row in original if row["family"] == "P"}
    new_p = [row for row in materialized if row["family"] == "P"]
    if (
        len(materialized) != 64
        or len(new_p) != 16
        or old_non_p != new_non_p
        or old_p_ids & {str(row["row_id"]) for row in new_p}
        or len({str(row["row_id"]) for row in materialized}) != 64
        or any(not all(row["construction_checks"].values()) for row in materialized)
    ):
        raise CandidateBankError("count, non-P identity, P novelty, uniqueness, or construction check failed")
    return digest


def build_rows(task_id: str = TASK_ID) -> list[dict]:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build()
    _validate(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID) -> str:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return _validate(rows)


def authority_sha256(task_id: str = TASK_ID) -> str:
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    sample = next(row for row in rows if row["family"] == "P")
    print(sample["base_text"], "->", sample["donor_text"], "answer", sample["base_answer"])
