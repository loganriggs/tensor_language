#!/usr/bin/env python3
"""Prospective v13 construction/lexicon bank for five-MLP confirmation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as v12

SCHEMA, SPLIT, GROUPS = v12.SCHEMA, v12.SPLIT, 16
SEED = 20261108
TASK_ID, TASK_SPEC, READOUT = v12.TASK_ID, v12.TASK_SPEC, v12.READOUT
canonical_sha256, CandidateBankError = v12.canonical_sha256, v12.CandidateBankError
_AGENTS = (
    "acoustician", "anesthetist", "aquarist", "bookbinder", "cinematographer", "dietitian",
    "endocrinologist", "gemologist", "immunologist", "lexicographer", "midwife", "oceanographer",
    "pathologist", "radiologist", "stenographer", "surveyor",
)


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[group_number], _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    plain = lambda present: f"At {'this' if present else 'that'} moment, the {agent}"
    embedded = lambda present: (
        f"The dispatch confirms that in {'this' if present else 'that'} period, the {agent}"
    )
    paraphrase = lambda present: f"At the {'present' if present else 'previous'} moment, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'five_mlp_v13_confirmation', SEED, group_number])[:24]}"
    sentence_types = (
        "present_progressive" if base_present else "past_progressive",
        "present_progressive" if donor_present else "past_progressive",
    )
    common_no_vocab = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=agent, alternate_reporter=alternate, adjective="five_mlp_v13_confirmation",
        object_name="moment_period", spec=TASK_SPEC,
    )
    common = dict(common_no_vocab, vocabulary=READOUT)
    builder = v12.v11.v10.v9.v8.v1.builder
    canonical = v12.v11.v10.v9.v8.v1.canonical
    return [
        builder._row(
            **common, transform_id="A1", construction_id="at_this_that_moment_auxiliary_v13",
            direction_id=direction, matched_suffix=f" moment, the {agent}",
            base_text=plain(base_present), donor_text=plain(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        builder._row(
            **common, transform_id="A2", construction_id="dispatch_in_this_that_period_auxiliary_v13",
            direction_id=direction, matched_suffix=f" period, the {agent}",
            base_text=embedded(base_present), donor_text=embedded(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        builder._row(
            **common, transform_id="P", construction_id="at_temporal_moment_paraphrase_auxiliary_v13",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f" moment, the {agent}", base_text=plain(base_present),
            donor_text=paraphrase(base_present), base_answer=answer(base_present),
            donor_answer=answer(base_present), sentence_types=(sentence_types[0], sentence_types[0]),
        ),
        builder._row(
            **common_no_vocab, transform_id="C",
            **canonical.row_kwargs(group_number, forward),
        ),
    ]


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v13 authority")
    base = v12.v11.v10.v9.v8.v1
    try:
        digest = base.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except base.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = (
        base, v12.v11.v10.v9.v8.v2, v12.v11.v10.v9.v8.v3, v12.v11.v10.v9.v8.v4,
        v12.v11.v10.v9.v8.v5, v12.v11.v10.v9.v8.v6, v12.v11.v10.v9.v8.v7,
        v12.v11.v10.v9.v8, v12.v11.v10.v9, v12.v11.v10, v12.v11, v12,
    )
    old_banks = tuple(module.build_rows() for module in old_modules)
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old_banks for row in bank}
    ids = {str(row["row_id"]) for row in materialized}
    if (len(materialized) != 64 or len(ids) != 64 or ids & old_ids or set(_AGENTS) & old_agents
            or any(not all(row["construction_checks"].values()) for row in materialized)):
        raise CandidateBankError("v13 count, uniqueness, novelty, or construction check failed")
    old_text = {row[key] for row in v12.build_rows() if row["transform_id"] in {"A1", "A2"}
                for key in ("base_text", "donor_text")}
    targets = [row for row in materialized if row["transform_id"] in {"A1", "A2"}]
    if any(row[key] in old_text for row in targets for key in ("base_text", "donor_text")):
        raise CandidateBankError("v13 target text overlaps v12")
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
