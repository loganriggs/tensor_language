#!/usr/bin/env python3
"""Prospective ninth is/was lexicon for complement-head union confirmation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v8 as v8

SCHEMA, SPLIT, GROUPS = v8.SCHEMA, v8.SPLIT, 16
SEED = 20261021
TASK_ID, TASK_SPEC, READOUT = v8.TASK_ID, v8.TASK_SPEC, v8.READOUT
canonical_sha256, CandidateBankError = v8.canonical_sha256, v8.CandidateBankError
_AGENTS = ("analyst", "announcer", "archivist", "auctioneer", "brewer", "butler",
    "caretaker", "choreographer", "consultant", "decorator", "diplomat", "dispatcher",
    "draftsman", "economist", "entrepreneur", "executive")


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent = _AGENTS[group_number]
    alternate = _AGENTS[(group_number+5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    plain = lambda present: f"During {'this' if present else 'that'} phase, the {agent}"
    embedded = lambda present: f"The memo states that during {'this' if present else 'that'} phase, the {agent}"
    paraphrase = lambda present: f"During the {'present' if present else 'previous'} phase, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'different_readout_v9', SEED, group_number])[:24]}"
    sentence_types = ("present_progressive" if base_present else "past_progressive",
                      "present_progressive" if donor_present else "past_progressive")
    common_no_vocab = dict(seed=SEED, task_id=TASK_ID, group_number=group_number,
        group_id=group_id, reporter=agent, alternate_reporter=alternate,
        adjective="different_readout_v9", object_name="phase", spec=TASK_SPEC)
    common = dict(common_no_vocab, vocabulary=READOUT)
    suffix = f" phase, the {agent}"
    return [
        v8.v1.builder._row(**common, transform_id="A1", construction_id="during_this_that_phase_auxiliary",
            direction_id=direction, matched_suffix=suffix, base_text=plain(base_present),
            donor_text=plain(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types),
        v8.v1.builder._row(**common, transform_id="A2", construction_id="memo_embedded_during_this_that_phase_auxiliary",
            direction_id=direction, matched_suffix=suffix, base_text=embedded(base_present),
            donor_text=embedded(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types),
        v8.v1.builder._row(**common, transform_id="P", construction_id="temporal_phase_paraphrase_same_auxiliary",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f" phase, the {agent}", base_text=plain(base_present), donor_text=paraphrase(base_present),
            base_answer=answer(base_present), donor_answer=answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0])),
        v8.v1.builder._row(**common_no_vocab, transform_id="C",
            **v8.v1.canonical.row_kwargs(group_number, forward)),
    ]


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v9 authority")
    try:
        digest = v8.v1.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except v8.v1.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = (v8.v1, v8.v2, v8.v3, v8.v4, v8.v5, v8.v6, v8.v7, v8)
    old_banks = tuple(module.build_rows() for module in old_modules)
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old_banks for row in bank}
    ids = {str(row["row_id"]) for row in materialized}
    if (len(materialized) != 64 or len(ids) != 64 or ids & old_ids or set(_AGENTS) & old_agents
            or any(not all(row["construction_checks"].values()) for row in materialized)):
        raise CandidateBankError("v9 count, uniqueness, novelty, or construction check failed")
    return digest


def build_rows(task_id: str = TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build(); _validate(rows); return rows


def validate_rows(rows, *, task_id: str = TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return _validate(rows)


def authority_sha256(task_id: str = TASK_ID):
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    print("authority:", authority_sha256())
