#!/usr/bin/env python3
"""Prospective v10 hybrid of capability-qualified v8-A1 and v9-A2 forms."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v9 as v9

SCHEMA, SPLIT, GROUPS = v9.SCHEMA, v9.SPLIT, 16
SEED = 20261022
TASK_ID, TASK_SPEC, READOUT = v9.TASK_ID, v9.TASK_SPEC, v9.READOUT
canonical_sha256, CandidateBankError = v9.canonical_sha256, v9.CandidateBankError
_AGENTS = ("advisor", "ambassador", "artisan", "astronomer", "attorney", "biographer",
    "bookkeeper", "broadcaster", "cabinetmaker", "cartwright", "cobbler", "copywriter",
    "cryptographer", "dean", "engraver", "filmmaker")


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[group_number], _AGENTS[(group_number+5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    plain = lambda present: f"Throughout {'this' if present else 'that'} interval, the {agent}"
    embedded = lambda present: f"The memo states that during {'this' if present else 'that'} phase, the {agent}"
    paraphrase = lambda present: f"Throughout the {'present' if present else 'previous'} interval, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'different_readout_v10_hybrid', SEED, group_number])[:24]}"
    sentence_types = ("present_progressive" if base_present else "past_progressive",
                      "present_progressive" if donor_present else "past_progressive")
    common_no_vocab = dict(seed=SEED, task_id=TASK_ID, group_number=group_number,
        group_id=group_id, reporter=agent, alternate_reporter=alternate,
        adjective="different_readout_v10_hybrid", object_name="interval_phase", spec=TASK_SPEC)
    common = dict(common_no_vocab, vocabulary=READOUT)
    return [
        v9.v8.v1.builder._row(**common, transform_id="A1", construction_id="throughout_this_that_interval_auxiliary_v10",
            direction_id=direction, matched_suffix=f" interval, the {agent}", base_text=plain(base_present),
            donor_text=plain(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types),
        v9.v8.v1.builder._row(**common, transform_id="A2", construction_id="memo_during_this_that_phase_auxiliary_v10",
            direction_id=direction, matched_suffix=f" phase, the {agent}", base_text=embedded(base_present),
            donor_text=embedded(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types),
        v9.v8.v1.builder._row(**common, transform_id="P", construction_id="temporal_interval_paraphrase_same_auxiliary_v10",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f" interval, the {agent}", base_text=plain(base_present), donor_text=paraphrase(base_present),
            base_answer=answer(base_present), donor_answer=answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0])),
        v9.v8.v1.builder._row(**common_no_vocab, transform_id="C",
            **v9.v8.v1.canonical.row_kwargs(group_number, forward)),
    ]


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v10 authority")
    try:
        digest = v9.v8.v1.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except v9.v8.v1.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = (v9.v8.v1, v9.v8.v2, v9.v8.v3, v9.v8.v4, v9.v8.v5,
                   v9.v8.v6, v9.v8.v7, v9.v8, v9)
    old_banks = tuple(module.build_rows() for module in old_modules)
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old_banks for row in bank}
    ids = {str(row["row_id"]) for row in materialized}
    if (len(materialized) != 64 or len(ids) != 64 or ids & old_ids or set(_AGENTS) & old_agents
            or any(not all(row["construction_checks"].values()) for row in materialized)):
        raise CandidateBankError("v10 count, uniqueness, novelty, or construction check failed")
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
