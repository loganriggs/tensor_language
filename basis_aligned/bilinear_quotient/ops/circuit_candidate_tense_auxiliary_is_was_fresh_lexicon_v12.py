#!/usr/bin/env python3
"""Prospective v12 lexical bank for frozen post-cue source confirmation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as v11

SCHEMA, SPLIT, GROUPS = v11.SCHEMA, v11.SPLIT, 16
SEED = 20261104
TASK_ID, TASK_SPEC, READOUT = v11.TASK_ID, v11.TASK_SPEC, v11.READOUT
canonical_sha256, CandidateBankError = v11.canonical_sha256, v11.CandidateBankError
_AGENTS = ("allergist", "animator", "arborist", "audiologist", "boatbuilder", "coroner",
    "cytologist", "demographer", "entomologist", "geneticist", "hydrologist", "metallurgist",
    "meteorologist", "neurologist", "oncologist", "orthopedist")


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[group_number], _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    plain = lambda present: f"Accordingly, throughout {'this' if present else 'that'} interval, the {agent}"
    embedded = lambda present: f"The bulletin notes that during {'this' if present else 'that'} phase, the {agent}"
    paraphrase = lambda present: f"Accordingly, throughout the {'present' if present else 'previous'} interval, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'different_readout_v12_source_transfer', SEED, group_number])[:24]}"
    sentence_types = ("present_progressive" if base_present else "past_progressive",
                      "present_progressive" if donor_present else "past_progressive")
    common_no_vocab = dict(seed=SEED, task_id=TASK_ID, group_number=group_number,
        group_id=group_id, reporter=agent, alternate_reporter=alternate,
        adjective="different_readout_v12_source_transfer", object_name="interval_phase", spec=TASK_SPEC)
    common = dict(common_no_vocab, vocabulary=READOUT)
    return [
        v11.v10.v9.v8.v1.builder._row(**common, transform_id="A1",
            construction_id="accordingly_throughout_this_that_interval_auxiliary_v12",
            direction_id=direction, matched_suffix=f" interval, the {agent}",
            base_text=plain(base_present), donor_text=plain(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
        v11.v10.v9.v8.v1.builder._row(**common, transform_id="A2",
            construction_id="bulletin_during_this_that_phase_auxiliary_v12",
            direction_id=direction, matched_suffix=f" phase, the {agent}",
            base_text=embedded(base_present), donor_text=embedded(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
        v11.v10.v9.v8.v1.builder._row(**common, transform_id="P",
            construction_id="accordingly_temporal_interval_paraphrase_auxiliary_v12",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f" interval, the {agent}", base_text=plain(base_present), donor_text=paraphrase(base_present),
            base_answer=answer(base_present), donor_answer=answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0])),
        v11.v10.v9.v8.v1.builder._row(**common_no_vocab, transform_id="C",
            **v11.v10.v9.v8.v1.canonical.row_kwargs(group_number, forward)),
    ]


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v12 authority")
    try:
        digest = v11.v10.v9.v8.v1.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except v11.v10.v9.v8.v1.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = (v11.v10.v9.v8.v1, v11.v10.v9.v8.v2, v11.v10.v9.v8.v3,
                   v11.v10.v9.v8.v4, v11.v10.v9.v8.v5, v11.v10.v9.v8.v6,
                   v11.v10.v9.v8.v7, v11.v10.v9.v8, v11.v10.v9, v11.v10, v11)
    old_banks = tuple(module.build_rows() for module in old_modules)
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old_banks for row in bank}
    ids = {str(row["row_id"]) for row in materialized}
    if (len(materialized) != 64 or len(ids) != 64 or ids & old_ids or set(_AGENTS) & old_agents
            or any(not all(row["construction_checks"].values()) for row in materialized)):
        raise CandidateBankError("v12 count, uniqueness, novelty, or construction check failed")
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
