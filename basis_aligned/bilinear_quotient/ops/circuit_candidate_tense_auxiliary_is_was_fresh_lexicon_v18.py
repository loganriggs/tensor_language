#!/usr/bin/env python3
"""Prospective history-disjoint v18 bank for frozen U8/H3/top-32 transfer."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17 as v17

SCHEMA, SPLIT, GROUPS = v17.SCHEMA, v17.SPLIT, 16
SEED = 20261113
TASK_ID, TASK_SPEC, READOUT = v17.TASK_ID, v17.TASK_SPEC, v17.READOUT
canonical_sha256, CandidateBankError = v17.canonical_sha256, v17.CandidateBankError
_AGENTS = (
    "aerologist", "acarologist", "arachnologist", "assayist",
    "astrobiologist", "bioinformatician", "carcinologist", "egyptologist",
    "embryologist", "geochemist", "geomorphologist", "heliophysicist",
    "lepidopterist", "oceanologist", "philologist", "physiologist",
)


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[group_number], _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    year = lambda present: f"{'This year' if present else 'Last year'}, the {agent}"
    week = lambda present: f"{'This week' if present else 'Last week'}, the {agent}"
    season = lambda present: f"{'This season' if present else 'Last season'}, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'frozen_tensor_v18', SEED, group_number])[:24]}"
    sentence_types = (
        "present_progressive" if base_present else "past_progressive",
        "present_progressive" if donor_present else "past_progressive",
    )
    common_no_vocab = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=agent, alternate_reporter=alternate, adjective="frozen_tensor_v18",
        object_name="year_week_season", spec=TASK_SPEC,
    )
    common = dict(common_no_vocab, vocabulary=READOUT)
    base = v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v1
    return [
        base.builder._row(
            **common, transform_id="A1", construction_id="this_last_year_auxiliary_v18",
            direction_id=direction, matched_suffix=f", the {agent}",
            base_text=year(base_present), donor_text=year(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="A2", construction_id="this_last_week_auxiliary_v18",
            direction_id=direction, matched_suffix=f", the {agent}",
            base_text=week(base_present), donor_text=week(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="P", construction_id="year_season_paraphrase_auxiliary_v18",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f", the {agent}", base_text=year(base_present),
            donor_text=season(base_present), base_answer=answer(base_present),
            donor_answer=answer(base_present), sentence_types=(sentence_types[0], sentence_types[0]),
        ),
        base.builder._row(
            **common_no_vocab, transform_id="C", **base.canonical.row_kwargs(group_number, forward),
        ),
    ]


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v18 authority")
    base = v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v1
    try:
        digest = base.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except base.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = (
        base, v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v2,
        v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v3,
        v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v4,
        v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v5,
        v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v6,
        v17.v16.v15.v14.v13.v12.v11.v10.v9.v8.v7,
        v17.v16.v15.v14.v13.v12.v11.v10.v9.v8,
        v17.v16.v15.v14.v13.v12.v11.v10.v9,
        v17.v16.v15.v14.v13.v12.v11.v10,
        v17.v16.v15.v14.v13.v12.v11,
        v17.v16.v15.v14.v13.v12,
        v17.v16.v15.v14.v13,
        v17.v16.v15.v14,
        v17.v16.v15,
        v17.v16,
        v17,
    )
    old_banks = tuple(
        module._build() if hasattr(module, "_build") else module.build_rows()
        for module in old_modules
    )
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old_banks for row in bank}
    old_text = {
        row[key]
        for bank in old_banks
        for row in bank
        if row["transform_id"] in {"A1", "A2", "P"}
        for key in ("base_text", "donor_text")
    }
    ids = {str(row["row_id"]) for row in materialized}
    targets = [row for row in materialized if row["transform_id"] in {"A1", "A2", "P"}]
    if (
        len(materialized) != 64
        or len(ids) != 64
        or ids & old_ids
        or set(_AGENTS) & old_agents
        or any(not all(row["construction_checks"].values()) for row in materialized)
        or any(row[key] in old_text for row in targets for key in ("base_text", "donor_text"))
    ):
        raise CandidateBankError("v18 count, novelty, or construction check failed")
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
