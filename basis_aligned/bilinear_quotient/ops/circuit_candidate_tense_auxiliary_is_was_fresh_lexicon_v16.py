#!/usr/bin/env python3
"""Prospective cue-strong v16 bank for a third is/was construction environment."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as v15

SCHEMA, SPLIT, GROUPS = v15.SCHEMA, v15.SPLIT, 16
SEED = 20261111
TASK_ID, TASK_SPEC, READOUT = v15.TASK_ID, v15.TASK_SPEC, v15.READOUT
canonical_sha256, CandidateBankError = v15.canonical_sha256, v15.CandidateBankError
_AGENTS = (
    "biophysicist", "glaciologist", "horologist", "lexicologist", "limnologist", "luthier",
    "musicologist", "papyrologist", "phycologist", "physiotherapist", "primatologist",
    "speleologist", "taxidermist", "urbanist", "vexillologist", "volcanologist",
)


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[group_number], _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    immediate = lambda present: f"{'Right now' if present else 'Back then'}, the {agent}"
    days = lambda present: f"In {'these' if present else 'those'} days, the {agent}"
    paraphrase = lambda present: f"{'Just now' if present else 'Long ago'}, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'multi_environment_v16', SEED, group_number])[:24]}"
    sentence_types = (
        "present_progressive" if base_present else "past_progressive",
        "present_progressive" if donor_present else "past_progressive",
    )
    common_no_vocab = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=agent, alternate_reporter=alternate, adjective="multi_environment_v16",
        object_name="immediate_days", spec=TASK_SPEC,
    )
    common = dict(common_no_vocab, vocabulary=READOUT)
    base = v15.v14.v13.v12.v11.v10.v9.v8.v1
    return [
        base.builder._row(
            **common, transform_id="A1", construction_id="right_now_back_then_auxiliary_v16",
            direction_id=direction, matched_suffix=f", the {agent}",
            base_text=immediate(base_present), donor_text=immediate(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="A2", construction_id="these_those_days_auxiliary_v16",
            direction_id=direction, matched_suffix=f" days, the {agent}",
            base_text=days(base_present), donor_text=days(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="P", construction_id="immediate_temporal_paraphrase_auxiliary_v16",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f", the {agent}", base_text=immediate(base_present),
            donor_text=paraphrase(base_present), base_answer=answer(base_present),
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
        raise CandidateBankError("rows differ from sealed v16 authority")
    base = v15.v14.v13.v12.v11.v10.v9.v8.v1
    try:
        digest = base.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except base.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = (
        base, v15.v14.v13.v12.v11.v10.v9.v8.v2, v15.v14.v13.v12.v11.v10.v9.v8.v3,
        v15.v14.v13.v12.v11.v10.v9.v8.v4, v15.v14.v13.v12.v11.v10.v9.v8.v5,
        v15.v14.v13.v12.v11.v10.v9.v8.v6, v15.v14.v13.v12.v11.v10.v9.v8.v7,
        v15.v14.v13.v12.v11.v10.v9.v8, v15.v14.v13.v12.v11.v10.v9,
        v15.v14.v13.v12.v11.v10, v15.v14.v13.v12.v11, v15.v14.v13.v12,
        v15.v14.v13, v15.v14, v15,
    )
    old_banks = tuple(module._build() if hasattr(module, "_build") else module.build_rows()
                      for module in old_modules)
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old_banks for row in bank}
    ids = {str(row["row_id"]) for row in materialized}
    target_text = {row[key] for bank in old_banks for row in bank
                   if row["transform_id"] in {"A1", "A2", "P"}
                   for key in ("base_text", "donor_text")}
    targets = [row for row in materialized if row["transform_id"] in {"A1", "A2", "P"}]
    if (len(materialized) != 64 or len(ids) != 64 or ids & old_ids or set(_AGENTS) & old_agents
            or any(not all(row["construction_checks"].values()) for row in materialized)
            or any(row[key] in target_text for row in targets for key in ("base_text", "donor_text"))):
        raise CandidateBankError("v16 count, novelty, or construction check failed")
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
