#!/usr/bin/env python3
"""Prospective history-disjoint v17 bank for sealed A11/M11 edge tests."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as v16

SCHEMA, SPLIT, GROUPS = v16.SCHEMA, v16.SPLIT, 16
SEED = 20261112
TASK_ID, TASK_SPEC, READOUT = v16.TASK_ID, v16.TASK_SPEC, v16.READOUT
canonical_sha256, CandidateBankError = v16.canonical_sha256, v16.CandidateBankError
_AGENTS = (
    "campanologist", "codicologist", "diplomatist", "herpetologist", "hydrographer",
    "mammalogist", "malacologist", "museologist", "nephrologist", "odontologist",
    "onomastician", "osteologist", "palynologist", "petrologist", "toxicologist",
    "zoogeographer",
)


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[group_number], _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    present_past = lambda present: f"{'At present' if present else 'In the past'}, the {agent}"
    now_then = lambda present: f"{'As of now' if present else 'At that time'}, the {agent}"
    paraphrase = lambda present: f"{'At this moment' if present else 'In earlier times'}, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'sealed_edge_v17', SEED, group_number])[:24]}"
    sentence_types = (
        "present_progressive" if base_present else "past_progressive",
        "present_progressive" if donor_present else "past_progressive",
    )
    common_no_vocab = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=agent, alternate_reporter=alternate, adjective="sealed_edge_v17",
        object_name="present_past_now_then", spec=TASK_SPEC,
    )
    common = dict(common_no_vocab, vocabulary=READOUT)
    base = v16.v15.v14.v13.v12.v11.v10.v9.v8.v1
    return [
        base.builder._row(
            **common, transform_id="A1", construction_id="at_present_in_past_auxiliary_v17",
            direction_id=direction, matched_suffix=f", the {agent}",
            base_text=present_past(base_present), donor_text=present_past(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="A2", construction_id="as_of_now_at_that_time_auxiliary_v17",
            direction_id=direction, matched_suffix=f", the {agent}",
            base_text=now_then(base_present), donor_text=now_then(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="P", construction_id="present_past_paraphrase_auxiliary_v17",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f", the {agent}", base_text=present_past(base_present),
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
        raise CandidateBankError("rows differ from sealed v17 authority")
    base = v16.v15.v14.v13.v12.v11.v10.v9.v8.v1
    try:
        digest = base.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except base.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = (
        base, v16.v15.v14.v13.v12.v11.v10.v9.v8.v2,
        v16.v15.v14.v13.v12.v11.v10.v9.v8.v3,
        v16.v15.v14.v13.v12.v11.v10.v9.v8.v4,
        v16.v15.v14.v13.v12.v11.v10.v9.v8.v5,
        v16.v15.v14.v13.v12.v11.v10.v9.v8.v6,
        v16.v15.v14.v13.v12.v11.v10.v9.v8.v7,
        v16.v15.v14.v13.v12.v11.v10.v9.v8,
        v16.v15.v14.v13.v12.v11.v10.v9,
        v16.v15.v14.v13.v12.v11.v10,
        v16.v15.v14.v13.v12.v11,
        v16.v15.v14.v13.v12,
        v16.v15.v14.v13,
        v16.v15.v14,
        v16.v15,
        v16,
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
        raise CandidateBankError("v17 count, novelty, or construction check failed")
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
