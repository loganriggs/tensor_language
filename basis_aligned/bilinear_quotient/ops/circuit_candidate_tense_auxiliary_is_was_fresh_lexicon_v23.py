#!/usr/bin/env python3
"""History-disjoint, position-aligned v23 bank for four-head confirmation."""
from __future__ import annotations

import importlib
from typing import Any, Mapping, Sequence

import circuit_candidate_aspectual_different_readout_is_was_v1 as base

SCHEMA, SPLIT, GROUPS = base.SCHEMA, base.SPLIT, 16
SEED = 20261123
TASK_ID, TASK_SPEC, READOUT = base.TASK_ID, base.TASK_SPEC, base.READOUT
canonical_sha256, CandidateBankError = base.canonical_sha256, base.CandidateBankError
_AGENTS = (
    "theorist", "logician", "geometer", "algebraist",
    "microbiologist", "anesthesiologist", "psychiatrist", "educator",
    "publisher", "author", "playwright", "screenwriter",
    "critic", "essayist", "grammarian", "semiotician",
)


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[group_number], _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    day = lambda present: f"{'In the present day' if present else 'In days gone by'}, the {agent}"
    age = lambda present: f"{'In the current age' if present else 'In a former age'}, the {agent}"
    paraphrase = lambda present: f"{'At the present time' if present else 'In times gone by'}, the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'aligned_four_head_confirm_v23', SEED, group_number])[:24]}"
    sentence_types = (
        "present_progressive" if base_present else "past_progressive",
        "present_progressive" if donor_present else "past_progressive",
    )
    common = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=agent, alternate_reporter=alternate, adjective="aligned_four_head_confirm_v23",
        object_name="present_day_gone_by_current_former_age", spec=TASK_SPEC,
        vocabulary=READOUT,
    )
    suffix = f", the {agent}"
    control_answer = answer(base_present)
    rows = [
        base.builder._row(
            **common, transform_id="A1", construction_id="present_day_days_gone_by_auxiliary_v23",
            direction_id=direction, matched_suffix=suffix,
            base_text=day(base_present), donor_text=day(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="A2", construction_id="current_former_age_auxiliary_v23",
            direction_id=direction, matched_suffix=suffix,
            base_text=age(base_present), donor_text=age(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types,
        ),
        base.builder._row(
            **common, transform_id="P", construction_id="day_age_paraphrase_auxiliary_v23",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=suffix, base_text=day(base_present),
            donor_text=paraphrase(base_present), base_answer=control_answer,
            donor_answer=control_answer, sentence_types=(sentence_types[0], sentence_types[0]),
        ),
        base.builder._row(
            **common, transform_id="C", construction_id="harbor_canyon_equal_length_control_v23",
            direction_id="harbor_to_canyon" if forward else "canyon_to_harbor",
            matched_suffix=suffix,
            base_text=f"Near the {'harbor' if forward else 'canyon'}{suffix}",
            donor_text=f"Near the {'canyon' if forward else 'harbor'}{suffix}",
            base_answer=control_answer, donor_answer=control_answer,
            sentence_types=(sentence_types[0], sentence_types[0]),
        ),
    ]
    if any(row["base_semantic_position"] != row["donor_semantic_position"] for row in rows):
        raise CandidateBankError("v23 requires position-compatible base/donor rows")
    return rows


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _history_modules():
    modules = [importlib.import_module(f"circuit_candidate_aspectual_different_readout_is_was_v{i}")
               for i in range(1, 4)]
    modules.extend(importlib.import_module(
        f"circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v{i}") for i in range(4, 23))
    return tuple(modules)


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v23 authority")
    try:
        digest = base.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except base.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_banks = tuple(module._build() if hasattr(module, "_build") else module.build_rows()
                      for module in _history_modules())
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old_banks for row in bank}
    old_text = {row[key] for bank in old_banks for row in bank
                for key in ("base_text", "donor_text")}
    ids = {str(row["row_id"]) for row in materialized}
    if (len(materialized) != 64 or len(ids) != 64 or ids & old_ids
            or set(_AGENTS) & old_agents
            or any(row[key] in old_text for row in materialized for key in ("base_text", "donor_text"))
            or any(not all(row["construction_checks"].values()) for row in materialized)
            or any(row["base_semantic_position"] != row["donor_semantic_position"]
                   for row in materialized)):
        raise CandidateBankError("v23 count, novelty, construction, or alignment check failed")
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
    rows = build_rows()
    print("authority:", authority_sha256())
    for family in ("A1", "A2", "P", "C"):
        row = next(item for item in rows if item["family"] == family)
        print(family, row["base_semantic_position"], row["base_text"], "->", row["donor_text"])
