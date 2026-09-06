"""Second disjoint is/was confirmation authority with a new agent lexicon."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_candidate_aspectual_different_readout_is_was_v2 as v2


SCHEMA, SPLIT, GROUPS = v2.SCHEMA, v2.SPLIT, v2.GROUPS
SEED = 20260923
TASK_ID, TASK_SPEC, READOUT = v2.TASK_ID, v2.TASK_SPEC, v2.READOUT
canonical_sha256, CandidateBankError = v2.canonical_sha256, v2.CandidateBankError
_AGENTS = ("architect", "baker", "botanist", "captain", "composer", "dentist", "geologist", "historian", "journalist", "librarian", "mechanic", "painter", "physician", "sculptor", "translator", "watchmaker")


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent = _AGENTS[group_number]
    alternate = _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: READOUT[0] if present else READOUT[1]
    plain = lambda present: f"At {'this' if present else 'that'} moment the {agent}"
    embedded = lambda present: f"The bulletin reports that at {'this' if present else 'that'} moment the {agent}"
    paraphrase = lambda present: f"At the {'present' if present else 'previous'} moment the {agent}"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'different_readout_v3', SEED, group_number])[:24]}"
    sentence_types = ("present_progressive" if base_present else "past_progressive", "present_progressive" if donor_present else "past_progressive")
    common_no_vocab = dict(seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id, reporter=agent, alternate_reporter=alternate, adjective="different_readout_v3", object_name="moment", spec=TASK_SPEC)
    common = dict(common_no_vocab, vocabulary=READOUT)
    suffix = f" moment the {agent}"
    return [
        v2.v1.builder._row(**common, transform_id="A1", construction_id="this_that_moment_auxiliary", direction_id=direction, matched_suffix=suffix, base_text=plain(base_present), donor_text=plain(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
        v2.v1.builder._row(**common, transform_id="A2", construction_id="bulletin_embedded_this_that_moment_auxiliary", direction_id=direction, matched_suffix=suffix, base_text=embedded(base_present), donor_text=embedded(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
        v2.v1.builder._row(**common, transform_id="P", construction_id="temporal_moment_paraphrase_same_auxiliary", direction_id="primary_to_alternative" if forward else "alternative_to_primary", matched_suffix=f" the {agent}", base_text=plain(base_present), donor_text=paraphrase(base_present), base_answer=answer(base_present), donor_answer=answer(base_present), sentence_types=(sentence_types[0], sentence_types[0])),
        v2.v1.builder._row(**common_no_vocab, transform_id="C", **v2.v1.canonical.row_kwargs(group_number, forward)),
    ]


def _build() -> list[dict[str, Any]]:
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed different-readout-v3 authority")
    try:
        digest = v2.v1.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except v2.v1.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    ids = {str(row["row_id"]) for row in materialized}
    old_ids = {str(row["row_id"]) for row in v2.build_rows()} | {str(row["row_id"]) for row in v2.v1.build_rows()}
    if len(materialized) != 64 or len(ids) != 64 or ids & old_ids or any(not all(row["construction_checks"].values()) for row in materialized):
        raise CandidateBankError("v3 count, disjointness, uniqueness, or construction check failed")
    return digest


def build_rows(task_id: str = TASK_ID) -> list[dict[str, Any]]:
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
    for family in ("A1", "A2", "P", "C"):
        row = next(item for item in rows if item["family"] == family)
        print(f"{family}: {row['base_text']!r} -> {row['base_answer']!r}")
        print(f"    {row['donor_text']!r} -> {row['donor_answer']!r}")
