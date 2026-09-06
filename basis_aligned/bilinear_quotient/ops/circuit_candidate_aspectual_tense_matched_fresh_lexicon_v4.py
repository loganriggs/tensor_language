#!/usr/bin/env python3
"""Fourth prospective matched lexicon for construction-complete mode validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v3 as previous


base = previous.previous
HAS_SEED, HAS_TAG = 20261021, "fresh_lexicon_v10_endpoint_mode_validation"
IS_SEED, GROUPS = 20261022, 16
_AGENTS = (
    "acrobat", "architect", "astronomer", "botanist", "carpenter", "composer",
    "dentist", "editor", "farmer", "glassblower", "historian", "illustrator",
    "journalist", "librarian", "mechanic", "navigator",
)
_PERIODS = (
    "apprenticeship", "campaign", "conference", "contract", "course", "deployment",
    "excursion", "internship", "residency", "retreat", "rotation", "sabbatical",
    "seminar", "term", "trial", "voyage",
)


class MatchedBankError(RuntimeError):
    pass


def _is_panel(group_number: int) -> list[dict[str, Any]]:
    is_v1 = base.is_v1
    agent = _AGENTS[group_number]
    alternate = _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: is_v1.READOUT[0] if present else is_v1.READOUT[1]
    plain = lambda present: f"At {'this' if present else 'that'} moment the {agent}"
    embedded = lambda present: f"The bulletin reports that at {'this' if present else 'that'} moment the {agent}"
    paraphrase = lambda present: f"At the {'present' if present else 'previous'} moment the {agent}"
    group_id = f"FIT:{is_v1.canonical_sha256([is_v1.SCHEMA, is_v1.TASK_ID, 'different_readout_v10_endpoint_mode_validation', IS_SEED, group_number])[:24]}"
    sentence_types = ("present_progressive" if base_present else "past_progressive",
                      "present_progressive" if donor_present else "past_progressive")
    common_no_vocab = dict(seed=IS_SEED, task_id=is_v1.TASK_ID,
        group_number=group_number, group_id=group_id, reporter=agent,
        alternate_reporter=alternate, adjective="different_readout_v10_endpoint_mode_validation",
        object_name="moment", spec=is_v1.TASK_SPEC)
    common = dict(common_no_vocab, vocabulary=is_v1.READOUT)
    suffix = f" moment the {agent}"
    return [
        is_v1.builder._row(**common, transform_id="A1",
            construction_id="this_that_moment_auxiliary", direction_id=direction,
            matched_suffix=suffix, base_text=plain(base_present), donor_text=plain(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types),
        is_v1.builder._row(**common, transform_id="A2",
            construction_id="bulletin_embedded_this_that_moment_auxiliary", direction_id=direction,
            matched_suffix=suffix, base_text=embedded(base_present), donor_text=embedded(donor_present),
            base_answer=answer(base_present), donor_answer=answer(donor_present),
            sentence_types=sentence_types),
        is_v1.builder._row(**common, transform_id="P",
            construction_id="temporal_moment_paraphrase_same_auxiliary",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=f" the {agent}", base_text=plain(base_present),
            donor_text=paraphrase(base_present), base_answer=answer(base_present),
            donor_answer=answer(base_present), sentence_types=(sentence_types[0], sentence_types[0])),
        is_v1.builder._row(**common_no_vocab, transform_id="C",
                          **is_v1.canonical.row_kwargs(group_number, forward)),
    ]


def build_rows_by_bank() -> dict[str, list[dict[str, Any]]]:
    return {"has_had": base.has_factory.build(
                seed=HAS_SEED, tag=HAS_TAG, agents=_AGENTS, periods=_PERIODS),
            "is_was": [row for group_number in range(GROUPS) for row in _is_panel(group_number)]}


def validate_rows_by_bank(rows_by_bank: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, str]:
    if set(rows_by_bank) != {"has_had", "is_was"}:
        raise MatchedBankError("exactly two named banks required")
    has_rows, is_rows = list(rows_by_bank["has_had"]), list(rows_by_bank["is_was"])
    old = previous.build_rows_by_bank()
    has_digest = base.has_factory.validate(
        has_rows, seed=HAS_SEED, tag=HAS_TAG, agents=_AGENTS, periods=_PERIODS,
        disjoint_row_sets=({str(row["row_id"]) for row in old["has_had"]},))
    if is_rows != [row for group_number in range(GROUPS) for row in _is_panel(group_number)]:
        raise MatchedBankError("is/was rows differ from sealed construction")
    try:
        is_digest = base.is_v1.battery.validate_rows(
            base.is_v1.TASK_SPEC, is_rows, required_phases=(base.is_v1.SPLIT,))
    except base.is_v1.battery.BatteryContractError as error:
        raise MatchedBankError(str(error)) from error
    new_ids = {str(row["row_id"]) for row in has_rows + is_rows}
    old_ids = {str(row["row_id"]) for bank in old.values() for row in bank}
    old_agents = {str(row.get("reporter")) for bank in old.values() for row in bank}
    if (len(has_rows) != 64 or len(is_rows) != 64 or len(new_ids) != 128
            or new_ids & old_ids or set(_AGENTS) & old_agents
            or any(not all(row["construction_checks"].values()) for row in is_rows)
            or {str(row.get("reporter")) for row in has_rows} != set(_AGENTS)
            or {str(row.get("reporter")) for row in is_rows} != set(_AGENTS)):
        raise MatchedBankError("count, uniqueness, novelty, vocabulary, or construction check failed")
    return {"has_had": has_digest, "is_was": is_digest}


def authority_sha256() -> dict[str, str]:
    return validate_rows_by_bank(build_rows_by_bank())


if __name__ == "__main__":
    print(authority_sha256())
