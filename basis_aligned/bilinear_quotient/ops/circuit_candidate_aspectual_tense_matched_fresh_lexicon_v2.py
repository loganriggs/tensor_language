#!/usr/bin/env python3
"""Second prospective matched-occupation authority for dual task-gate tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_candidate_aspectual_different_readout_is_was_v1 as is_v1
import circuit_candidate_aspectual_different_readout_is_was_v2 as is_v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as is_v3
import circuit_candidate_aspectual_fresh_construction_v2 as has_v2
import circuit_candidate_aspectual_fresh_lexicon_v3 as has_v3
import circuit_candidate_aspectual_fresh_lexicon_v4 as has_v4
import circuit_candidate_aspectual_fresh_lexicon_v5 as has_v5
import circuit_candidate_aspectual_fresh_lexicon_v6 as has_v6
import circuit_candidate_aspectual_lexicon_factory as has_factory
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v4 as is_v4
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v5 as is_v5
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6 as is_v6
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v7 as is_v7


HAS_SEED, HAS_TAG = 20261017, "fresh_lexicon_v7_affinity"
IS_SEED, GROUPS = 20261018, 16
_AGENTS = (
    "assistant", "consultant", "coordinator", "dispatcher", "executive", "machinist", "pharmacist", "politician",
    "reporter", "supervisor", "waiter", "welder", "conductor", "announcer", "salesperson", "cashier",
)
_PERIODS = (
    "day", "hour", "morning", "evening", "term", "administration", "broadcast", "census",
    "celebration", "demonstration", "exhibition", "game", "marathon", "presentation", "race", "retreat",
)
_HAS_PRIORS = (has_v2, has_v3, has_v4, has_v5, has_v6)
_IS_PRIORS = (is_v1, is_v2, is_v3, is_v4, is_v5, is_v6, is_v7)


class MatchedBankError(RuntimeError):
    pass


def _is_panel(group_number: int) -> list[dict[str, Any]]:
    agent = _AGENTS[group_number]
    alternate = _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    answer = lambda present: is_v1.READOUT[0] if present else is_v1.READOUT[1]
    plain = lambda present: f"At {'this' if present else 'that'} moment the {agent}"
    embedded = lambda present: f"The bulletin reports that at {'this' if present else 'that'} moment the {agent}"
    paraphrase = lambda present: f"At the {'present' if present else 'previous'} moment the {agent}"
    group_id = f"FIT:{is_v1.canonical_sha256([is_v1.SCHEMA, is_v1.TASK_ID, 'different_readout_v8_affinity', IS_SEED, group_number])[:24]}"
    sentence_types = ("present_progressive" if base_present else "past_progressive", "present_progressive" if donor_present else "past_progressive")
    common_no_vocab = dict(seed=IS_SEED, task_id=is_v1.TASK_ID, group_number=group_number, group_id=group_id, reporter=agent, alternate_reporter=alternate, adjective="different_readout_v8_affinity", object_name="moment", spec=is_v1.TASK_SPEC)
    common = dict(common_no_vocab, vocabulary=is_v1.READOUT)
    suffix = f" moment the {agent}"
    return [
        is_v1.builder._row(**common, transform_id="A1", construction_id="this_that_moment_auxiliary", direction_id=direction, matched_suffix=suffix, base_text=plain(base_present), donor_text=plain(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
        is_v1.builder._row(**common, transform_id="A2", construction_id="bulletin_embedded_this_that_moment_auxiliary", direction_id=direction, matched_suffix=suffix, base_text=embedded(base_present), donor_text=embedded(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
        is_v1.builder._row(**common, transform_id="P", construction_id="temporal_moment_paraphrase_same_auxiliary", direction_id="primary_to_alternative" if forward else "alternative_to_primary", matched_suffix=f" the {agent}", base_text=plain(base_present), donor_text=paraphrase(base_present), base_answer=answer(base_present), donor_answer=answer(base_present), sentence_types=(sentence_types[0], sentence_types[0])),
        is_v1.builder._row(**common_no_vocab, transform_id="C", **is_v1.canonical.row_kwargs(group_number, forward)),
    ]


def build_rows_by_bank() -> dict[str, list[dict[str, Any]]]:
    return {
        "has_had": has_factory.build(seed=HAS_SEED, tag=HAS_TAG, agents=_AGENTS, periods=_PERIODS),
        "is_was": [row for group_number in range(GROUPS) for row in _is_panel(group_number)],
    }


def validate_rows_by_bank(rows_by_bank: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, str]:
    if set(rows_by_bank) != {"has_had", "is_was"}:
        raise MatchedBankError("exactly two named banks required")
    has_rows, is_rows = list(rows_by_bank["has_had"]), list(rows_by_bank["is_was"])
    has_prior_rows = tuple(module.build_rows() for module in _HAS_PRIORS)
    has_digest = has_factory.validate(has_rows, seed=HAS_SEED, tag=HAS_TAG, agents=_AGENTS, periods=_PERIODS, disjoint_row_sets=tuple({str(row["row_id"]) for row in bank} for bank in has_prior_rows))
    if is_rows != [row for group_number in range(GROUPS) for row in _is_panel(group_number)]:
        raise MatchedBankError("is/was rows differ from sealed construction")
    try:
        is_digest = is_v1.battery.validate_rows(is_v1.TASK_SPEC, is_rows, required_phases=(is_v1.SPLIT,))
    except is_v1.battery.BatteryContractError as error:
        raise MatchedBankError(str(error)) from error
    is_prior_rows = tuple(module.build_rows() for module in _IS_PRIORS)
    old_ids = {str(row["row_id"]) for bank in is_prior_rows for row in bank}
    old_agents = {str(row.get("reporter")) for bank in is_prior_rows for row in bank}
    ids = {str(row["row_id"]) for row in is_rows}
    if len(has_rows) != 64 or len(is_rows) != 64 or len(ids) != 64 or ids & old_ids or set(_AGENTS) & old_agents or any(not all(row["construction_checks"].values()) for row in is_rows):
        raise MatchedBankError("count, uniqueness, disjointness, occupation novelty, or construction check failed")
    if {str(row.get("reporter")) for row in has_rows} != set(_AGENTS) or {str(row.get("reporter")) for row in is_rows} != set(_AGENTS):
        raise MatchedBankError("banks are not occupation matched")
    return {"has_had": has_digest, "is_was": is_digest}


def authority_sha256() -> dict[str, str]:
    return validate_rows_by_bank(build_rows_by_bank())


if __name__ == "__main__":
    print(authority_sha256())
