#!/usr/bin/env python3
"""degree_frame.comparative_vs_equative -- the corpus's second construction-obligatory variable.

A degree head fixes its own complementizer: `more` obliges `than`, `as` obliges `as`. The
recipient and donor differ ONLY in that degree word and end on the same adjective token, so a
site that transfers the than/as decision is carrying the degree frame rather than a surface
property of the patched position.

    A1 verb complement
      "The leader called the window more short" -> " than"
      "The leader called the window as short"   -> " as"
    A2 seem predicate
      "In the notes the window seemed more short" -> " than"
      "In the notes the window seemed as short"   -> " as"

The nearest existing behaviour is `correlative_state.either_vs_neither`, which is also
obligatory, but its variable is a COORDINATION choice fixed by a paired connective; here the
licenser is a degree head. P rewrites the agent while holding the degree word, so it holds the
answer.

Control. C is same-answer -- both sides answer " than" -- but drawn from a DIFFERENT licenser:
`rather than`, a preference construction that obliges " than" with no degree comparison at all.
That matters because my first correlative draft made C a copy of P, which tests nothing. Its
measured ceiling in this configuration is 0.034 to 0.142 against a 0.35 bar, so the C clause is
reported as a statistic and not as a passed test.

Code path. Reuses the row builder pinned by
`circuit_fast_screen_candidate_sentence_terminal_context_control` (digest d0da3cda...), so the
construction checks here are already-proven ones.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_candidates as lex

canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED

TASK_ID = "degree_frame.comparative_vs_equative"
DEGREE = (" than", " as")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_degree_frame_fit_panels",
    answer_role="score_jointly_tokenized_than_versus_as",
    transforms=(
        battery.TransformSpec("A1", "verb_complement_degree_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "seem_predicate_degree_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        battery.TransformSpec("C", "rather_than_preference_frame", False, "registered_active"),
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _verb_complement(agent: str, adjective: str, noun: str, more: bool) -> str:
    return f"The {agent} called {_phrase(adjective, noun)} {'more' if more else 'as'} {adjective}"


def _seem_predicate(agent: str, adjective: str, noun: str, more: bool) -> str:
    return f"In the notes {_phrase(adjective, noun)} seemed {'more' if more else 'as'} {adjective}"


def _control(agent: str, adjective: str, noun: str, kept: bool) -> str:
    """A PREFERENCE frame: 'rather than'.

    " than" here is licensed by `rather`, a preference construction with no degree comparison,
    so C tests a different licenser rather than repeating P's answer-preserving edit. Both sides
    end on the same ` rather` token.
    """
    verb = "chose" if kept else "kept"
    return f"The {agent} {verb} the {adjective} {noun} rather"


def _answer(more: bool) -> str:
    return DEGREE[0] if more else DEGREE[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "comparative_to_equative" if forward else "equative_to_comparative"
    phrase = f" {adjective}"
    common = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, vocabulary=DEGREE, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("comparative" if base_more else "equative",
             "comparative" if donor_more else "equative")
    return [
        builder._row(**common, transform_id="A1", construction_id="verb_complement",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_verb_complement(agent, adjective, noun, base_more),
                     donor_text=_verb_complement(agent, adjective, noun, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="seem_predicate",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_seem_predicate(agent, adjective, noun, base_more),
                     donor_text=_seem_predicate(agent, adjective, noun, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"verb_complement_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_verb_complement(p_agent, adjective, noun, base_more),
                     donor_text=_verb_complement(p_donor_agent, adjective, noun, base_more),
                     base_answer=_answer(base_more), donor_answer=_answer(base_more),
                     sentence_types=(kinds[0], kinds[0])),
        builder._row(**common, transform_id="C",
                     construction_id="rather_than_preference_frame",
                     direction_id="base_to_donor" if forward else "donor_to_base",
                     matched_suffix=" rather",
                     base_text=_control(agent if forward else alternate, adjective, noun, True),
                     donor_text=_control(alternate if forward else agent, adjective, noun, False),
                     base_answer=DEGREE[0], donor_answer=DEGREE[0],
                     sentence_types=("preference", "preference")),
    ]


def _build(groups: int, seed: int) -> list[dict[str, Any]]:
    order = lex._permutation(seed)
    return [row for group_number in range(groups)
            for row in _panel(seed, group_number, order[group_number])]


def _validate(rows, groups: int, seed: int) -> str:
    if type(groups) is not int or not 2 <= groups <= DEFAULT_GROUPS or groups % 2:
        raise CandidateBankError("groups must be an even integer from 2 through 32")
    materialized = [dict(row) for row in rows]
    if materialized != _build(groups, seed):
        raise CandidateBankError("rows differ from the deterministic semantic authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    if len({str(row["row_id"]) for row in materialized}) != len(materialized):
        raise CandidateBankError("row IDs are not unique")
    cells: dict[tuple[str, str], int] = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise CandidateBankError("a stored construction check is false")
        key = (str(row["transform_id"]), str(row["direction_id"]))
        cells[key] = cells.get(key, 0) + 1
    half = groups // 2
    for transform in ("A1", "A2"):
        if cells.get((transform, "comparative_to_equative")) != half \
                or cells.get((transform, "equative_to_comparative")) != half:
            raise CandidateBankError(f"{transform} ordered directions are unbalanced")
    if cells.get(("C", "base_to_donor")) != half or cells.get(("C", "donor_to_base")) != half:
        raise CandidateBankError("C ordered directions are unbalanced")
    return digest


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rows = _build(groups, seed)
    _validate(rows, groups, seed)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
                  groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    return _validate(rows, groups, seed)


def authority_sha256(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
                     seed: int = DEFAULT_SEED) -> str:
    return _validate(build_rows(task_id, groups, seed), groups, seed)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for family in ("A1", "A2", "P", "C"):
        r = next(x for x in rows if x["family"] == family)
        print(f"  {family} [{r['direction_id']}] changes={r['answer_changes']}")
        print(f"    base : {r['base_text']!r} -> {r['base_answer']!r}")
        print(f"    donor: {r['donor_text']!r} -> {r['donor_answer']!r}")
