#!/usr/bin/env python3
"""correlative_state.either_vs_neither -- the corpus's first construction-OBLIGATORY variable.

Every behaviour screened so far tracks either an unclosed delimiter (bracket_pending_opener,
quote_parity), a feature of an earlier phrase (subject_verb, narrative_tense), a value
(numbered_list, numeric_sequence), or a sentence mode (sentence_terminal). This one is
different in kind: having opened `either`, the model is grammatically COMMITTED to `or`, and
having opened `neither`, to `nor`. The cue creates a syntactic obligation and the prediction is
its satisfaction.

    A1 complement clause
      "The record notes that either the bright lantern"  -> " or"
      "The record notes that neither the bright lantern" -> " nor"
    A2 verb-object
      "The clerk chose either the bright lantern"        -> " or"
      "The clerk chose neither the bright lantern"       -> " nor"

Recipient and donor differ ONLY in the correlative word and end on the same final token, so a
site that transfers the or/nor decision is carrying the open correlative rather than a surface
property of the patched position. A2 re-expresses the same variable in a different construction
so a single-construction cue cannot satisfy both. P rewrites the noun while holding the
correlative, and therefore holds the answer.

Control, stated plainly. C is same-answer -- both sides answer " or" under the `either` frame,
varying only lexical slots irrelevant to the correlative. Measured ceilings for this
configuration run 0.034 to 0.142 against a 0.35 bar, so the C clause is NOT at risk; the honest
content of a passing screen is target recovery plus P invariance, with C/A1 at the selected site
as the discriminating statistic. Both control sides use ONE frame the model demonstrably
continues: using two different frames is what made my narrative_tense v1 stop at the capability
gate.

Control on the code path. This reuses the row builder from
`circuit_fast_screen_candidate_sentence_terminal_context_control`, whose faithfulness is pinned
by rebuilding the v1 sentence-terminal rows to digest d0da3cda... The construction checks here
are therefore already-proven ones, not new checks written for this behaviour.
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

TASK_ID = "correlative_state.either_vs_neither"
CORRELATIVE = (" or", " nor")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_correlative_state_fit_panels",
    answer_role="score_jointly_tokenized_or_versus_nor",
    transforms=(
        battery.TransformSpec("A1", "complement_clause_correlative_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "verb_object_correlative_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        battery.TransformSpec("C", "numeric_range_disjunction", False, "registered_active"),
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


def _phrase(adjective: str, noun: str) -> str:
    return f"the {adjective} {noun}"


def _complement(agent: str, adjective: str, noun: str, either: bool) -> str:
    word = "either" if either else "neither"
    return f"The {agent} notes that {word} {_phrase(adjective, noun)}"


def _verb_object(agent: str, adjective: str, noun: str, either: bool) -> str:
    word = "either" if either else "neither"
    return f"The {agent} chose {word} {_phrase(adjective, noun)}"


def _control(agent: str, verb_phrase: str) -> str:
    """A NUMERIC-RANGE disjunction: 'take two or three'.

    The first draft made C a copy of P -- the same either-frame with the agent swapped -- which
    tests nothing, since a control must be an UNRELATED behaviour rather than a second
    answer-preserving edit. Here " or" is licensed by an open numeric range, not by a
    correlative obligation, so the two families genuinely differ in what determines the answer,
    while both sides end on the same numeral token.
    """
    return f"The {agent} said the {verb_phrase} two"


def _answer(either: bool) -> str:
    return CORRELATIVE[0] if either else CORRELATIVE[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_either, donor_either = (True, False) if forward else (False, True)
    direction = "either_to_neither" if forward else "neither_to_either"
    phrase = _phrase(adjective, noun)
    common = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, vocabulary=CORRELATIVE, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("either" if base_either else "neither", "either" if donor_either else "neither")
    return [
        builder._row(**common, transform_id="A1", construction_id="complement_clause",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_complement(agent, adjective, noun, base_either),
                     donor_text=_complement(agent, adjective, noun, donor_either),
                     base_answer=_answer(base_either), donor_answer=_answer(donor_either),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="verb_object",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_verb_object(agent, adjective, noun, base_either),
                     donor_text=_verb_object(agent, adjective, noun, donor_either),
                     base_answer=_answer(base_either), donor_answer=_answer(donor_either),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"complement_clause_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_complement(p_agent, adjective, noun, base_either),
                     donor_text=_complement(p_donor_agent, adjective, noun, base_either),
                     base_answer=_answer(base_either), donor_answer=_answer(base_either),
                     sentence_types=(kinds[0], kinds[0])),
        builder._row(**common, transform_id="C",
                     construction_id="numeric_range_disjunction",
                     direction_id="base_to_donor" if forward else "donor_to_base",
                     matched_suffix=" two",
                     base_text=_control(agent if forward else alternate, "trip would take"),
                     donor_text=_control(alternate if forward else agent, "walk would last"),
                     base_answer=CORRELATIVE[0], donor_answer=CORRELATIVE[0],
                     sentence_types=("numeric_range", "numeric_range")),
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
        if cells.get((transform, "either_to_neither")) != half \
                or cells.get((transform, "neither_to_either")) != half:
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
