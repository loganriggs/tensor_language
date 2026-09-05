"""correlative_state.canonical_control -- correlative_state.either_vs_neither scored against the CANONICAL same-answer control.

A1, A2 and P are unchanged from the correlative_state.either_vs_neither authority. Only the control changes: it becomes
the fixed control in `ops/circuit_fast_screen_canonical_control.py`, byte-identical to the one
used by this candidate's sibling on the other side of the comparison.

Why. Measured at 11:46Z, swapping only a control frame moved C from 0.230 to 0.141 with the
verdict, selected site, passing band and P all unchanged, so C values from screens with
different controls are not comparable and the dependency-type ordering reported at 11:33Z was
withdrawn. This candidate and its sibling hold the control FIXED, so any difference in C between
them is a difference between the behaviours. This one is the obligation side.

The registered prediction for the pair, recorded in the prior-art receipt before either ran: a
real type difference shows the licensing behaviour's C exceeding the obligation's by at least
0.05; values within 0.03 of each other retire the finding.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_canonical_control as canonical
import circuit_fast_screen_candidates as lex

canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED

TASK_ID = "correlative_state.canonical_control"
CORRELATIVE = (" or", " nor")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_correlative_state_fit_panels",
    answer_role="score_jointly_tokenized_or_versus_nor",
    transforms=(
        battery.TransformSpec("A1", "complement_clause_correlative_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "verb_object_correlative_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
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
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("either" if base_either else "neither", "either" if donor_either else "neither")
    common = dict(common_no_vocab, vocabulary=CORRELATIVE)
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
        builder._row(**common_no_vocab, transform_id="C",
                     **canonical.row_kwargs(case_index, forward)),
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
