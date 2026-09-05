"""animacy_state.who_vs_which -- a LEXICAL-SEMANTIC cue, run to test a boundary.

Nothing in the form of `pilot` or `beacon` marks animacy, yet the relative pronoun that follows
is obligatory: a person is `who`, a thing is `which`.

    A1 notes frame
      "In the notes the pilot appeared often, the one"  -> " who"
      "In the notes the beacon appeared often, the one" -> " which"
    A2 record frame
      "The record mentions a pilot twice, the one"  -> " who"
      "The record mentions a beacon twice, the one" -> " which"

Recipient and donor differ only in that noun and end on the same ` one` token.

Why this behaviour. Every cue in the corpus that is a FUNCTION WORD or a verb's grammatical
subcategorization has passed native capability -- correlative, degree, complementizer, additive
scope, polarity, interrogative mood. Both cues that are LEXICAL-SEMANTIC properties of a content
word have failed: pronoun_antecedent.gender_reference sat at chance, and
countability_state.count_vs_mass was overridden by the verb frame's default. Animacy is squarely
in the second group.

Prediction registered in the prior-art receipt before this ran: a capability stop supports the
boundary; a selective site refutes it and sends me back to what the two earlier nulls actually
share. P holds the answer by swapping to a different noun of the SAME animacy.

Control. Canonical same-answer control v2, so this C enters the comparable set directly.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_canonical_control_v2 as canonical
import circuit_fast_screen_candidates as lex

canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED

TASK_ID = "animacy_state.who_vs_which"
ANIMACY = (" who", " which")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_animacy_fit_panels",
    answer_role="score_jointly_tokenized_who_versus_which",
    transforms=(
        battery.TransformSpec("A1", "notes_frame_animacy_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "record_frame_animacy_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_ANIMATE = tuple(pair[0] for pair in lex._REPORTERS)
_ANIMATE_ALT = tuple(pair[1] for pair in lex._REPORTERS)
_INANIMATE = lex._OBJECTS
if not (len(_ANIMATE) == len(_INANIMATE) == DEFAULT_GROUPS):
    raise RuntimeError("animacy tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _notes(noun: str) -> str:
    return f"In the notes the {noun} appeared often, the one"


def _record(noun: str) -> str:
    return f"The record mentions a {noun} twice, the one"


def _answer(animate: bool) -> str:
    return ANIMACY[0] if animate else ANIMACY[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "animate_to_inanimate" if forward else "inanimate_to_animate"
    animate, inanimate = _ANIMATE[case_index], _INANIMATE[case_index]
    base_noun = animate if base_more else inanimate
    donor_noun = animate if donor_more else inanimate
    # P must preserve the answer, so it swaps to a different noun of the SAME animacy.
    p_noun = _ANIMATE_ALT[case_index] if base_more else lex._OBJECTS[(case_index + 7) % DEFAULT_GROUPS]
    phrase = " the one"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("animate" if base_more else "inanimate",
             "animate" if donor_more else "inanimate")
    common = dict(common_no_vocab, vocabulary=ANIMACY)
    return [
        builder._row(**common, transform_id="A1", construction_id="notes_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_notes(base_noun),
                     donor_text=_notes(donor_noun),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="record_frame",
                     direction_id=direction, matched_suffix=" the one",
                     base_text=_record(base_noun),
                     donor_text=_record(donor_noun),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"notes_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_notes(base_noun),
                     donor_text=_notes(p_noun),
                     base_answer=_answer(base_more), donor_answer=_answer(base_more),
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
        if cells.get((transform, "animate_to_inanimate")) != half \
                or cells.get((transform, "inanimate_to_animate")) != half:
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
