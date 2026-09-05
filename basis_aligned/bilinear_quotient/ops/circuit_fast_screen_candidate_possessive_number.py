"""possessive_number.their_vs_his -- grammatical NUMBER in the position where GENDER failed.

`pronoun_antecedent.gender_reference` is a null: A1 and A2 at chance, margin 0.00 over 64 rows.
The model does not carry an antecedent's gender to a pronoun. Three lexical-semantic cues have
now failed cross-construction capability while seven grammatical ones have passed -- but that
summary has a confound I did not control: **gender failed on a PRONOUN, and no passing behaviour
predicts a pronoun at all.**

This holds the position and the word class fixed -- the prediction is a possessive pronoun,
exactly as in the gender screen -- and varies only the feature carried:

    A1 conjunct frame
      "The pilots finished the work and put away" -> " their"
      "The pilot finished the work and put away"  -> " his"
    A2 notes frame
      "In the notes the pilots signed the form and collected" -> " their"
      "In the notes the pilot signed the form and collected"  -> " his"

Recipient and donor differ only in the number of an early noun and end on the same verb token.

Prediction registered in the prior-art receipt before this ran: if the boundary is about feature
TYPE, this passes where gender failed; if it fails too, the boundary is about the pronoun
position and the seven-versus-three summary is wrong.

Control. Canonical same-answer control v2, so this C joins the comparable set directly.
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

TASK_ID = "possessive_number.their_vs_his"
POSSESSIVE = (" their", " his")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_possessive_number_fit_panels",
    answer_role="score_jointly_tokenized_their_versus_his",
    transforms=(
        battery.TransformSpec("A1", "conjunct_frame_number_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "notes_frame_number_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_SINGULAR = tuple(pair[0] for pair in lex._REPORTERS)
_SINGULAR_ALT = tuple(pair[1] for pair in lex._REPORTERS)
_A1_VERBS = ("finished the work and put away", "checked the list and packed away",
             "closed the shed and stored away", "read the notice and filed away",
             "cleared the bench and tucked away", "sealed the crate and locked away",
             "signed the form and set away", "sorted the mail and put away",
             "ended the shift and packed away", "swept the floor and stored away",
             "fixed the latch and filed away", "counted the coins and tucked away",
             "washed the pans and locked away", "trimmed the rope and set away",
             "folded the cloth and put away", "stacked the crates and packed away",
             "labelled the jars and stored away", "measured the plank and filed away",
             "polished the lamp and tucked away", "gathered the tools and locked away",
             "emptied the sack and set away", "weighed the grain and put away",
             "mapped the route and packed away", "traced the wire and stored away",
             "framed the notice and filed away", "listed the goods and tucked away",
             "checked the seal and locked away", "marked the box and set away",
             "handled the freight and put away", "stored the barrels and packed away",
             "shifted the planks and stored away", "tested the pump and filed away")
_A2_VERBS = ("signed the form and collected", "read the notice and collected",
             "checked the list and collected", "closed the ledger and collected",
             "sealed the crate and collected", "sorted the mail and collected",
             "counted the coins and collected", "cleared the bench and collected",
             "ended the shift and collected", "folded the cloth and collected",
             "labelled the jars and collected", "measured the plank and collected",
             "polished the lamp and collected", "gathered the tools and collected",
             "emptied the sack and collected", "weighed the grain and collected",
             "mapped the route and collected", "traced the wire and collected",
             "framed the notice and collected", "listed the goods and collected",
             "checked the seal and collected", "marked the box and collected",
             "handled the freight and collected", "stored the barrels and collected",
             "shifted the planks and collected", "tested the pump and collected",
             "washed the pans and collected", "trimmed the rope and collected",
             "fixed the latch and collected", "swept the floor and collected",
             "stacked the crates and collected", "opened the gate and collected")
if not (len(_SINGULAR) == len(_A1_VERBS) == len(_A2_VERBS) == DEFAULT_GROUPS):
    raise RuntimeError("possessive tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _conjunct(noun: str, plural: bool, action: str) -> str:
    return f"The {noun}{'s' if plural else ''} {action}"


def _notes(noun: str, plural: bool, action: str) -> str:
    return f"In the notes the {noun}{'s' if plural else ''} {action}"


def _answer(plural: bool) -> str:
    return POSSESSIVE[0] if plural else POSSESSIVE[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "plural_to_singular" if forward else "singular_to_plural"
    noun, alt_noun = _SINGULAR[case_index], _SINGULAR_ALT[case_index]
    act1, act2 = _A1_VERBS[case_index], _A2_VERBS[case_index]
    # P preserves the answer by swapping the noun while holding the number.
    phrase = " " + act1.rsplit(" ", 1)[-1]
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("plural" if base_more else "singular",
             "plural" if donor_more else "singular")
    common = dict(common_no_vocab, vocabulary=POSSESSIVE)
    return [
        builder._row(**common, transform_id="A1", construction_id="conjunct_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_conjunct(noun, base_more, act1),
                     donor_text=_conjunct(noun, donor_more, act1),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="notes_frame",
                     direction_id=direction, matched_suffix=" " + act2.rsplit(" ", 1)[-1],
                     base_text=_notes(noun, base_more, act2),
                     donor_text=_notes(noun, donor_more, act2),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"conjunct_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_conjunct(noun, base_more, act1),
                     donor_text=_conjunct(alt_noun, base_more, act1),
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
        if cells.get((transform, "plural_to_singular")) != half \
                or cells.get((transform, "singular_to_plural")) != half:
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
