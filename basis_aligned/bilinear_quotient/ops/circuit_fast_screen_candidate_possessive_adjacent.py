"""possessive_number with an ADJACENT antecedent -- isolating controller locality.

Three number screens now read: copula agreeing with its own local subject PASSES; possessive
pronoun six tokens from its antecedent FAILS; existential auxiliary separated from its notional
subject by an expletive FAILS. That is consistent with two different stories -- controller
locality, or the target simply not being the agreeing verb -- and they have not been separated.

This varies ONE thing against the `possessive_number` authority: the DISTANCE. The target is
still a possessive pronoun, the answer vocabulary, control and construction contrast are
unchanged.

    v1   "The pilots finished the work and put away" -> " their"     antecedent 6 tokens back
    here "The clerks lost"                           -> " their"     antecedent 1 token back
         "The clerk lost"                            -> " his"

Prediction registered in the prior-art receipt before this ran: if locality is what matters, an
adjacent antecedent passes where the six-token version failed; if it fails, the pronoun target is
the blocker regardless of distance. Either way one of the two live explanations dies.

Control. Canonical same-answer control v2, unchanged from v1.
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

TASK_ID = "possessive_number.adjacent_antecedent"
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
_A1_VERBS = ("lost", "kept", "signed", "sealed", "packed", "sorted", "counted", "cleared",
             "folded", "labelled", "measured", "polished", "gathered", "emptied", "weighed",
             "mapped", "traced", "framed", "listed", "checked", "marked", "handled", "stored",
             "shifted", "tested", "washed", "trimmed", "fixed", "swept", "stacked", "opened",
             "closed")
_A2_VERBS = ("kept", "lost", "sealed", "signed", "sorted", "packed", "cleared", "counted",
             "labelled", "folded", "polished", "measured", "emptied", "gathered", "mapped",
             "weighed", "framed", "traced", "checked", "listed", "handled", "marked", "shifted",
             "stored", "washed", "tested", "fixed", "trimmed", "stacked", "swept", "closed",
             "opened")
if not (len(_SINGULAR) == len(_A1_VERBS) == len(_A2_VERBS) == DEFAULT_GROUPS):
    raise RuntimeError("possessive tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _conjunct(noun: str, plural: bool, action: str) -> str:
    """Antecedent ONE token before the pronoun; v1 put six tokens of material between them."""
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
    phrase = f" {act1}"
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
                     direction_id=direction, matched_suffix=f" {act2}",
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
