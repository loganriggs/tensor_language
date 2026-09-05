"""existential_agreement.were_vs_was -- grammatical number across an EXPLETIVE.

The notional subject sits several tokens back and the auxiliary agrees with it across an
intervening `there`, which is neither a copula agreeing with its own subject nor a pronoun
referring back to an antecedent.

    A1 locative frame
      "Beside the harbor the crates meant that there" -> " were"
      "Beside the harbor the crate meant that there"  -> " was"
    A2 notes frame
      "In the notes the crates showed that there" -> " were"
      "In the notes the crate showed that there"  -> " was"

Recipient and donor differ only in that noun's number and end on the same ` there` token.

Why this behaviour. Grammatical number passes at a COPULA (`subject_verb.number_agreement`,
Codex's authority) and fails at a POSSESSIVE PRONOUN (`possessive_number`, mine). Those are the
only two targets tested, so the position claim rests on a two-point contrast. This is a third
target. The prediction registered before the run: if position is what matters and the pronoun is
the hard case, this passes like the copula; if it fails, what fails is agreement across an
intervening element rather than pronouns as such, and the position claim needs restating.

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

TASK_ID = "existential_agreement.were_vs_was"
NUMBER = (" were", " was")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_existential_agreement_fit_panels",
    answer_role="score_jointly_tokenized_were_versus_was",
    transforms=(
        battery.TransformSpec("A1", "locative_frame_number_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "notes_frame_number_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_PLACES = lex._OBJECTS
_ADJECTIVES = lex._ADJECTIVES
_NOTIONAL = ("crate", "letter", "coin", "stone", "brick", "rope", "lamp", "basket",
             "plank", "nail", "wheel", "box", "cup", "plate", "tool", "wire",
             "tile", "blanket", "candle", "barrel", "bucket", "sack", "ladder", "bench",
             "chair", "door", "window", "mirror", "key", "lock", "hinge", "shelf")
if not (len(_PLACES) == len(_ADJECTIVES) == len(_NOTIONAL) == DEFAULT_GROUPS):
    raise RuntimeError("existential tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _locative(place: str, notional: str, plural: bool) -> str:
    return f"Beside the {place} the {notional}{'s' if plural else ''} meant that there"


def _notes(notional: str, plural: bool) -> str:
    return f"In the notes the {notional}{'s' if plural else ''} showed that there"


def _answer(plural: bool) -> str:
    return NUMBER[0] if plural else NUMBER[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "plural_to_singular" if forward else "singular_to_plural"
    place, adjective = _PLACES[case_index], _ADJECTIVES[case_index]
    notional = _NOTIONAL[case_index]
    # P preserves the answer by swapping the PLACE while holding the number.
    alt_place = _PLACES[(case_index + 7) % DEFAULT_GROUPS]
    phrase = " that there"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("plural" if base_more else "singular",
             "plural" if donor_more else "singular")
    common = dict(common_no_vocab, vocabulary=NUMBER)
    return [
        builder._row(**common, transform_id="A1", construction_id="locative_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_locative(place, notional, base_more),
                     donor_text=_locative(place, notional, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="notes_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_notes(notional, base_more),
                     donor_text=_notes(notional, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"locative_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_locative(place, notional, base_more),
                     donor_text=_locative(alt_place, notional, base_more),
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
