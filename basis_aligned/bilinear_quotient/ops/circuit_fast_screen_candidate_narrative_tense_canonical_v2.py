"""narrative_tense.canonical_control -- narrative_tense.past_vs_present scored against canonical control V2 (disjoint vocabulary).

A1, A2 and P are unchanged from the narrative_tense.past_vs_present authority. Only the control changes, to
the fixed control in `ops/circuit_fast_screen_canonical_control.py`, byte-identical to the one
used by every other canonical-control candidate.

Why. Measured at 11:46Z, swapping only a control frame moved C from 0.230 to 0.141 with verdict,
selected site, passing band and P unchanged, so C values from screens with different controls are
not comparable; the dependency-type ordering was retired at 12:34Z on a preregistered criterion.
Holding the control fixed is what makes a cross-behaviour C comparison mean anything.

The prediction registered for the five-behaviour set before these ran: if the control was the
whole story, all five C values fall within a spread of 0.05; a behaviour outside that band is a
real behaviour effect surviving a fixed control.
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

TASK_ID = "narrative_tense.canonical_control_v2"
TENSE = (" was", " is")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_narrative_tense_fit_panels",
    answer_role="score_jointly_tokenized_past_versus_present_copula",
    transforms=(
        battery.TransformSpec("A1", "direct_narration_tense_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "relative_clause_tense_swap", True, "toward_donor"),
        battery.TransformSpec("P", "subject_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_SUBJECTS = tuple(pair[0] for pair in lex._REPORTERS)          # 32 singular animate/inanimate slots
_ALTERNATES = tuple(pair[1] for pair in lex._REPORTERS)
_PLACES = lex._OBJECTS                                          # 32 place-like nouns
_FOCUS = lex._ADJECTIVES                                        # 32 adjectives for the focus noun


def _tail(place: str, focus: str) -> str:
    return f"The main reason for the {focus} {place}"


def _direct(subject: str, place: str, focus: str, past: bool) -> str:
    when, verb = ("Last winter", "stood") if past else ("Every winter", "stands")
    return f"{when} the {subject} {verb} nearby. {_tail(place, focus)}"


def _relative(subject: str, place: str, focus: str, past: bool) -> str:
    verb, when, draw = (("stood", "last winter", "drew")
                        if past else ("stands", "every winter", "draws"))
    return (f"The {subject} that {verb} nearby {when} {draw} crowds. "
            f"{_tail(place, focus)}")


def _control(subject: str, place: str, focus: str, variant: int) -> str:
    # v1 used two DIFFERENT frames here and the model handled only one of them: the
    # "earliest note on" frame scored 1.00 while "oldest record of" scored 0.06, so the
    # control failed the capability gate and stopped the screen -- the same authoring error
    # this candidate's own prior art documents in sentence_terminal and quote_parity.
    # Both sides now use the frame the model demonstrably continues, varying only subject
    # and place, which are irrelevant to tense.
    return (f"In those days the {subject} held a market nearby. "
            f"The earliest note on the {focus} {place}")


def _answer(past: bool) -> str:
    return TENSE[0] if past else TENSE[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    subject, alternate = _SUBJECTS[case_index], _ALTERNATES[case_index]
    place, focus = _PLACES[case_index], _FOCUS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_past, donor_past = (True, False) if forward else (False, True)
    direction = ("past_to_present" if forward else "present_to_past")
    tail = _tail(place, focus)
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=subject, alternate_reporter=alternate, adjective=focus,
                  object_name=place, spec=TASK_SPEC)
    p_subject, p_donor_subject = (subject, alternate) if forward else (alternate, subject)
    control_tail = f"{focus} {place}"
    common = dict(common_no_vocab, vocabulary=TENSE)
    return [
        builder._row(**common, transform_id="A1", construction_id="direct_narration",
                     direction_id=direction, matched_suffix=tail,
                     base_text=_direct(subject, place, focus, base_past),
                     donor_text=_direct(subject, place, focus, donor_past),
                     base_answer=_answer(base_past), donor_answer=_answer(donor_past),
                     sentence_types=("past" if base_past else "present",
                                     "past" if donor_past else "present")),
        builder._row(**common, transform_id="A2", construction_id="relative_clause",
                     direction_id=direction, matched_suffix=tail,
                     base_text=_relative(subject, place, focus, base_past),
                     donor_text=_relative(subject, place, focus, donor_past),
                     base_answer=_answer(base_past), donor_answer=_answer(donor_past),
                     sentence_types=("past" if base_past else "present",
                                     "past" if donor_past else "present")),
        builder._row(**common, transform_id="P",
                     construction_id=f"direct_narration_{'past' if base_past else 'present'}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=tail,
                     base_text=_direct(p_subject, place, focus, base_past),
                     donor_text=_direct(p_donor_subject, place, focus, base_past),
                     base_answer=_answer(base_past), donor_answer=_answer(base_past),
                     sentence_types=("past" if base_past else "present",) * 2),
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
        if cells.get((transform, "past_to_present")) != half \
                or cells.get((transform, "present_to_past")) != half:
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
