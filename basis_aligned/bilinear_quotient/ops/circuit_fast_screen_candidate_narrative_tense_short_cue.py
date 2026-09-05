#!/usr/bin/env python3
"""narrative_tense with the tense cue ADJACENT to the prediction, to test the instrument.

Five behaviours I have screened (bracket_positive_control, sentence_terminal, quote_parity,
narrative_tense, correlative_state) all select resid:18 and pass ONLY residual sites -- no
attention or module site passes in any of them. The numeral behaviours (numbered_list,
numeric_sequence, task14) DO pass attention sites, strongest at attn:08.

There is an alternative to "these variables are carried only late". **The instrument patches
only the final input token.** A head that writes the state at an EARLIER position and lets it
flow forward in the residual stream is invisible to it: by the final position the work is done
and only the residual carries it. Every one of my five stimuli places a long shared tail between
the cue and the prediction; the numeral stimuli require the value to be read AT the prediction
point.

So this varies exactly one thing. A1, A2 and P lose the long tail, putting the cue two to four
tokens before the patched position:

    v2 long cue   "Last winter the leader stood nearby. The main reason for the short window" -> " was"
    this          "Last winter the leader"                                                    -> " was"
                  "Every winter the leader"                                                   -> " is"
    A2            "The record from last winter says the leader"                               -> " was"
                  "The record from this winter says the leader"                               -> " is"

The control is held BYTE-IDENTICAL to v2's, so the only difference between the two screens is
cue distance in the target families (standing lesson 2).

Registered prediction. If the resid-only profile is a fact about the behaviour, this screen
passes only residual sites too. If it is a property of long cue distance, attention sites should
now pass -- and the localization claims for all five behaviours would need restating.

Code path. Reuses the row builder pinned by
`circuit_fast_screen_candidate_sentence_terminal_context_control` (digest d0da3cda...), and the
control text is imported from the narrative_tense module rather than retyped.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_narrative_tense as long_cue
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_candidates as lex

canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED

TASK_ID = "narrative_tense.short_cue_distance"
TENSE = long_cue.TENSE

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_short_cue_narrative_tense_fit_panels",
    answer_role="score_jointly_tokenized_past_versus_present_copula",
    transforms=(
        battery.TransformSpec("A1", "adjacent_cue_tense_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_frame_adjacent_cue_tense_swap", True, "toward_donor"),
        battery.TransformSpec("P", "subject_lexical_rewrite", False, "invariant"),
        battery.TransformSpec("C", "past_description_same_answer_rewrite", False,
                              "registered_active"),
    ),
)

_SUBJECTS = long_cue._SUBJECTS
_ALTERNATES = long_cue._ALTERNATES
_PLACES = long_cue._PLACES
_FOCUS = long_cue._FOCUS


def _adjacent(subject: str, past: bool) -> str:
    return f"{'Last' if past else 'Every'} winter the {subject}"


def _season(subject: str, past: bool, primary: bool) -> str:
    """The answer-preserving edit, rewriting the SEASON rather than the subject.

    In the long-cue design P swapped the subject noun, but here the subject IS the final input
    token, so swapping it breaks the matched-final-token invariant that makes base and donor
    comparable. The season is the only lexical slot before it that leaves the tense untouched.
    """
    season = "winter" if primary else "autumn"
    return f"{'Last' if past else 'Every'} {season} the {subject}"


def _report(subject: str, past: bool) -> str:
    """PP-fronted attribution, a different construction from A1's bare adverbial.

    v1 used "The record from last/this winter says the ..." and A2 failed native capability at
    40/64 with a margin of +0.25: "this winter" is tense-ambiguous in English -- it reads as
    future as readily as present -- so the model would not commit to " is". A1 passed at 64/64
    (+2.42) on the same cue distance, so the short-cue design was sound and only this phrasing
    was weak. "years ago" versus "today" carries no such ambiguity.
    """
    return f"According to reports from {'years ago' if past else 'today'}, the {subject}"


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    subject, alternate = _SUBJECTS[case_index], _ALTERNATES[case_index]
    place, focus = _PLACES[case_index], _FOCUS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_past, donor_past = (True, False) if forward else (False, True)
    direction = "past_to_present" if forward else "present_to_past"
    common = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=subject, alternate_reporter=alternate, adjective=focus,
                  object_name=place, vocabulary=TENSE, spec=TASK_SPEC)
    p_subject, p_donor_subject = (subject, alternate) if forward else (alternate, subject)
    kinds = ("past" if base_past else "present", "past" if donor_past else "present")
    answer = long_cue._answer
    return [
        builder._row(**common, transform_id="A1", construction_id="adjacent_cue",
                     direction_id=direction, matched_suffix=f"the {subject}",
                     base_text=_adjacent(subject, base_past),
                     donor_text=_adjacent(subject, donor_past),
                     base_answer=answer(base_past), donor_answer=answer(donor_past),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="report_frame_adjacent_cue",
                     direction_id=direction, matched_suffix=f"the {subject}",
                     base_text=_report(subject, base_past),
                     donor_text=_report(subject, donor_past),
                     base_answer=answer(base_past), donor_answer=answer(donor_past),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"adjacent_cue_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=f"the {subject}",
                     base_text=_season(subject, base_past, forward),
                     donor_text=_season(subject, base_past, not forward),
                     base_answer=answer(base_past), donor_answer=answer(base_past),
                     sentence_types=(kinds[0], kinds[0])),
        # Control held byte-identical to narrative_tense v2.
        builder._row(**common, transform_id="C",
                     construction_id="past_description_same_answer_rewrite",
                     direction_id="base_to_donor" if forward else "donor_to_base",
                     matched_suffix=f"{focus} {place}",
                     base_text=long_cue._control(subject if forward else alternate, place, focus, 0),
                     donor_text=long_cue._control(alternate if forward else subject, place, focus, 0),
                     base_answer=TENSE[0], donor_answer=TENSE[0],
                     sentence_types=("past", "past")),
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
        print(f"  {family}  base: {r['base_text']!r} -> {r['base_answer']!r}")
        print(f"     donor: {r['donor_text']!r} -> {r['donor_answer']!r}")
