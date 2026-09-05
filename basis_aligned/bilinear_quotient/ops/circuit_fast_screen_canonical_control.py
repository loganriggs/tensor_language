#!/usr/bin/env python3
"""ONE same-answer control, byte-identical across behaviours, so C becomes comparable.

Measured 2026-09-05T11:46Z: swapping only the control frame on `interrogative_licensing` moved
C from 0.230 to 0.141 while the verdict, the selected site, the passing band and P were all
unchanged. So C is substantially a property of the control chosen, and C values from screens
with different controls cannot be compared. That withdrew the dependency-type ordering reported
at 11:33Z.

The fix is a control that does not vary: the SAME rows, the same texts, the same answers, in
every candidate that uses this module. Then a difference in C between two screens is a
difference between the behaviours, because the control was held fixed by construction.

The battery permits this. A row's answer vocabulary is per-row, not per-candidate -- the
construction checks only require base and donor of one row to share their answer/foil pair --
so a control may use a vocabulary the target families do not. `sentence_terminal` already
relies on that: its targets score punctuation while its control scores agreement.

The control is same-answer (both sides answer " is") and varies only place, subject and
reporter, none of which bear on any target variable here. Its ceiling in this configuration is
roughly 0.07-0.23, so it remains a statistic rather than a clause that can fail (standing lesson
4); what it now supports is a COMPARISON between behaviours, which is what it is for.
"""
from __future__ import annotations

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidates as lex

VOCABULARY = (" is", " are")
TRANSFORM = battery.TransformSpec(
    "C", "canonical_same_answer_locative_agreement", False, "registered_active")

_PLACES = ("harbor", "market", "valley", "meadow", "orchard", "canyon", "island", "field",
           "garden", "forest", "bridge", "tower", "cabin", "river", "ocean", "road")
_SUBJECTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATES = tuple(pair[1] for pair in lex._REPORTERS)
_VERBS = ("repaired", "described", "counted", "praised", "sketched", "measured",
          "cleaned", "labelled", "inspected", "recorded", "painted", "moved",
          "carried", "opened", "closed", "lifted", "guarded", "polished",
          "gathered", "sorted", "packed", "weighed", "mapped", "traced",
          "framed", "listed", "checked", "marked", "handled", "stored",
          "shifted", "tested")


def text(case_index: int, primary: bool) -> str:
    """One side of the canonical control row. Identical for a given case across candidates."""
    subject = (_SUBJECTS if primary else _ALTERNATES)[case_index]
    place = _PLACES[(case_index + (0 if primary else 5)) % len(_PLACES)]
    reporter = _SUBJECTS[(case_index + 11) % len(_SUBJECTS)]
    return f"Beside the {place} the {subject} that the {reporter} {_VERBS[case_index]}"


def suffix(case_index: int) -> str:
    """The trailing text both sides share, so the final input token matches."""
    return f" {_VERBS[case_index]}"


def row_kwargs(case_index: int, forward: bool) -> dict:
    """Everything the shared row builder needs for the canonical control row."""
    return {
        "construction_id": "canonical_same_answer_locative_agreement",
        "direction_id": "base_to_donor" if forward else "donor_to_base",
        "matched_suffix": suffix(case_index),
        "base_text": text(case_index, forward),
        "donor_text": text(case_index, not forward),
        "base_answer": VOCABULARY[0],
        "donor_answer": VOCABULARY[0],
        "vocabulary": VOCABULARY,
        "sentence_types": ("canonical_control", "canonical_control"),
    }
