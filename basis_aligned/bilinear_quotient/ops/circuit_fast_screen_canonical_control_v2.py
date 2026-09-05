#!/usr/bin/env python3
"""Canonical same-answer control, v2: a vocabulary disjoint from every target in the corpus.

v1 (`circuit_fast_screen_canonical_control.py`) answers " is". Measured 2026-09-05T12:47Z, that
made it non-neutral for `narrative_tense`, whose targets score " was"/" is": across five
behaviours under v1, the four with vocabularies DISJOINT from the control spanned C 0.043-0.087
while narrative_tense -- the only one sharing a token with the control -- sat at 0.134. A control
that shares an answer token with the behaviour it is controlling is measuring the behaviour.

v1 is left byte-stable so the five screens already run against it stay reproducible; this is a
separate control, used alongside.

v2 answers " night" against the foil " day". Neither appears in any target vocabulary in the
corpus:

    " than"/" as"    " whether"/" that"    " or"/" nor"    " any"/" some"
    " was"/" is"     "."/"?"               '"'             " he"/" she"

The frame is a strongly-preferred completion ("in the middle of the night"), so the control
clears its native-capability bar comfortably, and both sides end on the same ` the` token.

Registered prediction for the pair of screens this was built for: narrative_tense under v2 falls
below 0.09, into the range of the disjoint behaviours, if the v1 spread was the token overlap;
staying above 0.12 would mean a real behaviour effect that survives a disjoint control.
"""
from __future__ import annotations

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidates as lex

VOCABULARY = (" night", " day")
TRANSFORM = battery.TransformSpec(
    "C", "canonical_same_answer_nocturnal_completion", False, "registered_active")

_PLACES = ("harbor", "market", "valley", "meadow", "orchard", "canyon", "island", "field",
           "garden", "forest", "bridge", "tower", "cabin", "river", "ocean", "road")
_SUBJECTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATES = tuple(pair[1] for pair in lex._REPORTERS)
_TASKS = ("work", "task", "report", "survey", "repair", "record", "sketch", "count",
          "review", "notice", "letter", "message", "drawing", "list", "plan", "study")


def text(case_index: int, primary: bool) -> str:
    subject = (_SUBJECTS if primary else _ALTERNATES)[case_index]
    place = _PLACES[(case_index + (0 if primary else 5)) % len(_PLACES)]
    task = _TASKS[(case_index + (0 if primary else 3)) % len(_TASKS)]
    verb = "finished" if primary else "completed"
    where = "Beside" if primary else "Inside"
    return f"{where} the {place} the {subject} {verb} the {task} in the middle of the"


def suffix() -> str:
    return "in the middle of the"


def row_kwargs(case_index: int, forward: bool) -> dict:
    return {
        "construction_id": "canonical_same_answer_nocturnal_completion",
        "direction_id": "base_to_donor" if forward else "donor_to_base",
        "matched_suffix": suffix(),
        "base_text": text(case_index, forward),
        "donor_text": text(case_index, not forward),
        "base_answer": VOCABULARY[0],
        "donor_answer": VOCABULARY[0],
        "vocabulary": VOCABULARY,
        "sentence_types": ("canonical_control_v2", "canonical_control_v2"),
    }
