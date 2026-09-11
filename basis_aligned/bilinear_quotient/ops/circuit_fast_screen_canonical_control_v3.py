#!/usr/bin/env python3
"""Canonical same-answer control, v3: a SECOND control, so selectivity is not measured against one behaviour.

WHY THIS EXISTS. I scanned 88 spec-authored candidate cells on 2026-09-11: every one draws its C family from
`canonical_same_answer_nocturnal_completion` (v2, answers " night"). Today's rungs made row 4 -- the C upper bound --
the general constraint on countable behaviours: it binds at 2/6 in noun_preposition, 3/8 across eight
non-prepositional constructions and 1/5 under direction-matched fits, while row 2 turned out to be a prepositional
problem (7/8 pass it outside the preposition families against 9/33 inside). So the scarce property is SELECTIVITY,
and selectivity is currently measured against a single nocturnal completion for the whole corpus.

v2 IS NOT BROKEN AND THIS DOES NOT REPLACE IT. v2 was itself built to fix a real v1 defect (v1 answered " is" and
shared a token with narrative_tense, which scored 0.134 against 0.043-0.087 for the disjoint behaviours), and its
vocabulary is disjoint from every target in the corpus. It stays the registered control. The gap this fills is that
one control cannot distinguish "this site is specific" from "this site happens not to disturb nocturnal completion".

WHAT IS VARIED, AND WHAT IS HELD. Held: same-answer design (both sides answer the same token), a strongly-preferred
completion so the control clears its own capability bar, a vocabulary disjoint from every target vocabulary in the
corpus, and both sides ending on the SAME final input token -- all four are the properties that make v2 a usable
control, and changing them would make a disagreement uninterpretable. Varied: the FRAME and the completion. v2 is a
locative-adjunct frame ending "in the middle of the" -> " night"; v3 is a temporal-subordinate frame ending
"for the rest of the" -> " week". Standing lesson 3 says a control drawn from a related behaviour masks a real site
and that "related" includes POSITION, so the point of a second control is a different position and a different
completion, not a different wording of the same one.

VOCABULARY DISJOINTNESS, checked against the target vocabularies v2 lists: " than"/" as", " whether"/" that",
" or"/" nor", " any"/" some", " was"/" is", "."/"?", '"', " he"/" she" -- and against the preposition and particle
readouts this corpus uses (" at", " to", " of", " in", " on", " for", " with", " about", " amid", " from", " by",
" into", " toward", " beyond", " under", " within", " against", " up", " down", " out", " away", " over", " past",
" across", " beneath", " between", " among", " through"). Neither " week" nor its foil " year" appears in any of them.

REGISTERED PREDICTION for the rung that uses it: if row-4 verdicts under v3 agree with v2 on the cells already
measured, row 4 is reporting the SITE and the corpus's selectivity claims stand as stated. If they disagree, every
row-4 verdict in the tier protocol is control-specific, and that has to be said before more behaviours are counted
on it. Disagreement in EITHER direction is informative; agreement is the outcome I expect and would be glad to get.
"""
from __future__ import annotations

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidates as lex

VOCABULARY = (" week", " year")
TRANSFORM = battery.TransformSpec(
    "C", "canonical_same_answer_weekly_completion", False, "registered_active")

_SETTINGS = ("workshop", "station", "gallery", "pantry", "quarry", "stable", "chapel", "depot",
             "lodge", "mill", "dock", "vault", "barn", "study", "porch", "yard")
_SUBJECTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATES = tuple(pair[1] for pair in lex._REPORTERS)
_OBJECTS = ("ledger", "bundle", "crate", "sample", "notice", "permit", "banner", "parcel",
            "cable", "folder", "ticket", "handle", "basket", "poster", "wagon", "kettle")


def text(case_index: int, primary: bool) -> str:
    subject = (_SUBJECTS if primary else _ALTERNATES)[case_index]
    setting = _SETTINGS[(case_index + (0 if primary else 5)) % len(_SETTINGS)]
    thing = _OBJECTS[(case_index + (0 if primary else 3)) % len(_OBJECTS)]
    verb = "guarded" if primary else "minded"
    when = "Once" if primary else "After"
    return f"{when} the {setting} reopened the {subject} {verb} the {thing} for the rest of the"


def suffix() -> str:
    return "for the rest of the"


def row_kwargs(case_index: int, forward: bool) -> dict:
    return {
        "construction_id": "canonical_same_answer_weekly_completion",
        "direction_id": "base_to_donor" if forward else "donor_to_base",
        "matched_suffix": suffix(),
        "base_text": text(case_index, forward),
        "donor_text": text(case_index, not forward),
        "base_answer": VOCABULARY[0],
        "donor_answer": VOCABULARY[0],
        "vocabulary": VOCABULARY,
        "sentence_types": ("canonical_control_v3", "canonical_control_v3"),
    }
