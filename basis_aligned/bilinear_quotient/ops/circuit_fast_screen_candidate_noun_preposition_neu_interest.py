#!/usr/bin/env python3
"""noun_preposition_neu_interest.interest_vs_respect -- the SAME heavy frame with a NEUTRAL distractor.

`The pilot showed interest, despite the ongoing delay, of course,` obliges ` in`;
`The pilot showed respect, despite the ongoing delay, of course,` obliges ` for`.

THIS IS THE CONTROL THAT MAKES THE RECENCY CELL READABLE. In noun_preposition_rec_interest the interposed noun is
the OTHER member of the cue pair, and every row of both A1 and A2 was dropped by the capability screen -- 32 of 32,
on both sides. That is consistent with the model following the nearest noun, but it is EQUALLY consistent with the
frame simply being too heavy: a comma-bracketed `despite the ongoing X` between the cue noun and the readout is a
long interposition this family has never used, and a stimulus the model cannot perform proves nothing about recency.
So this cell holds the frame, the length, the cue pair, the suffix and the readout pair fixed, and changes ONE thing:
the interposed noun is ` delay`, which is neutral -- it is not a member of the cue pair and does not select either
answer. If this screens cleanly while the recency cell collapses, the collapse is attributable to the COMPETING CUE
NOUN and not to the frame. If this collapses too, the frame is the problem and the recency cell says nothing.
That is lesson 2 applied: vary only the thing the conclusion rests on.
Authored 2026-09-10 as the control for rec_interest; screened on CPU.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R, ADJ, OBJ = lex._REPORTERS, lex._ADJECTIVES, lex._OBJECTS
_agent = lambda i: R[i][0]
_alt = lambda i: R[i][1]
_adj = lambda i: ADJ[i]
_adj2 = lambda i: ADJ[(i + 1) % len(ADJ)]
_obj = lambda i: OBJ[i]
_obj2 = lambda i: OBJ[(i + 1) % len(OBJ)]
GENDER = (("king", "queen"), ("father", "mother"), ("brother", "sister"), ("uncle", "aunt"), ("son", "daughter"),
          ("husband", "wife"), ("boy", "girl"), ("man", "woman"), ("prince", "princess"), ("grandfather", "grandmother"),
          ("nephew", "niece"), ("actor", "actress"), ("waiter", "waitress"), ("duke", "duchess"), ("lord", "lady"),
          ("monk", "nun"))
_g = lambda i, male: GENDER[i % len(GENDER)][0 if male else 1]

SPEC = bs.BehaviourSpec(
    task_id="noun_preposition_neu_interest.interest_vs_respect",
    vocabulary=(' in', ' for'),
    generator_role="generate_linked_noun_preposition_neu_interest_fit_panels",
    answer_role="score_jointly_tokenized_in_versus_for",
    a1=bs.Family("distractor_frame", "distractor_frame_noun_swap", lambda i, pos: f"The {_agent(i)} showed {'interest' if pos else 'respect'}, despite the ongoing delay, of course,"),
    a2=bs.Family("dawn_distractor_frame", "dawn_distractor_frame_noun_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} showed {'interest' if pos else 'respect'}, despite the ongoing delay, of course,"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} showed {'interest' if pos else 'respect'}, despite the ongoing delay, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('in_to_for', 'for_to_in'),
    kinds=('interest', 'respect'),
    p_generator_role="object_lexical_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
build_rows, validate_rows, authority_sha256 = SPEC.api()

if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for f in ("A1", "A2", "P", "C"):
        r = next(x for x in rows if x["family"] == f)
        print(f"  {f} base : {r['base_text']!r} -> {r['base_answer']!r}")
        print(f"     donor: {r['donor_text']!r} -> {r['donor_answer']!r}")
