#!/usr/bin/env python3
"""correlative_open_neu.both_vs_neither -- the SAME long frame with NO correlative interposed.

`The pilot praised both the crate beneath the wooden shelf` obliges ` and`; with `neither`, ` nor`.

THIS IS THE CONTROL THAT MAKES THE RECENCY CELL READABLE. correlative_open_rec interposes a discharged `either ... or`
between the live correlative and the readout; if its rows collapse, that is only evidence about recency if the same
long interposition WITHOUT a correlative is performable. This cell holds the frame shape, the cue pair, the readout
pair and the construction fixed, and removes the correlative from the interposed material.
If this screens cleanly while rec collapses, the collapse is attributable to the DISCHARGED CORRELATIVE and not to
the distance or the heavier frame. If both collapse, the frame is the problem and rec says nothing. That is lesson 2
applied, and it is the same pairing that made v489 readable.
Authored 2026-09-10 as the control for correlative_open_rec.
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
    task_id="correlative_open_neu.both_vs_neither",
    vocabulary=(' and', ' nor'),
    generator_role="generate_linked_correlative_open_neu_fit_panels",
    answer_role="score_jointly_tokenized_and_versus_nor",
    a1=bs.Family("open_frame", "open_frame_correlative_swap", lambda i, pos: f"The {_agent(i)} praised {'both' if pos else 'neither'} the crate beneath the wooden shelf"),
    a2=bs.Family("noted_frame", "noted_frame_correlative_swap", lambda i, pos: f"In the notes the {_agent(i)} named {'both' if pos else 'neither'} the crate beneath the wooden shelf"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} praised {'both' if pos else 'neither'} the crate beneath the wooden shelf",
    a1_suffix=lambda i: " the crate beneath the wooden shelf",
    a2_suffix=lambda i: " the crate beneath the wooden shelf",
    directions=('both_to_neither', 'neither_to_both'),
    kinds=('both', 'neither'),
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
