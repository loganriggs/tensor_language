#!/usr/bin/env python3
"""person_agreement_am_is.I_vs_he -- does a site carry a subject pronoun's PERSON (I -> `am`, he -> `is`) across a parenthetical to the copula?

`Near the lantern I, of course,` obliges ` am`; `Near the lantern he, of course,` obliges ` is`. Person family; person_agreement_be (I/you -> am/are) missed row 4 by 0.0012 at v213 -- this is a fresh frame with a first-vs-third cue and a NEW readout token pair am/is.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v241 batch).
Capability is checked on CPU before the battery: rows where the donor does not beat the base on the
donor axis are dropped by `g.prepare(valid_only=True)` and counted as `n_dropped` in the receipt.
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
    task_id="person_agreement_am_is.I_vs_he",
    vocabulary=(' am', ' is'),
    generator_role="generate_linked_person_agreement_am_is_fit_panels",
    answer_role="score_jointly_tokenized_am_versus_is",
    a1=bs.Family("bare_frame", "bare_frame_pronoun_swap", lambda i, pos: f"Near the {_obj(i)} {'I' if pos else 'he'}, of course,"),
    a2=bs.Family("because_frame", "because_frame_pronoun_swap", lambda i, pos: f"Because of the {_adj(i)} {_agent(i)}, {'I' if pos else 'he'}, of course,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} {'I' if pos else 'he'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('first_to_third', 'third_to_first'),
    kinds=('first', 'third'),
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
