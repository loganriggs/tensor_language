#!/usr/bin/env python3
"""person_agreement_be.i_vs_you -- does a site carry a subject pronoun's PERSON (I vs you) past an adverb to the agreeing copula (am vs are)?

`As the pilot knows, I surely` obliges ` am`; `you surely` obliges ` are`. Person-readout sibling of reflexive_person with a verb-agreement answer instead of a reflexive.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v213 batch).
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
    task_id="person_agreement_be.i_vs_you",
    vocabulary=(' am', ' are'),
    generator_role="generate_linked_person_agreement_be_fit_panels",
    answer_role="score_jointly_tokenized_am_versus_are",
    a1=bs.Family("knows_frame", "knows_frame_person_swap", lambda i, pos: f"As the {_agent(i)} knows, {'I' if pos else 'you'} surely"),
    a2=bs.Family("near_frame", "near_frame_person_swap", lambda i, pos: f"Near the {_obj(i)} {'I' if pos else 'you'} surely"),
    p_donor=lambda i, pos: f"As the {_alt(i)} knows, {'I' if pos else 'you'} surely",
    a1_suffix=lambda i: " surely",
    a2_suffix=lambda i: " surely",
    directions=('first_to_second', 'second_to_first'),
    kinds=('first_person', 'second_person'),
    p_generator_role="agent_lexical_rewrite",
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
