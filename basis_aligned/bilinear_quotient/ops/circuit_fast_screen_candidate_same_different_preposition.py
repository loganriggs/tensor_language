#!/usr/bin/env python3
"""same_different_preposition.same_vs_different -- does a site carry a comparison adjective's preposition selection (the same -> as, a different -> from) across the noun and a parenthetical?

`The pilot near the lantern chose the same crate, of course,` obliges ` as`; `chose a different crate, of course,` obliges ` from`. Adjective-preposition family (fond/similar readouts of/to; this pair reads as/from).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v227 batch).
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
    task_id="same_different_preposition.same_vs_different",
    vocabulary=(' as', ' from'),
    generator_role="generate_linked_same_different_preposition_fit_panels",
    answer_role="score_jointly_tokenized_as_versus_from",
    a1=bs.Family("bare_frame", "bare_frame_adjective_swap", lambda i, pos: f"The {_agent(i)} near the {_obj2(i)} chose {'the same' if pos else 'a different'} {_obj(i)}, of course,"),
    a2=bs.Family("dawn_frame", "dawn_frame_adjective_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} chose {'the same' if pos else 'a different'} {_obj(i)}, of course,"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj2(i)} chose {'the same' if pos else 'a different'} {_obj(i)}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('same_to_different', 'different_to_same'),
    kinds=('same', 'different'),
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
