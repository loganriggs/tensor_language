#!/usr/bin/env python3
"""adjective_preposition_nf_of_at.proud_vs_good -- the same cell with an OF-FREE suffix of the same length.

`Near the lantern the pilot was proud, back then,` obliges ` of`; `... was good, back then,` obliges ` at`.

The family's shared suffix `, of course,` CONTAINS the token ` of`, and 12 of 36 cells in the family have ` of` in
their answer vocabulary -- so a plain induction head predicts ` of` at the readout for free in a third of the family.
`, back then,` is four GPT-2 tokens, exactly like `, of course,`, so the cue-to-readout DISTANCE is held: v473 showed
distance moves these directions on its own, and a three-token replacement would have confounded the echo with it.
Authored 2026-09-10 for the of-echo test; capability screened on CPU first.
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
    task_id="adjective_preposition_nf_of_at.proud_vs_good",
    vocabulary=(' of', ' at'),
    generator_role="generate_linked_adjective_preposition_nf_of_at_fit_panels",
    answer_role="score_jointly_tokenized_of_versus_at",
    a1=bs.Family("bare_frame", "bare_frame_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} was {'proud' if pos else 'good'}, back then,"),
    a2=bs.Family("storm_frame", "storm_frame_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} seemed {'proud' if pos else 'good'}, back then,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} was {'proud' if pos else 'good'}, back then,",
    a1_suffix=lambda i: ", back then,",
    a2_suffix=lambda i: ", back then,",
    directions=('proud_to_good', 'good_to_proud'),
    kinds=('proud', 'good'),
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
