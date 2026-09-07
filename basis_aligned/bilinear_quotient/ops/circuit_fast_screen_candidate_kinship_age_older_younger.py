#!/usr/bin/env python3
"""kinship_age_older_younger.grandfather_vs_grandson -- does a site carry a kinship noun's GENERATION (grandfather -> `older`, grandson -> `younger`) across a PP, a copula and a parenthetical to the age comparative?

`The pilot's grandfather near the lantern was, of course, much` obliges ` older`; `The pilot's grandson near the lantern was, of course, much` obliges ` younger`. NEW family (relative age from a kinship term; the gender family reads he/she, his/her from the same kind of noun).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v229 batch).
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
    task_id="kinship_age_older_younger.grandfather_vs_grandson",
    vocabulary=(' older', ' younger'),
    generator_role="generate_linked_kinship_age_older_younger_fit_panels",
    answer_role="score_jointly_tokenized_older_versus_younger",
    a1=bs.Family("bare_frame", "bare_frame_kin_swap", lambda i, pos: f"The {_agent(i)}'s {'grandfather' if pos else 'grandson'} near the {_obj(i)} was, of course, much"),
    a2=bs.Family("met_frame", "met_frame_kin_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} met a {'grandfather' if pos else 'grandson'} who was, of course, far"),
    p_donor=lambda i, pos: f"The {_agent(i)}'s {'grandfather' if pos else 'grandson'} near the {_adj2(i)} {_obj(i)} was, of course, much",
    a1_suffix=lambda i: " was, of course, much",
    a2_suffix=lambda i: " who was, of course, far",
    directions=('grandfather_to_grandson', 'grandson_to_grandfather'),
    kinds=('grandfather', 'grandson'),
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
