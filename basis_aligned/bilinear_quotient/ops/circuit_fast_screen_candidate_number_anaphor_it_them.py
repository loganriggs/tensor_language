#!/usr/bin/env python3
"""number_anaphor_it_them.crate_vs_crates -- does a site carry an object noun's NUMBER (crate -> `it`, crates -> `them`) across a parenthetical and a second verb phrase to the object pronoun?

`Near the lantern the pilot kept the crate, of course, and later sold` obliges ` it`; `Near the lantern the pilot kept the crates, of course, and later sold` obliges ` them`. Number family; NEW readout pair it/them (one/it and us/them exist).

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
    task_id="number_anaphor_it_them.crate_vs_crates",
    vocabulary=(' it', ' them'),
    generator_role="generate_linked_number_anaphor_it_them_fit_panels",
    answer_role="score_jointly_tokenized_it_versus_them",
    a1=bs.Family("bare_frame", "bare_frame_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} kept the {'crate' if pos else 'crates'}, of course, and later sold"),
    a2=bs.Family("because_frame", "because_frame_swap", lambda i, pos: f"Because the {_adj(i)} {_agent(i)} lifted the {'crate' if pos else 'crates'}, of course, nobody else touched"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} kept the {'crate' if pos else 'crates'}, of course, and later sold",
    a1_suffix=lambda i: ", of course, and later sold",
    a2_suffix=lambda i: ", of course, nobody else touched",
    directions=('singular_to_plural', 'plural_to_singular'),
    kinds=('singular', 'plural'),
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
