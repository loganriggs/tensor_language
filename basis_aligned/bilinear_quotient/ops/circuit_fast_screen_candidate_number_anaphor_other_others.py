#!/usr/bin/env python3
"""number_anaphor_other_others.this_vs_these -- does a site carry a demonstrative object's NUMBER (this -> `other`, these -> `others`) across a parenthetical and a second verb phrase to the contrastive anaphor?

`Near the lantern the pilot kept this, of course, and sold the` obliges ` other`; `Near the lantern the pilot kept these, of course, and sold the` obliges ` others`. A2 REPAIRED once (`Because the pilot lifted this, of course, and not the` read -0.41/+0.40 on CPU -> `The pilot chose this, of course, rather than the`). Number family; NEW readout pair other/others (numeral_other_others with an arithmetic cue was fully incapable and dropped at v241; this cue is a bare demonstrative).

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
    task_id="number_anaphor_other_others.this_vs_these",
    vocabulary=(' other', ' others'),
    generator_role="generate_linked_number_anaphor_other_others_fit_panels",
    answer_role="score_jointly_tokenized_other_versus_others",
    a1=bs.Family("bare_frame", "bare_frame_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} kept {'this' if pos else 'these'}, of course, and sold the"),
    a2=bs.Family("chose_frame", "chose_frame_swap", lambda i, pos: f"The {_adj(i)} {_agent(i)} chose {'this' if pos else 'these'}, of course, rather than the"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} kept {'this' if pos else 'these'}, of course, and sold the",
    a1_suffix=lambda i: ", of course, and sold the",
    a2_suffix=lambda i: ", of course, rather than the",
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
