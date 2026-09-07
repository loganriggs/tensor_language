#!/usr/bin/env python3
"""vertical_deixis_up_down.roof_vs_cellar -- does a site carry an object's LOCATION (on the roof -> carried `down`, in the cellar -> carried `up`) across a clause boundary and a parenthetical to the directional particle?

`The crate near the lantern was on the roof, so the pilot carried it, of course,` obliges ` down`; `The crate near the lantern was in the cellar, so the pilot carried it, of course,` obliges ` up`. NEW family (vertical deixis; the old `deixis` behaviour read here/there from came/went).

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
    task_id="vertical_deixis_up_down.roof_vs_cellar",
    vocabulary=(' down', ' up'),
    generator_role="generate_linked_vertical_deixis_up_down_fit_panels",
    answer_role="score_jointly_tokenized_down_versus_up",
    a1=bs.Family("bare_frame", "bare_frame_location_swap", lambda i, pos: f"The crate near the {_obj(i)} was {'on the roof' if pos else 'in the cellar'}, so the {_agent(i)} carried it, of course,"),
    a2=bs.Family("because_frame", "because_frame_location_swap", lambda i, pos: f"Because the {_adj(i)} crate sat {'on the roof' if pos else 'in the cellar'}, the {_agent(i)} brought it, of course,"),
    p_donor=lambda i, pos: f"The crate near the {_adj2(i)} {_obj(i)} was {'on the roof' if pos else 'in the cellar'}, so the {_agent(i)} carried it, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('roof_to_cellar', 'cellar_to_roof'),
    kinds=('roof', 'cellar'),
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
