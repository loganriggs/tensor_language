#!/usr/bin/env python3
"""measure_wh_far_long.miles_vs_hours -- does a site carry a measure noun's DIMENSION (miles -> `far`, hours -> `long`) across a parenthetical to the wh-measure word after `how`?

`The pilot near the lantern walked ten miles, of course, though nobody asked how` obliges ` far`; `... walked ten hours ...` obliges ` long`. NEW family (measure-dimension selection of the degree wh-word).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v237 batch).
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
    task_id="measure_wh_far_long.miles_vs_hours",
    vocabulary=(' far', ' long'),
    generator_role="generate_linked_measure_wh_far_long_fit_panels",
    answer_role="score_jointly_tokenized_far_versus_long",
    a1=bs.Family("bare_frame", "bare_frame_measure_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} walked ten {'miles' if pos else 'hours'}, of course, though nobody asked how"),
    a2=bs.Family("later_frame", "later_frame_measure_swap", lambda i, pos: f"Ten {'miles' if pos else 'hours'} later the {_adj(i)} {_agent(i)} wondered, of course, how"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} walked ten {'miles' if pos else 'hours'}, of course, though nobody asked how",
    a1_suffix=lambda i: ", of course, though nobody asked how",
    a2_suffix=lambda i: f" later the {_adj(i)} {_agent(i)} wondered, of course, how",
    directions=('distance_to_time', 'time_to_distance'),
    kinds=('distance', 'time'),
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
