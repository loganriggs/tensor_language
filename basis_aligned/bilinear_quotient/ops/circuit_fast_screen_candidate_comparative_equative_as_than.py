#!/usr/bin/env python3
"""comparative_equative_as_than.as_vs_more -- does a site carry a DEGREE MARKER'S construction (as careful -> `as`, more careful -> `than`) across an adjective and a parenthetical to the comparative complement?

`Near the lantern the pilot was as careful, of course,` obliges ` as`; `Near the lantern the pilot was more careful, of course,` obliges ` than`. Degree family (comparative_complement_superior/inferior counted); NEW pair as/than.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v249 batch).
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
    task_id="comparative_equative_as_than.as_vs_more",
    vocabulary=(' as', ' than'),
    generator_role="generate_linked_comparative_equative_as_than_fit_panels",
    answer_role="score_jointly_tokenized_as_versus_than",
    a1=bs.Family("bare_frame", "bare_frame_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} was {'as' if pos else 'more'} careful, of course,"),
    a2=bs.Family("storm_frame", "storm_frame_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} seemed {'as' if pos else 'more'} patient, of course,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} was {'as' if pos else 'more'} careful, of course,",
    a1_suffix=lambda i: " careful, of course,",
    a2_suffix=lambda i: " patient, of course,",
    directions=('equative_to_comparative', 'comparative_to_equative'),
    kinds=('equative', 'comparative'),
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
