#!/usr/bin/env python3
"""wh_adjunct_when_where.time_vs_place -- does a site carry a fronted adjunct's TYPE (a time adverb -> `when`, a place adverb -> `where`) across the whole clause and a parenthetical to the embedded wh-word?

`Yesterday the pilot lifted the crate, of course, though nobody remembers exactly` obliges ` when`; `Upstairs the pilot lifted the crate, of course, though nobody remembers exactly` obliges ` where`. NEW family (wh readout selected by an antecedent adjunct; wh_argument_selection reads the VERB from a wh-word).

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
    task_id="wh_adjunct_when_where.time_vs_place",
    vocabulary=(' when', ' where'),
    generator_role="generate_linked_wh_adjunct_when_where_fit_panels",
    answer_role="score_jointly_tokenized_when_versus_where",
    a1=bs.Family("bare_frame", "bare_frame_adjunct_swap", lambda i, pos: f"{'Yesterday' if pos else 'Upstairs'} the {_agent(i)} lifted the {_obj(i)}, of course, though nobody remembers exactly"),
    a2=bs.Family("night_frame", "night_frame_adjunct_swap", lambda i, pos: f"{'Last night' if pos else 'Downstairs'} the {_adj(i)} {_agent(i)} lifted it, of course, though nobody remembers exactly"),
    p_donor=lambda i, pos: f"{'Yesterday' if pos else 'Upstairs'} the {_agent(i)} lifted the {_obj2(i)}, of course, though nobody remembers exactly",
    a1_suffix=lambda i: ", of course, though nobody remembers exactly",
    a2_suffix=lambda i: ", of course, though nobody remembers exactly",
    directions=('time_to_place', 'place_to_time'),
    kinds=('time', 'place'),
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
