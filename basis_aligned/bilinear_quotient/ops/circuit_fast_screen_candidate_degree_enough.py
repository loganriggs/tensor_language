#!/usr/bin/env python3
"""degree_enough.too_vs_barely -- does a site carry `too` vs `barely` to the degree complement (`to` vs `enough`)?

`The lantern was far too bright` obliges ` to`; `barely bright` obliges ` enough`. Sibling of degree_result (far too / really so -> to / that): same `too` side, a different contrast partner.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v207 batch).
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
    task_id="degree_enough.too_vs_barely",
    vocabulary=(' to', ' enough'),
    generator_role="generate_linked_degree_enough_fit_panels",
    answer_role="score_jointly_tokenized_to_versus_enough",
    a1=bs.Family("bare_frame", "bare_frame_degree_swap", lambda i, pos: f"The {_obj(i)} was {'far too' if pos else 'barely'} {_adj(i)}"),
    a2=bs.Family("notes_frame", "notes_frame_degree_swap", lambda i, pos: f"In the notes the {_obj(i)} seemed {'far too' if pos else 'barely'} {_adj2(i)}"),
    p_donor=lambda i, pos: f"The {_obj2(i)} was {'far too' if pos else 'barely'} {_adj(i)}",
    a1_suffix=lambda i: f" {_adj(i)}",
    a2_suffix=lambda i: f" {_adj2(i)}",
    directions=('too_to_barely', 'barely_to_too'),
    kinds=('excess', 'insufficiency'),
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
