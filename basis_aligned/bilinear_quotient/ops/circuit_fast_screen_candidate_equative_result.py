#!/usr/bin/env python3
"""equative_result.as_vs_so -- does a site carry `as` versus `so` across the adjective to its correlate?

`as bright` obliges ` as`; `so bright` obliges ` that`. The sibling of comparative_frame (less/as) and degree_frame (so/too).

    A1  "The lantern was as bright" -> " as" / "The lantern was so bright" -> " that"
    A2  "To the pilot the lantern seemed just as bright" -> " as"

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, the ninth standing lesson).
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
    task_id="equative_result.as_vs_so",
    vocabulary=(" as", " that"),
    generator_role="generate_linked_equative_result_fit_panels",
    answer_role="score_jointly_tokenized_as_versus_that",
    a1=bs.Family("bare_frame", "bare_frame_marker_swap", lambda i, pos: f"The {_obj(i)} was {'as' if pos else 'so'} {_adj(i)}"),
    a2=bs.Family("seem_frame", "seem_frame_marker_swap", lambda i, pos: f"To the {_agent(i)} the {_obj(i)} seemed {'just as' if pos else 'so very'} {_adj(i)}"),
    p_donor=lambda i, pos: f"The {_obj2(i)} was {'as' if pos else 'so'} {_adj(i)}",
    a1_suffix=lambda i: f" {_adj(i)}",
    a2_suffix=lambda i: f" {_adj(i)}",
    directions=("equative_to_result", "result_to_equative"),
    kinds=("equative", "result"),
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
