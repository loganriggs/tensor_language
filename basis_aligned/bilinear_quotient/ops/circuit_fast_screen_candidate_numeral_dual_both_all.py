#!/usr/bin/env python3
"""numeral_dual_both_all.two_vs_three -- does a site carry a numeral's DUALITY (two -> `both`, three -> `all`) across a noun phrase, a PP and a parenthetical to the universal quantifier?

`Of the two crates near the lantern, the pilot lifted, of course,` obliges ` both`; `Of the three crates near the lantern, the pilot lifted, of course,` obliges ` all`. NEW family (dual vs plural numeral selects the quantifier).

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
    task_id="numeral_dual_both_all.two_vs_three",
    vocabulary=(' both', ' all'),
    generator_role="generate_linked_numeral_dual_both_all_fit_panels",
    answer_role="score_jointly_tokenized_both_versus_all",
    a1=bs.Family("bare_frame", "bare_frame_numeral_swap", lambda i, pos: f"Of the {'two' if pos else 'three'} crates near the {_obj(i)}, the {_agent(i)} lifted, of course,"),
    a2=bs.Family("saw_frame", "saw_frame_numeral_swap", lambda i, pos: f"The {_adj(i)} {_agent(i)} saw {'two' if pos else 'four'} {_alt(i)}s, of course, and greeted"),
    p_donor=lambda i, pos: f"Of the {'two' if pos else 'three'} crates near the {_adj2(i)} {_obj(i)}, the {_agent(i)} lifted, of course,",
    a1_suffix=lambda i: f", the {_agent(i)} lifted, of course,",
    a2_suffix=lambda i: f" {_alt(i)}s, of course, and greeted",
    directions=('dual_to_plural', 'plural_to_dual'),
    kinds=('dual', 'plural'),
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
