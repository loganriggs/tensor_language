#!/usr/bin/env python3
"""quantity_change_more_less.rose_vs_fell -- does a site carry a change verb's DIRECTION (the water rose -> `more`, fell -> `less`) across an adverb, a parenthetical and a result clause to the quantity word?

`The water in the tank near the lantern rose overnight, of course, so there was` obliges ` more`; `The water in the tank near the lantern fell overnight, of course, so there was` obliges ` less`. NEW family (quantity from a change-of-level verb).

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
    task_id="quantity_change_more_less.rose_vs_fell",
    vocabulary=(' more', ' less'),
    generator_role="generate_linked_quantity_change_more_less_fit_panels",
    answer_role="score_jointly_tokenized_more_versus_less",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The water in the tank near the {_obj(i)} {'rose' if pos else 'fell'} overnight, of course, so there was"),
    a2=bs.Family("river_frame", "river_frame_verb_swap", lambda i, pos: f"Before dawn the {_adj(i)} river {'rose' if pos else 'fell'}, of course, leaving"),
    p_donor=lambda i, pos: f"The water in the tank near the {_adj2(i)} {_obj(i)} {'rose' if pos else 'fell'} overnight, of course, so there was",
    a1_suffix=lambda i: ", of course, so there was",
    a2_suffix=lambda i: ", of course, leaving",
    directions=('rose_to_fell', 'fell_to_rose'),
    kinds=('rose', 'fell'),
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
