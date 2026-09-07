#!/usr/bin/env python3
"""comparative_superlative_two_three.two_vs_three -- does a site carry a set's SIZE (two -> comparative `heavier`, three -> superlative `heaviest`) across the set noun, a PP, a parenthetical and the chooser to the adjective degree?

`Of the two crates near the lantern, of course, the pilot chose the` obliges ` heavier`; `Of the three crates near the lantern, of course, the pilot chose the` obliges ` heaviest`. NEW family (comparative vs superlative from cardinality; the degree family reads better/good and best/superior from frames, not from a count).

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
    task_id="comparative_superlative_two_three.two_vs_three",
    vocabulary=(' heavier', ' heaviest'),
    generator_role="generate_linked_comparative_superlative_two_three_fit_panels",
    answer_role="score_jointly_tokenized_heavier_versus_heaviest",
    a1=bs.Family("bare_frame", "bare_frame_count_swap", lambda i, pos: f"Of the {'two' if pos else 'three'} crates near the {_obj(i)}, of course, the {_agent(i)} chose the"),
    a2=bs.Family("between_frame", "between_frame_count_swap", lambda i, pos: f"Between the {'two' if pos else 'three'} {_adj(i)} loads, of course, the {_agent(i)} lifted the"),
    p_donor=lambda i, pos: f"Of the {'two' if pos else 'three'} crates near the {_adj2(i)} {_obj(i)}, of course, the {_agent(i)} chose the",
    a1_suffix=lambda i: f", of course, the {_agent(i)} chose the",
    a2_suffix=lambda i: f", of course, the {_agent(i)} lifted the",
    directions=('two_to_three', 'three_to_two'),
    kinds=('two', 'three'),
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
