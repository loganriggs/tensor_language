#!/usr/bin/env python3
"""category_selection_during_while.during_vs_while -- does a site carry a connective's CATEGORY (preposition during -> NP `the`, conjunction while -> clause `it`) across a parenthetical to the first token of its complement?

`The pilot near the lantern slept soundly during, of course,` obliges ` the`; `The pilot near the lantern slept soundly while, of course,` obliges ` it`. NEW family (complement-category selection by a connective: NP vs clause), NEW readout token pair the/it.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v233 batch).
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
    task_id="category_selection_during_while.during_vs_while",
    vocabulary=(' the', ' it'),
    generator_role="generate_linked_category_selection_during_while_fit_panels",
    answer_role="score_jointly_tokenized_the_versus_it",
    a1=bs.Family("bare_frame", "bare_frame_connective_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} slept soundly {'during' if pos else 'while'}, of course,"),
    a2=bs.Family("because_frame", "because_frame_connective_swap", lambda i, pos: f"Because the {_adj(i)} {_agent(i)} kept working {'during' if pos else 'while'}, of course,"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} slept soundly {'during' if pos else 'while'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('preposition_to_conjunction', 'conjunction_to_preposition'),
    kinds=('preposition', 'conjunction'),
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
