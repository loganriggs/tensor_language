#!/usr/bin/env python3
"""predicate_category_slow_slowly.seemed_vs_moved -- does a site carry a verb's CATEGORY SELECTION (seemed -> adjective `slow`, moved -> adverb `slowly`) across a parenthetical and a degree word to the predicate?

`The pilot near the lantern seemed, of course, rather` obliges ` slow`; `The pilot near the lantern moved, of course, rather` obliges ` slowly`. Predicate-category family (quiet/quietly after feel/seem, well/good exist), NEW readout token pair slow/slowly.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v239 batch).
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
    task_id="predicate_category_slow_slowly.seemed_vs_moved",
    vocabulary=(' slow', ' slowly'),
    generator_role="generate_linked_predicate_category_slow_slowly_fit_panels",
    answer_role="score_jointly_tokenized_slow_versus_slowly",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} {'seemed' if pos else 'moved'}, of course, rather"),
    a2=bs.Family("to_frame", "to_frame_verb_swap", lambda i, pos: f"To the {_adj(i)} {_alt(i)} the {_agent(i)} {'looked' if pos else 'worked'}, of course, quite"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} {'seemed' if pos else 'moved'}, of course, rather",
    a1_suffix=lambda i: ", of course, rather",
    a2_suffix=lambda i: ", of course, quite",
    directions=('copular_to_lexical', 'lexical_to_copular'),
    kinds=('copular', 'lexical'),
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
