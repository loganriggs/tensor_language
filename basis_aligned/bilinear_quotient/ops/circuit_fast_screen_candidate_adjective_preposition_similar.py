#!/usr/bin/env python3
"""adjective_preposition_similar.similar_vs_different -- does a site carry an adjective's PREPOSITION selection (similar -> to, different -> from) across `in size` to the preposition?

`The crate near the lantern was similar in size` obliges ` to`; `different in size` obliges ` from`. NEW family (adjective-selected preposition).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v217 batch).
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
    task_id="adjective_preposition_similar.similar_vs_different",
    vocabulary=(' to', ' from'),
    generator_role="generate_linked_adjective_preposition_similar_fit_panels",
    answer_role="score_jointly_tokenized_to_versus_from",
    a1=bs.Family("bare_frame", "bare_frame_adj_swap", lambda i, pos: f"The {_obj(i)} near the {_obj2(i)} was {'similar' if pos else 'different'} in size"),
    a2=bs.Family("notes_frame", "notes_frame_adj_swap", lambda i, pos: f"In the notes the {_adj(i)} {_obj(i)} seemed {'similar' if pos else 'different'} in size"),
    p_donor=lambda i, pos: f"The {_obj(i)} near the {_adj2(i)} {_obj2(i)} was {'similar' if pos else 'different'} in size",
    a1_suffix=lambda i: " in size",
    a2_suffix=lambda i: " in size",
    directions=('to_to_from', 'from_to_to'),
    kinds=('similar', 'different'),
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
