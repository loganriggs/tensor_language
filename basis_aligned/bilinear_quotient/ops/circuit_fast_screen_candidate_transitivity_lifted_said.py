#!/usr/bin/env python3
"""transitivity_lifted_said.lifted_vs_said -- does a site carry a verb's SUBCATEGORIZATION (a transitive action verb -> a noun-phrase object, a speech verb -> a that-clause) across a parenthetical?

`The pilot near the lantern lifted, of course,` obliges ` the`; `The pilot near the lantern said, of course,` obliges ` that`. NEW family (transitivity: object vs clause after the verb). REPAIR of transitivity_told_said (told/said: A1 26 of 32 rows dropped -- `told, of course, the` is not preferred over `that` across the parenthetical), the one repair a one-sided capability failure licenses.

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
    task_id="transitivity_lifted_said.lifted_vs_said",
    vocabulary=(' the', ' that'),
    generator_role="generate_linked_transitivity_lifted_said_fit_panels",
    answer_role="score_jointly_tokenized_the_versus_that",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} {'lifted' if pos else 'said'}, of course,"),
    a2=bs.Family("dawn_frame", "dawn_frame_verb_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} {'lifted' if pos else 'said'}, of course,"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_obj2(i)} {'lifted' if pos else 'said'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('object_to_clause', 'clause_to_object'),
    kinds=('lifted', 'said'),
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
