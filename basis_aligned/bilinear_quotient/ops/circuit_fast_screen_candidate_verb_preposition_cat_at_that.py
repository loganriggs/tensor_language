#!/usr/bin/env python3
"""verb_preposition_cat_at_that.stared_vs_remarked -- a readout pair that crosses a CATEGORY boundary.

`Near the lantern the pilot stared, of course,` obliges ` at`; `... remarked, of course,` obliges ` that`.

An adversarial audit reported that all 119 readout pairs in this family are preposition-vs-preposition, drawn from a
closed set of about twenty single tokens, and that there is no cell anywhere in which the alternative is a different
syntactic category or in which the sentence could legally end. So the fitted rank-1 direction has only ever had to
separate two tokens that are neighbours in a preposition sub-space; it could be an output-side lexical axis over
that class and never encode "a PP is obliged here", and it has never been tested where the distinction matters.
Here the two answers are a PREPOSITION and a COMPLEMENTIZER: `stared` resists a that-clause and `remarked` resists a
PP, so the contrast is categorial rather than gradient, and a within-preposition output axis cannot represent ` that`
at all.
Authored 2026-09-10 from the audit. The audit rated this MODERATE capability risk -- ` at` after `stared, of course,`
competes with several continuations -- so a capability failure is a registered outcome here rather than a surprise,
and is screened on CPU before any battery.
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
    task_id="verb_preposition_cat_at_that.stared_vs_remarked",
    vocabulary=(' at', ' that'),
    generator_role="generate_linked_verb_preposition_cat_at_that_fit_panels",
    answer_role="score_jointly_tokenized_at_versus_that",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} {'stared' if pos else 'remarked'}, of course,"),
    a2=bs.Family("storm_frame", "storm_frame_verb_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} {'stared' if pos else 'remarked'}, of course,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} {'stared' if pos else 'remarked'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('stared_to_remarked', 'remarked_to_stared'),
    kinds=('stared', 'remarked'),
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
