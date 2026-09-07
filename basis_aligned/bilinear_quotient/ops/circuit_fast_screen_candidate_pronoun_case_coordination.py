#!/usr/bin/env python3
"""pronoun_case_coordination.between_vs_both -- does a site carry a coordinator's CASE requirement (Between -> objective me, Both -> subjective I) across `the NOUN and` to the pronoun?

`Between the sailor and` obliges ` me`; `Both the sailor and` obliges ` I`. NEW family (pronoun case): the case is assigned by the first word and read at the coordinated pronoun.

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
    task_id="pronoun_case_coordination.between_vs_both",
    vocabulary=(' me', ' I'),
    generator_role="generate_linked_pronoun_case_coordination_fit_panels",
    answer_role="score_jointly_tokenized_me_versus_I",
    a1=bs.Family("bare_frame", "bare_frame_head_swap", lambda i, pos: f"{'Between' if pos else 'Both'} the {_alt(i)} and"),
    a2=bs.Family("adj_frame", "adj_frame_head_swap", lambda i, pos: f"{'Between' if pos else 'Both'} the {_adj(i)} {_alt(i)} and"),
    p_donor=lambda i, pos: f"{'Between' if pos else 'Both'} the {_agent(i)} and",
    a1_suffix=lambda i: " and",
    a2_suffix=lambda i: " and",
    directions=('objective_to_subjective', 'subjective_to_objective'),
    kinds=('objective', 'subjective'),
    p_generator_role="agent_lexical_rewrite",
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
