#!/usr/bin/env python3
"""number_reflexive_myself_ourselves.I_vs_we -- does a site carry a first-person subject's NUMBER (I -> `myself`, we -> `ourselves`) across a PP and a parenthetical to the reflexive?

`Near the lantern I, of course, kept to` obliges ` myself`; `Near the lantern we, of course, kept to` obliges ` ourselves`. Number family (himself/themselves, ourselves/yourselves exist), NEW readout token pair myself/ourselves.

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
    task_id="number_reflexive_myself_ourselves.I_vs_we",
    vocabulary=(' myself', ' ourselves'),
    generator_role="generate_linked_number_reflexive_myself_ourselves_fit_panels",
    answer_role="score_jointly_tokenized_myself_versus_ourselves",
    a1=bs.Family("bare_frame", "bare_frame_pronoun_swap", lambda i, pos: f"Near the {_obj(i)} {'I' if pos else 'we'}, of course, kept to"),
    a2=bs.Family("because_frame", "because_frame_pronoun_swap", lambda i, pos: f"Because {'I' if pos else 'we'} helped the {_adj(i)} {_agent(i)}, of course, by"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} {'I' if pos else 'we'}, of course, kept to",
    a1_suffix=lambda i: ", of course, kept to",
    a2_suffix=lambda i: f" helped the {_adj(i)} {_agent(i)}, of course, by",
    directions=('singular_to_plural', 'plural_to_singular'),
    kinds=('singular', 'plural'),
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
