#!/usr/bin/env python3
"""person_reflexive_ourselves_themselves.I_vs_third -- does a site carry a coordinated subject's PERSON (`the pilots and I` -> `ourselves`, `the pilots and the sailor` -> `themselves`) across a PP and a parenthetical to the plural reflexive?

`The pilots and I near the lantern, of course, kept to` obliges ` ourselves`; `The pilots and the sailor near the lantern, of course, kept to` obliges ` themselves`. Person family (myself/yourself, ourselves/yourselves exist), NEW readout token pair ourselves/themselves (first/third plural).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v235 batch).
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
    task_id="person_reflexive_ourselves_themselves.I_vs_third",
    vocabulary=(' ourselves', ' themselves'),
    generator_role="generate_linked_person_reflexive_ourselves_themselves_fit_panels",
    answer_role="score_jointly_tokenized_ourselves_versus_themselves",
    a1=bs.Family("bare_frame", "bare_frame_conjunct_swap", lambda i, pos: f"The {_agent(i)}s and {'I' if pos else 'the ' + _alt(i)} near the {_obj(i)}, of course, kept to"),
    a2=bs.Family("because_frame", "because_frame_conjunct_swap", lambda i, pos: f"Because the {_adj(i)} {_agent(i)}s and {'I' if pos else 'the ' + _alt(i)} worked, of course, by"),
    p_donor=lambda i, pos: f"The {_agent(i)}s and {'I' if pos else 'the ' + _alt(i)} near the {_adj2(i)} {_obj(i)}, of course, kept to",
    a1_suffix=lambda i: ", of course, kept to",
    a2_suffix=lambda i: ", of course, by",
    directions=('first_to_third', 'third_to_first'),
    kinds=('first', 'third'),
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
