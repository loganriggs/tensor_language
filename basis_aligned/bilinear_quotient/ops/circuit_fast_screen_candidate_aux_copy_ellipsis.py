#!/usr/bin/env python3
"""aux_copy_ellipsis.can_vs_should -- does a site COPY the clause's modal auxiliary (can vs should) into a second-conjunct ellipsis (`and the sailor __`)?

`The pilot near the lantern can leave, and the sailor` obliges ` can`; `should leave, and the sailor` obliges ` should`. Sibling of aux_copy_so without the `so` trigger.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v215 batch).
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
    task_id="aux_copy_ellipsis.can_vs_should",
    vocabulary=(' can', ' should'),
    generator_role="generate_linked_aux_copy_ellipsis_fit_panels",
    answer_role="score_jointly_tokenized_can_versus_should",
    a1=bs.Family("bare_frame", "bare_frame_modal_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} {'can' if pos else 'should'} leave, and the {_alt(i)}"),
    a2=bs.Family("dawn_frame", "dawn_frame_modal_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} {'can' if pos else 'should'} leave, and the {_alt(i)}"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_obj2(i)} {'can' if pos else 'should'} leave, and the {_alt(i)}",
    a1_suffix=lambda i: f" leave, and the {_alt(i)}",
    a2_suffix=lambda i: f" leave, and the {_alt(i)}",
    directions=('can_to_should', 'should_to_can'),
    kinds=('can', 'should'),
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
