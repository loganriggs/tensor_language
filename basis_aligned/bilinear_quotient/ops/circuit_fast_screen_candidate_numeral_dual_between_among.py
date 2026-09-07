#!/usr/bin/env python3
"""numeral_dual_between_among.two_vs_three -- does a site carry a numeral's DUALITY (two -> `between`, three -> `among`) across a clause and a parenthetical to the distributive preposition?

`There were two sailors near the lantern, and the pilot divided the crate` obliges ` between`; `There were three sailors ...` obliges ` among`. (A1 repaired once after the capability check: the parenthetical version read -0.17/-0.02.) Second member of the numeral-duality family (both/all), NEW readout token pair between/among.

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
    task_id="numeral_dual_between_among.two_vs_three",
    vocabulary=(' between', ' among'),
    generator_role="generate_linked_numeral_dual_between_among_fit_panels",
    answer_role="score_jointly_tokenized_between_versus_among",
    a1=bs.Family("bare_frame", "bare_frame_numeral_swap", lambda i, pos: f"There were {'two' if pos else 'three'} {_alt(i)}s near the {_obj(i)}, and the {_agent(i)} divided the crate"),
    a2=bs.Family("waiting_frame", "waiting_frame_numeral_swap", lambda i, pos: f"With {'two' if pos else 'four'} {_alt(i)}s waiting, the {_adj(i)} {_agent(i)} split the food, of course,"),
    p_donor=lambda i, pos: f"There were {'two' if pos else 'three'} {_alt(i)}s near the {_adj2(i)} {_obj(i)}, and the {_agent(i)} divided the crate",
    a1_suffix=lambda i: f", and the {_agent(i)} divided the crate",
    a2_suffix=lambda i: f" {_alt(i)}s waiting, the {_adj(i)} {_agent(i)} split the food, of course,",
    directions=('dual_to_plural', 'plural_to_dual'),
    kinds=('dual', 'plural'),
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
