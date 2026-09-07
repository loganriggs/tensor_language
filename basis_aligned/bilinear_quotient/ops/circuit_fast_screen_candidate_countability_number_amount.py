#!/usr/bin/env python3
"""countability_number_amount.crates_vs_water -- does a site carry a noun's COUNTABILITY (count crates -> `number`, mass water -> `amount`) across a PP, a parenthetical and a second clause to the quantity noun after `sheer`?

`The pilot checked the crates near the lantern, of course, and was struck by the sheer` obliges ` number`; `... the water ...` obliges ` amount`. Countability family, NEW readout token pair number/amount.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v231 batch).
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
    task_id="countability_number_amount.crates_vs_water",
    vocabulary=(' number', ' amount'),
    generator_role="generate_linked_countability_number_amount_fit_panels",
    answer_role="score_jointly_tokenized_number_versus_amount",
    a1=bs.Family("bare_frame", "bare_frame_noun_swap", lambda i, pos: f"The {_agent(i)} checked the {'crates' if pos else 'water'} near the {_obj(i)}, of course, and was struck by the sheer"),
    a2=bs.Family("seeing_frame", "seeing_frame_noun_swap", lambda i, pos: f"Seeing the {'crates' if pos else 'water'} the {_adj(i)} {_agent(i)} had brought, of course, the {_alt(i)} noted the sheer"),
    p_donor=lambda i, pos: f"The {_agent(i)} checked the {'crates' if pos else 'water'} near the {_adj2(i)} {_obj(i)}, of course, and was struck by the sheer",
    a1_suffix=lambda i: ", of course, and was struck by the sheer",
    a2_suffix=lambda i: f", of course, the {_alt(i)} noted the sheer",
    directions=('count_to_mass', 'mass_to_count'),
    kinds=('count', 'mass'),
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
