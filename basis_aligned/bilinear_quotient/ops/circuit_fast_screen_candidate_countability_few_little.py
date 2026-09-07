#!/usr/bin/env python3
"""countability_few_little.crates_vs_water -- does a site carry a noun's COUNTABILITY (count crates -> `few`, mass water -> `little`) across a PP, a parenthetical and a result clause to the quantity determiner after `a`?

`The crates near the lantern ran low, of course, so the pilot took only a` obliges ` few`; `The water near the lantern ran low, of course, so the pilot took only a` obliges ` little`. Countability family (existing readouts many/much, water/crates, crate/water), NEW readout token pair few/little.

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
    task_id="countability_few_little.crates_vs_water",
    vocabulary=(' few', ' little'),
    generator_role="generate_linked_countability_few_little_fit_panels",
    answer_role="score_jointly_tokenized_few_versus_little",
    a1=bs.Family("bare_frame", "bare_frame_noun_swap", lambda i, pos: f"The {'crates' if pos else 'water'} near the {_obj(i)} ran low, of course, so the {_agent(i)} took only a"),
    a2=bs.Family("since_frame", "since_frame_noun_swap", lambda i, pos: f"Since the {_adj(i)} {_agent(i)} had {'crates' if pos else 'water'} to spare, of course, the {_alt(i)} got quite a"),
    p_donor=lambda i, pos: f"The {'crates' if pos else 'water'} near the {_adj2(i)} {_obj(i)} ran low, of course, so the {_agent(i)} took only a",
    a1_suffix=lambda i: f", of course, so the {_agent(i)} took only a",
    a2_suffix=lambda i: f", of course, the {_alt(i)} got quite a",
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
