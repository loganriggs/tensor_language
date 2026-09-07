#!/usr/bin/env python3
"""countability_fewer_less.crates_vs_water -- does a site carry an object noun's COUNTABILITY (crates -> `fewer`, water -> `less`) across a parenthetical and a clause to the comparative quantifier?

`The pilot near the lantern wanted crates, of course, but was given` obliges ` fewer`; `... wanted water ...` obliges ` less`. Countability family (few/little, much/many, number/amount exist; less_fewer_countability reads the NOUN from the quantifier — this reads the quantifier from the noun), NEW readout token pair fewer/less.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v239 batch).
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
    task_id="countability_fewer_less.crates_vs_water",
    vocabulary=(' fewer', ' less'),
    generator_role="generate_linked_countability_fewer_less_fit_panels",
    answer_role="score_jointly_tokenized_fewer_versus_less",
    a1=bs.Family("bare_frame", "bare_frame_noun_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} wanted {'crates' if pos else 'water'}, of course, but was given"),
    a2=bs.Family("because_frame", "because_frame_noun_swap", lambda i, pos: f"Because {'crates' if pos else 'water'} ran short, of course, the {_adj(i)} {_alt(i)} expected"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} wanted {'crates' if pos else 'water'}, of course, but was given",
    a1_suffix=lambda i: ", of course, but was given",
    a2_suffix=lambda i: f" ran short, of course, the {_adj(i)} {_alt(i)} expected",
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
