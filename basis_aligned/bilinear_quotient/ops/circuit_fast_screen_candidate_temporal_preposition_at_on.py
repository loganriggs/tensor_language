#!/usr/bin/env python3
"""temporal_preposition_at_on.noon_vs_Monday -- does a site carry a time noun's CLASS (noon -> `at`, Monday -> `on`) across a parenthetical and a clause to the temporal preposition?

`The pilot near the lantern chose noon, of course, and told everyone to come` obliges ` at`; `... chose Monday ...` obliges ` on`. (A1 repaired once after the capability check: the `set the meeting for ..., so everyone gathered` version read -0.10/+0.08.) Duration/point family neighbour (for/since, within/by read the noun FROM the preposition; this reads the preposition FROM the noun), NEW readout token pair at/on.

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
    task_id="temporal_preposition_at_on.noon_vs_Monday",
    vocabulary=(' at', ' on'),
    generator_role="generate_linked_temporal_preposition_at_on_fit_panels",
    answer_role="score_jointly_tokenized_at_versus_on",
    a1=bs.Family("bare_frame", "bare_frame_time_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} chose {'noon' if pos else 'Monday'}, of course, and told everyone to come"),
    a2=bs.Family("chose_frame", "chose_frame_time_swap", lambda i, pos: f"Because the {_adj(i)} {_agent(i)} chose {'noon' if pos else 'Monday'}, of course, the {_alt(i)} arrived"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} chose {'noon' if pos else 'Monday'}, of course, and told everyone to come",
    a1_suffix=lambda i: ", of course, and told everyone to come",
    a2_suffix=lambda i: f", of course, the {_alt(i)} arrived",
    directions=('hour_to_day', 'day_to_hour'),
    kinds=('hour', 'day'),
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
