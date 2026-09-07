#!/usr/bin/env python3
"""temporal_order_first_second.before_vs_after -- does a site carry a temporal connective's ORDER (arrived before -> `first`, arrived after -> `second`) across the other participant, a parenthetical and a result clause to the ordinal?

`The pilot arrived before the sailor, of course, so the pilot was` obliges ` first`; `The pilot arrived after the sailor, of course, so the pilot was` obliges ` second`. NEW family (ordinal from a temporal connective).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v229 batch).
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
    task_id="temporal_order_first_second.before_vs_after",
    vocabulary=(' first', ' second'),
    generator_role="generate_linked_temporal_order_first_second_fit_panels",
    answer_role="score_jointly_tokenized_first_versus_second",
    a1=bs.Family("bare_frame", "bare_frame_connective_swap", lambda i, pos: f"The {_agent(i)} arrived {'before' if pos else 'after'} the {_alt(i)}, of course, so the {_agent(i)} was"),
    a2=bs.Family("since_frame", "since_frame_connective_swap", lambda i, pos: f"Since the {_adj(i)} {_agent(i)} came {'before' if pos else 'after'} the {_alt(i)}, of course, the {_agent(i)} finished"),
    p_donor=lambda i, pos: f"The {_agent(i)} arrived {'before' if pos else 'after'} the {_adj2(i)} {_alt(i)}, of course, so the {_agent(i)} was",
    a1_suffix=lambda i: f", of course, so the {_agent(i)} was",
    a2_suffix=lambda i: f", of course, the {_agent(i)} finished",
    directions=('before_to_after', 'after_to_before'),
    kinds=('before', 'after'),
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
