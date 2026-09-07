#!/usr/bin/env python3
"""wh_manner_how_why.method_vs_reason -- does a site carry an antecedent noun's SEMANTIC CLASS (method -> `how`, reason -> `why`) across a parenthetical and a second clause to the wh-adverb?

`Near the lantern the pilot explained the method, of course, and that is` obliges ` how`; `Near the lantern the pilot explained the reason, of course, and that is` obliges ` why`. Re-authored at v245 after being dropped from v243 (A2 `knew the method, of course, and that is` read -0.49/+0.27 on CPU); A2 `Everyone soon learned the method, of course, and that is` read -0.23/+0.23 on CPU and was REPAIRED once (v245) to `The wary pilot finally explained the method, of course, and that is`. animacy_wh family; `how` has never been a readout token.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v245 batch).
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
    task_id="wh_manner_how_why.method_vs_reason",
    vocabulary=(' how', ' why'),
    generator_role="generate_linked_wh_manner_how_why_fit_panels",
    answer_role="score_jointly_tokenized_how_versus_why",
    a1=bs.Family("bare_frame", "bare_frame_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} explained the {'method' if pos else 'reason'}, of course, and that is"),
    a2=bs.Family("finally_frame", "finally_frame_swap", lambda i, pos: f"The {_adj(i)} {_agent(i)} finally explained the {'method' if pos else 'reason'}, of course, and that is"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} explained the {'method' if pos else 'reason'}, of course, and that is",
    a1_suffix=lambda i: ", of course, and that is",
    a2_suffix=lambda i: ", of course, and that is",
    directions=('method_to_reason', 'reason_to_method'),
    kinds=('method', 'reason'),
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
