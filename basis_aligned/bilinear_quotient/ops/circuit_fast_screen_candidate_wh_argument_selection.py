#!/usr/bin/env python3
"""wh_argument_selection.what_vs_where -- does a site carry a fronted wh-word's TYPE (What -> object gap, Where -> place gap) across the subject to the selected verb?

`What did the pilot near the lantern` obliges ` carry`; `Where did the pilot near the lantern` obliges ` go`. NEW family (wh-type to verb selection).

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
    task_id="wh_argument_selection.what_vs_where",
    vocabulary=(' carry', ' go'),
    generator_role="generate_linked_wh_argument_selection_fit_panels",
    answer_role="score_jointly_tokenized_carry_versus_go",
    a1=bs.Family("bare_frame", "bare_frame_wh_swap", lambda i, pos: f"{'What' if pos else 'Where'} did the {_agent(i)} near the {_obj(i)}"),
    a2=bs.Family("adverb_frame", "adverb_frame_wh_swap", lambda i, pos: f"{'What' if pos else 'Where'} did the {_adj(i)} {_agent(i)} finally"),
    p_donor=lambda i, pos: f"{'What' if pos else 'Where'} did the {_alt(i)} near the {_obj(i)}",
    a1_suffix=lambda i: f" near the {_obj(i)}",
    a2_suffix=lambda i: " finally",
    directions=('object_to_place', 'place_to_object'),
    kinds=('object_gap', 'place_gap'),
    p_generator_role="agent_lexical_rewrite",
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
