#!/usr/bin/env python3
"""deixis.came_vs_went -- does a site carry the motion verb's deictic centre to the locative?

`came straight` obliges ` here`, `went straight` obliges ` there`; the cue is the verb two tokens back.

    A1  "The pilot came straight" -> " here" / "The pilot went straight" -> " there"
    A2  "After the storm the pilot had come back" -> " here"

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, the ninth standing lesson).
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
    task_id="deixis.came_vs_went",
    vocabulary=(" here", " there"),
    generator_role="generate_linked_deixis_fit_panels",
    answer_role="score_jointly_tokenized_here_versus_there",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The {_agent(i)} {'came' if pos else 'went'} straight"),
    a2=bs.Family("storm_frame", "storm_frame_verb_swap", lambda i, pos: f"After the storm the {_agent(i)} had {'come' if pos else 'gone'} back"),
    p_donor=lambda i, pos: f"The {_alt(i)} {'came' if pos else 'went'} straight",
    a1_suffix=lambda i: " straight",
    a2_suffix=lambda i: " back",
    directions=("proximal_to_distal", "distal_to_proximal"),
    kinds=("proximal_verb", "distal_verb"),
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
