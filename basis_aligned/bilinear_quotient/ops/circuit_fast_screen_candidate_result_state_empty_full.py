#!/usr/bin/env python3
"""result_state_empty_full.drained_vs_filled -- does a site carry a verb's RESULT STATE (drained -> `empty`, filled -> `full`) across the object, a parenthetical and a result clause to the state adjective?

`The pilot drained the tank near the lantern, of course, so it was` obliges ` empty`; `The pilot filled the tank near the lantern, of course, so it was` obliges ` full`. Sibling of result_state_open_closed (result-state family, second verb pair). REPAIR: the first A2 frame (`... drained it, of course, leaving it`) dropped 26 of 32 A2 rows (both sides preferred `full`); the one licensed repair replaced it by `drained the tank, of course, and now it is` (A1 was fine at -0.68/+0.27).

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
    task_id="result_state_empty_full.drained_vs_filled",
    vocabulary=(' empty', ' full'),
    generator_role="generate_linked_result_state_empty_full_fit_panels",
    answer_role="score_jointly_tokenized_empty_versus_full",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The {_agent(i)} {'drained' if pos else 'filled'} the tank near the {_obj(i)}, of course, so it was"),
    a2=bs.Family("dawn_frame", "dawn_frame_verb_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} {'drained' if pos else 'filled'} the tank, of course, and now it is"),
    p_donor=lambda i, pos: f"The {_agent(i)} {'drained' if pos else 'filled'} the tank near the {_adj2(i)} {_obj(i)}, of course, so it was",
    a1_suffix=lambda i: ", of course, so it was",
    a2_suffix=lambda i: ", of course, and now it is",
    directions=('drained_to_filled', 'filled_to_drained'),
    kinds=('drained', 'filled'),
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
