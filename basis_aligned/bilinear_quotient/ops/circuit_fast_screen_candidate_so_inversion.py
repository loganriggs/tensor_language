#!/usr/bin/env python3
"""so_inversion.so_vs_then -- does a site carry a clause-initial `so` (after a positive clause) to subject-auxiliary inversion?

`The pilot near the lantern left early, and so` obliges ` did` (so, naturally, did the sailor); `and then` obliges the plain subject (` the`). Sibling of negative_inversion (v207): the same inversion readout under a different trigger.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v209 batch).
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
    task_id="so_inversion.so_vs_then",
    vocabulary=(' did', ' the'),
    generator_role="generate_linked_so_inversion_fit_panels",
    answer_role="score_jointly_tokenized_did_versus_the",
    a1=bs.Family("bare_frame", "bare_frame_trigger_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} left early, and {'so' if pos else 'then'}, naturally,"),
    a2=bs.Family("notes_frame", "notes_frame_trigger_swap", lambda i, pos: f"In the notes the {_agent(i)} had agreed, and {'so' if pos else 'then'}, naturally,"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_obj2(i)} left early, and {'so' if pos else 'then'}, naturally,",
    a1_suffix=lambda i: ", naturally,",
    a2_suffix=lambda i: ", naturally,",
    directions=('so_to_then', 'then_to_so'),
    kinds=('inverting', 'plain'),
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
