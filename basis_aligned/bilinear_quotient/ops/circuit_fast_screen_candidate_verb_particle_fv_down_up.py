#!/usr/bin/env python3
"""verb_particle_fv_down_up.recent_vs_matrix -- when a competing particle-selecting verb sits between the matrix verb and the readout, which one does the model follow?

`The pilot woke the cadets who calmed, of course,` -- the MATRIX verb is `woke` (-> ` up`) and the NEAREST verb is `calmed` (-> ` down`). MIRROR of verb_particle_fu_up_down: the SENTENCES are identical and only the vocabulary is re-keyed, to the NEAREST verb. fu keyed to the matrix verb dropped 32 of 32 rows in both A1 and A2 -- the donor never beat the base on the matrix verb's axis -- so if this mirror clears the floor the model is following the nearest cue and not the syntactic subject's verb. Every other cell in this corpus has exactly one particle-selecting verb, so the two accounts have never been separated.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v385 competing-cue batch).
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
    task_id="verb_particle_fv_down_up.recent_vs_matrix",
    vocabulary=(' down', ' up'),
    generator_role="generate_linked_verb_particle_fv_down_up_fit_panels",
    answer_role="score_jointly_tokenized_down_versus_up",
    a1=bs.Family("competing_cue_frame", "competing_cue_verb_swap", lambda i, pos: f"The {_agent(i)} {'woke' if pos else 'calmed'} the cadets who {'calmed' if pos else 'woke'}, of course,"),
    a2=bs.Family("competing_cue_storm_frame", "competing_cue_storm_verb_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} {'woke' if pos else 'calmed'} the cadets who {'calmed' if pos else 'woke'}, of course,"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} {'woke' if pos else 'calmed'} the cadets who {'calmed' if pos else 'woke'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('woke_matrix_to_calmed_matrix', 'calmed_matrix_to_woke_matrix'),
    kinds=('woke', 'calmed'),
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
