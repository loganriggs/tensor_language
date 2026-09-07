#!/usr/bin/env python3
"""object_pronoun_gender.king_vs_queen -- does a site carry an earlier noun's gender to an object pronoun?

`The lantern belonged to the king, so the crew returned it to` obliges ` him`; queen obliges ` her`. The cue is seven tokens back.

    A1  "The lantern belonged to the king so the crew returned it to" -> " him"
    A2  "Because the king had asked, the pilot gave the lantern to" -> " him"

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
    task_id="object_pronoun_gender.king_vs_queen",
    vocabulary=(" him", " her"),
    generator_role="generate_linked_object_pronoun_gender_fit_panels",
    answer_role="score_jointly_tokenized_him_versus_her",
    a1=bs.Family("belong_frame", "belong_frame_gender_swap", lambda i, pos: f"The {_obj(i)} belonged to the {_g(i, pos)} so the {_agent(i)} returned it to"),
    a2=bs.Family("ask_frame", "ask_frame_gender_swap", lambda i, pos: f"Because the {_g(i, pos)} had asked, the {_agent(i)} gave the {_obj(i)} to"),
    p_donor=lambda i, pos: f"The {_obj(i)} belonged to the {_g(i, pos)} so the {_alt(i)} returned it to",
    a1_suffix=lambda i: " returned it to",
    a2_suffix=lambda i: " to",
    directions=("male_to_female", "female_to_male"),
    kinds=("male_antecedent", "female_antecedent"),
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
