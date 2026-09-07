#!/usr/bin/env python3
"""reflexive_gender.king_vs_queen -- does a site carry the antecedent's gender to a reflexive?

A reflexive object agrees in gender with the subject noun; the cue is the noun four tokens back and the answer is read at the verb.
No behaviour in the corpus reads GENDER; pronoun_gender (same cue, different readout) is its separability sibling.

    A1  "The king near the lantern admired" -> " himself" / "The queen ..." -> " herself"
    A2  "In the story the king had quietly praised" -> " himself"

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
    task_id="reflexive_gender.king_vs_queen",
    vocabulary=(" himself", " herself"),
    generator_role="generate_linked_reflexive_gender_fit_panels",
    answer_role="score_jointly_tokenized_himself_versus_herself",
    a1=bs.Family("bare_frame", "bare_frame_gender_swap", lambda i, pos: f"The {_g(i, pos)} near the {_obj(i)} admired"),
    a2=bs.Family("story_frame", "story_frame_gender_swap", lambda i, pos: f"In the story the {_g(i, pos)} had quietly praised"),
    p_donor=lambda i, pos: f"The {_g(i, pos)} near the {_obj2(i)} admired",
    a1_suffix=lambda i: " admired",
    a2_suffix=lambda i: " praised",
    directions=("male_to_female", "female_to_male"),
    kinds=("male_subject", "female_subject"),
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
