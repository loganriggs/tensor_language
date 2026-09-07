#!/usr/bin/env python3
"""binding.local_vs_embedded -- does a site carry whether the antecedent is a local subject (reflexive) or a matrix subject (pronoun)?

Principle A/B: with the male noun as the local subject the object is ` himself`; with a female embedded subject it is ` him`.
The answer is read at the same verb in both frames.

    A1  "The king near the lantern admired" -> " himself" / "The king said the queen admired" -> " him"
    A2  "In the story the king had quietly praised" -> " himself" / "In the story the king said the queen had praised" -> " him"

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
    task_id="binding.local_vs_embedded",
    vocabulary=(" himself", " him"),
    generator_role="generate_linked_binding_fit_panels",
    answer_role="score_jointly_tokenized_himself_versus_him",
    a1=bs.Family("bare_frame", "bare_frame_embedding_swap", lambda i, pos: f"The {_g(i, True)} near the {_obj(i)} admired" if pos else f"The {_g(i, True)} said the {_g(i, False)} admired"),
    a2=bs.Family("story_frame", "story_frame_embedding_swap", lambda i, pos: f"In the story the {_g(i, True)} had quietly praised" if pos else f"In the story the {_g(i, True)} said the {_g(i, False)} had praised"),
    p_donor=lambda i, pos: f"The {_g(i, True)} near the {_obj2(i)} admired" if pos else f"The {_g(i, True)} said the {_g((i + 1), False)} admired",
    a1_suffix=lambda i: " admired",
    a2_suffix=lambda i: " praised",
    directions=("local_to_embedded", "embedded_to_local"),
    kinds=("local_subject", "embedded_subject"),
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
