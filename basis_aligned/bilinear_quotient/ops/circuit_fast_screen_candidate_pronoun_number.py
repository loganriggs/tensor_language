#!/usr/bin/env python3
"""pronoun_number.plural_vs_singular -- does a site carry the subject's number to a subject pronoun?

A clause-initial subject pronoun agrees in number with the noun of the preceding clause; the answer is read at the conjunction.

    A1  "The pilots found the lantern and then" -> " they" / "The pilot ..." -> " he"
    A2  "The lantern belonged to the pilots and" -> " they"

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
    task_id="pronoun_number.plural_vs_singular",
    vocabulary=(" they", " he"),
    generator_role="generate_linked_pronoun_number_fit_panels",
    answer_role="score_jointly_tokenized_they_versus_he",
    a1=bs.Family("bare_frame", "bare_frame_number_swap", lambda i, pos: f"The {_agent(i)}{'s' if pos else ''} found the {_obj(i)} and then"),
    a2=bs.Family("belong_frame", "belong_frame_number_swap", lambda i, pos: f"The {_obj(i)} belonged to the {_agent(i)}{'s' if pos else ''} and"),
    p_donor=lambda i, pos: f"The {_alt(i)}{'s' if pos else ''} found the {_obj(i)} and then",
    a1_suffix=lambda i: " and then",
    a2_suffix=lambda i: " and",
    directions=("plural_to_singular", "singular_to_plural"),
    kinds=("plural_subject", "singular_subject"),
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
