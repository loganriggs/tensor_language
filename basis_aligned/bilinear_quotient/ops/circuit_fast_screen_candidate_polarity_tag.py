#!/usr/bin/env python3
"""polarity_tag.not_vs_also -- does a site carry a clause's POLARITY (did not / did also) across `leave, and` to the additive tag (neither vs so)?

`The pilot near the lantern did not leave, and` obliges ` neither`; `did also leave, and` obliges ` so`. Polarity sibling of polarity_addition (too/either) with the tag fronted; the inversion pair so/neither_inversion uses these words as CUES, here they are the answers.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v213 batch).
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
    task_id="polarity_tag.not_vs_also",
    vocabulary=(' neither', ' so'),
    generator_role="generate_linked_polarity_tag_fit_panels",
    answer_role="score_jointly_tokenized_neither_versus_so",
    a1=bs.Family("bare_frame", "bare_frame_polarity_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} did {'not' if pos else 'also'} leave, and"),
    a2=bs.Family("dawn_frame", "dawn_frame_polarity_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} did {'not' if pos else 'also'} leave, and"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_obj2(i)} did {'not' if pos else 'also'} leave, and",
    a1_suffix=lambda i: " leave, and",
    a2_suffix=lambda i: " leave, and",
    directions=('negative_to_positive', 'positive_to_negative'),
    kinds=('negative', 'positive'),
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
