#!/usr/bin/env python3
"""noun_complementizer.question_vs_claim -- does a site carry a head NOUN's complementizer selection (question -> whether, claim -> that) across a by-phrase to the copula's complement?

`The question raised by the pilot was` obliges ` whether`; `The claim raised by the pilot was` obliges ` that`. Complementizer sibling of verb_complementizer (wondered/believed) with the selector a noun.

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
    task_id="noun_complementizer.question_vs_claim",
    vocabulary=(' whether', ' that'),
    generator_role="generate_linked_noun_complementizer_fit_panels",
    answer_role="score_jointly_tokenized_whether_versus_that",
    a1=bs.Family("raised_frame", "raised_frame_noun_swap", lambda i, pos: f"The {'question' if pos else 'claim'} raised by the {_agent(i)} was"),
    a2=bs.Family("later_frame", "later_frame_noun_swap", lambda i, pos: f"Later the {'question' if pos else 'claim'} from the {_adj(i)} {_agent(i)} was"),
    p_donor=lambda i, pos: f"The {'question' if pos else 'claim'} raised by the {_alt(i)} was",
    a1_suffix=lambda i: " was",
    a2_suffix=lambda i: " was",
    directions=('interrogative_to_declarative', 'declarative_to_interrogative'),
    kinds=('interrogative', 'declarative'),
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
