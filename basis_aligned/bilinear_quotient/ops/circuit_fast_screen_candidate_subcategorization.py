#!/usr/bin/env python3
"""subcategorization.put_vs_saw -- does a site carry the verb's argument frame past the object to the obligatory locative?

`put the bright lantern` obliges a locative ` on`; `saw the bright lantern` continues with ` and`. The cue is the verb three tokens back.

    A1  "The pilot put the bright lantern" -> " on" / "The pilot saw the bright lantern" -> " and"
    A2  "Before dawn the pilot had put the lantern" -> " on"

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
    task_id="subcategorization.put_vs_saw",
    vocabulary=(" on", " and"),
    generator_role="generate_linked_subcategorization_fit_panels",
    answer_role="score_jointly_tokenized_on_versus_and",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The {_agent(i)} {'put' if pos else 'saw'} the {_adj(i)} {_obj(i)}"),
    a2=bs.Family("dawn_frame", "dawn_frame_verb_swap", lambda i, pos: f"Before dawn the {_agent(i)} had {'put' if pos else 'seen'} the {_obj(i)}"),
    p_donor=lambda i, pos: f"The {_alt(i)} {'put' if pos else 'saw'} the {_adj(i)} {_obj(i)}",
    a1_suffix=lambda i: f" {_obj(i)}",
    a2_suffix=lambda i: f" {_obj(i)}",
    directions=("locative_to_plain", "plain_to_locative"),
    kinds=("locative_verb", "plain_verb"),
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
