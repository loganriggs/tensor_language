#!/usr/bin/env python3
"""polarity_addition.too_vs_either -- does a site carry clause polarity to the additive particle?

An affirmative pair of clauses closes with ` too`, a negative pair with ` either`; the cue is `often`/`never` in both clauses and the answer is read at the adjective.
polarity_licensing reads the same licensor into anything/something -- separability sibling.

    A1  "The pilot was often bright and the sailor was often bright" -> " too" / "... never ... never bright" -> " either"
    A2  "The lantern was often bright and the pilot was often so" -> " too"

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
    task_id="polarity_addition.too_vs_either",
    vocabulary=(" too", " either"),
    generator_role="generate_linked_polarity_addition_fit_panels",
    answer_role="score_jointly_tokenized_too_versus_either",
    a1=bs.Family("bare_frame", "bare_frame_polarity_swap", lambda i, pos: f"The {_agent(i)} was {'often' if pos else 'never'} {_adj(i)} and the {_alt(i)} was {'often' if pos else 'never'} {_adj(i)}"),
    a2=bs.Family("so_frame", "so_frame_polarity_swap", lambda i, pos: f"The {_obj(i)} was {'often' if pos else 'never'} {_adj(i)} and the {_agent(i)} was {'often' if pos else 'never'} so"),
    p_donor=lambda i, pos: f"The {_agent(i)} was {'often' if pos else 'never'} {_adj2(i)} and the {_alt(i)} was {'often' if pos else 'never'} {_adj(i)}",
    a1_suffix=lambda i: f" {_adj(i)}",
    a2_suffix=lambda i: " so",
    directions=("affirmative_to_negative", "negative_to_affirmative"),
    kinds=("affirmative_pair", "negative_pair"),
    p_generator_role="adjective_lexical_rewrite",
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
