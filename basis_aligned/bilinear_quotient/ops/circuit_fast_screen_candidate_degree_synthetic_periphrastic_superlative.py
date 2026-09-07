#!/usr/bin/env python3
"""degree_synthetic_periphrastic_superlative.tall_vs_careful -- does a site carry an adjective's MORPHOLOGICAL CLASS (tall -> synthetic `tallest`, careful -> periphrastic `most`) across a parenthetical and a second subject to the superlative form?

`The pilots near the lantern were tall, of course, and the sailor was the` obliges ` tallest`; `... were careful ...` obliges ` most`. Second member of the synthetic/periphrastic degree family (comparative sibling: degree_synthetic_periphrastic_comparative).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v231 batch).
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
    task_id="degree_synthetic_periphrastic_superlative.tall_vs_careful",
    vocabulary=(' tallest', ' most'),
    generator_role="generate_linked_degree_synthetic_periphrastic_superlative_fit_panels",
    answer_role="score_jointly_tokenized_tallest_versus_most",
    a1=bs.Family("bare_frame", "bare_frame_adjective_swap", lambda i, pos: f"The {_agent(i)}s near the {_obj(i)} were {'tall' if pos else 'careful'}, of course, and the {_alt(i)} was the"),
    a2=bs.Family("every_frame", "every_frame_adjective_swap", lambda i, pos: f"Every {_adj(i)} {_agent(i)} there was {'tall' if pos else 'careful'}, of course, but the {_alt(i)} was the"),
    p_donor=lambda i, pos: f"The {_agent(i)}s near the {_adj2(i)} {_obj(i)} were {'tall' if pos else 'careful'}, of course, and the {_alt(i)} was the",
    a1_suffix=lambda i: f", of course, and the {_alt(i)} was the",
    a2_suffix=lambda i: f", of course, but the {_alt(i)} was the",
    directions=('synthetic_to_periphrastic', 'periphrastic_to_synthetic'),
    kinds=('synthetic', 'periphrastic'),
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
