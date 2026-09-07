#!/usr/bin/env python3
"""degree_synthetic_periphrastic_adverb.fast_vs_gracefully -- does a site carry an ADVERB's MORPHOLOGICAL CLASS (monosyllabic fast -> synthetic `faster`, polysyllabic gracefully -> periphrastic `more`) across a parenthetical and a second subject to the comparative form?

`The pilot near the lantern ran fast, of course, and the sailor ran even` obliges ` faster`; `... ran gracefully ...` obliges ` more`. Third member of the synthetic/periphrastic degree family (comparative taller/more and superlative tallest/most exist), NEW readout token pair faster/more and the first ADVERB member.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v233 batch).
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
    task_id="degree_synthetic_periphrastic_adverb.fast_vs_gracefully",
    vocabulary=(' faster', ' more'),
    generator_role="generate_linked_degree_synthetic_periphrastic_adverb_fit_panels",
    answer_role="score_jointly_tokenized_faster_versus_more",
    a1=bs.Family("bare_frame", "bare_frame_adverb_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} ran {'fast' if pos else 'gracefully'}, of course, and the {_alt(i)} ran even"),
    a2=bs.Family("but_frame", "but_frame_adverb_swap", lambda i, pos: f"The {_agent(i)} worked {'fast' if pos else 'gracefully'}, of course, but the {_adj(i)} {_alt(i)} worked even"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} ran {'fast' if pos else 'gracefully'}, of course, and the {_alt(i)} ran even",
    a1_suffix=lambda i: f", of course, and the {_alt(i)} ran even",
    a2_suffix=lambda i: f", of course, but the {_adj(i)} {_alt(i)} worked even",
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
