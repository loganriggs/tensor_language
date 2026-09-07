#!/usr/bin/env python3
"""each_all_agreement.each_vs_all -- does a site carry a distributive quantifier's NUMBER (each -> singular, all -> plural) past a plural attractor to the verb?

`Each of the pilots near the lantern` obliges ` has`; `All of the pilots near the lantern` obliges ` have`. Sibling of partitive_agreement (One/Two of).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v211 batch).
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
    task_id="each_all_agreement.each_vs_all",
    vocabulary=(' has', ' have'),
    generator_role="generate_linked_each_all_agreement_fit_panels",
    answer_role="score_jointly_tokenized_has_versus_have",
    a1=bs.Family("bare_frame", "bare_frame_quant_swap", lambda i, pos: f"{'Each' if pos else 'All'} of the {_agent(i)}s near the {_obj(i)}"),
    a2=bs.Family("notes_frame", "notes_frame_quant_swap", lambda i, pos: f"In the notes {'each' if pos else 'all'} of the {_adj(i)} {_agent(i)}s"),
    p_donor=lambda i, pos: f"{'Each' if pos else 'All'} of the {_alt(i)}s near the {_obj(i)}",
    a1_suffix=lambda i: f" {_obj(i)}",
    a2_suffix=lambda i: f" {_agent(i)}s",
    directions=('singular_to_plural', 'plural_to_singular'),
    kinds=('singular', 'plural'),
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
