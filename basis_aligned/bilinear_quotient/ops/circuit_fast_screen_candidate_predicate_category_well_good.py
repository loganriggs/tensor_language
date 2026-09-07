#!/usr/bin/env python3
"""predicate_category_well_good.sang_vs_looked -- does a site carry the VERB'S CLASS (lexical sang -> adverb `well`, copular looked -> adjective `good`) across a parenthetical and a degree word to the manner/quality slot?

`The pilot near the lantern sang, of course, remarkably` obliges ` well`; `The pilot near the lantern looked, of course, remarkably` obliges ` good`. Predicate-category family (quiet/quietly after seem/feel exists), NEW readout token pair well/good (suppletive adverb).

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
    task_id="predicate_category_well_good.sang_vs_looked",
    vocabulary=(' well', ' good'),
    generator_role="generate_linked_predicate_category_well_good_fit_panels",
    answer_role="score_jointly_tokenized_well_versus_good",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} {'sang' if pos else 'looked'}, of course, remarkably"),
    a2=bs.Family("agreed_frame", "agreed_frame_verb_swap", lambda i, pos: f"Everyone agreed the {_adj(i)} {_agent(i)} {'performed' if pos else 'seemed'}, of course, quite"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} {'sang' if pos else 'looked'}, of course, remarkably",
    a1_suffix=lambda i: ", of course, remarkably",
    a2_suffix=lambda i: ", of course, quite",
    directions=('lexical_to_copular', 'copular_to_lexical'),
    kinds=('lexical', 'copular'),
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
