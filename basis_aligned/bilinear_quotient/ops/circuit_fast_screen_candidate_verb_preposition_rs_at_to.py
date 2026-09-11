#!/usr/bin/env python3
"""verb_preposition_rs_at_to.aimed_vs_appealed -- a RESTRUCTURED frame for the same cue->readout mapping (aimed ->` at`, appealed -> ` to`).

Authored to test what row 2 is worth. Every verb_preposition cell in the corpus pairs an A1 `Near the {obj} the
{agent} aimed, of course,` with an A2 that changes ONLY the sentence-initial adjunct (`After the {adj} storm ...`),
a template similarity of 0.909 -- so an A2 pass licenses "survives an adjunct swap", not "generalizes to a second
construction". This cell keeps the mapping and the `, of course,` suffix but restructures the clause: the subject
becomes a complex NP with a relative clause, and a parenthetical `everyone agreed` sits between cue and readout, so
the cue-to-readout distance and the intervening material both change.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-11).
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
    task_id="verb_preposition_rs_at_to.aimed_vs_appealed",
    vocabulary=(' at', ' to'),
    generator_role="generate_linked_verb_preposition_rs_at_to_fit_panels",
    answer_role="score_jointly_tokenized_aimed_versus_appealed",
    a1=bs.Family("relative_frame", "relative_frame_verb_swap", lambda i, pos: f"The {_agent(i)} that the {_alt(i)} had trusted {'aimed' if pos else 'appealed'}, everyone agreed, of course,"),
    a2=bs.Family("storm_relative_frame", "storm_relative_frame_verb_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} that the {_alt(i)} had trusted {'aimed' if pos else 'appealed'}, everyone agreed, of course,"),
    p_donor=lambda i, pos: f"The {_agent(i)} that the {_obj2(i)} keeper had trusted {'aimed' if pos else 'appealed'}, everyone agreed, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('aimed_to_appealed', 'appealed_to_aimed'),
    kinds=('aimed', 'appealed'),
    p_generator_role="object_lexical_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
build_rows, validate_rows, authority_sha256 = SPEC.api()

if __name__ == "__main__":
    rows = build_rows()
    print(TASK_ID, len(rows), authority_sha256()[:12])
