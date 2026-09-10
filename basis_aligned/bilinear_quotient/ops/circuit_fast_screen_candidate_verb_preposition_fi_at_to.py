#!/usr/bin/env python3
"""verb_preposition_fi_at_to.aimed_vs_appealed -- does a site carry the cared/appealed selection when the cue is a BARE verb in a DECLARATIVE clause?

`The pilot did really care, then,` obliges ` at`; `... really appeal, then,` obliges ` to`. This cell exists to separate two things that frame G confounded: G changed the clause type to interrogative AND the cue's surface form to the bare verb under do-support. Here the bare form is kept and the clause stays declarative, so a failure here is about the FORM.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-10, v463 second-mapping replication batch).
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
    task_id="verb_preposition_fi_at_to.aimed_vs_appealed",
    vocabulary=(' at', ' to'),
    generator_role="generate_linked_verb_preposition_fi_at_to_fit_panels",
    answer_role="score_jointly_tokenized_at_versus_to",
    a1=bs.Family("emphatic_frame", "emphatic_frame_verb_swap", lambda i, pos: f"The {_agent(i)} did really {'aim' if pos else 'appeal'}, then,"),
    a2=bs.Family("emphatic_adj_frame", "emphatic_adj_frame_verb_swap", lambda i, pos: f"The {_adj(i)} {_agent(i)} did truly {'aim' if pos else 'appeal'}, then,"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} did really {'aim' if pos else 'appeal'}, then,",
    a1_suffix=lambda i: ", then,",
    a2_suffix=lambda i: ", then,",
    directions=('aimed_to_appealed', 'appealed_to_aimed'),
    kinds=('aimed', 'appealed'),
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
