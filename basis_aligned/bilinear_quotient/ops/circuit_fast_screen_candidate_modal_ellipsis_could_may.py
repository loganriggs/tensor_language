#!/usr/bin/env python3
"""modal_ellipsis_could_may.could_vs_may -- does a site carry the MODAL'S IDENTITY (could -> `could`, may -> `may`) across the verb phrase and a parenthetical to the elliptical `and so ___` slot?

`The pilot near the lantern could lift the crate, of course, and so` obliges ` could`; `... may lift ...` obliges ` may`. Third member of the modal-ellipsis family (can/must counted, should/might screened in v233), NEW readout token pair could/may.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v235 batch).
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
    task_id="modal_ellipsis_could_may.could_vs_may",
    vocabulary=(' could', ' may'),
    generator_role="generate_linked_modal_ellipsis_could_may_fit_panels",
    answer_role="score_jointly_tokenized_could_versus_may",
    a1=bs.Family("bare_frame", "bare_frame_modal_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} {'could' if pos else 'may'} lift the crate, of course, and so"),
    a2=bs.Family("since_frame", "since_frame_modal_swap", lambda i, pos: f"Since the {_adj(i)} {_agent(i)} {'could' if pos else 'may'} carry it, of course, the {_alt(i)} surely"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} {'could' if pos else 'may'} lift the crate, of course, and so",
    a1_suffix=lambda i: ", of course, and so",
    a2_suffix=lambda i: f", of course, the {_alt(i)} surely",
    directions=('could_to_may', 'may_to_could'),
    kinds=('could', 'may'),
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
