#!/usr/bin/env python3
"""polarity_anyone_someone.nobody_vs_somebody -- does a site carry a subject's POLARITY (Nobody -> `anyone`, Somebody -> `someone`) across a PP, a verb and a parenthetical to the object indefinite?

`Nobody near the lantern told, of course,` obliges ` anyone`; `Somebody near the lantern told, of course,` obliges ` someone`. Polarity family, negative-SUBJECT cue (as ever/never in v239), NEW readout token pair anyone/someone.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v241 batch).
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
    task_id="polarity_anyone_someone.nobody_vs_somebody",
    vocabulary=(' anyone', ' someone'),
    generator_role="generate_linked_polarity_anyone_someone_fit_panels",
    answer_role="score_jointly_tokenized_anyone_versus_someone",
    a1=bs.Family("bare_frame", "bare_frame_subject_swap", lambda i, pos: f"{'Nobody' if pos else 'Somebody'} near the {_obj(i)} told, of course,"),
    a2=bs.Family("met_frame", "met_frame_subject_swap", lambda i, pos: f"{'Nobody' if pos else 'Somebody'} the {_adj(i)} {_agent(i)} knew had warned, of course,"),
    p_donor=lambda i, pos: f"{'Nobody' if pos else 'Somebody'} near the {_adj2(i)} {_obj(i)} told, of course,",
    a1_suffix=lambda i: " told, of course,",
    a2_suffix=lambda i: f" the {_adj(i)} {_agent(i)} knew had warned, of course,",
    directions=('negative_to_affirmative', 'affirmative_to_negative'),
    kinds=('negative', 'affirmative'),
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
