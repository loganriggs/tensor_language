#!/usr/bin/env python3
"""verb_particle_fb_up_down.woke_vs_calmed -- does a site carry a verb's selection (woke -> `up`, calmed -> `down`) in the FRAME-B sentence shape?

`Everyone agreed the pilot woke, frankly,` obliges ` up`; `... calmed, frankly,` obliges ` down`. Both cue -> token mappings are reused verbatim from verb_particle_up_down, which lives in the OTHER frame. v337 measured that reusing both mappings across a frame change does NOT fuse (0.0191 against 0.1163 for the same reuse within a frame), so this is authored as a NEW circuit under the frame-bound account -- and it is also a further test of it, since a fusion here would refute what v337 just measured.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v339 frame-B batch).
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
    task_id="verb_particle_fb_up_down.woke_vs_calmed",
    vocabulary=(' up', ' down'),
    generator_role="generate_linked_verb_particle_fb_up_down_fit_panels",
    answer_role="score_jointly_tokenized_up_versus_down",
    a1=bs.Family("direct_frame", "direct_frame_verb_swap", lambda i, pos: f"Everyone agreed the {_agent(i)} {'woke' if pos else 'calmed'}, frankly,"),
    a2=bs.Family("recall_frame", "recall_frame_verb_swap", lambda i, pos: f"The {_adj(i)} report said the {_agent(i)} {'woke' if pos else 'calmed'}, frankly,"),
    p_donor=lambda i, pos: f"Everyone agreed the {_adj2(i)} {_agent(i)} {'woke' if pos else 'calmed'}, frankly,",
    a1_suffix=lambda i: ", frankly,",
    a2_suffix=lambda i: ", frankly,",
    directions=('woke_to_calmed', 'calmed_to_woke'),
    kinds=('woke', 'calmed'),
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
