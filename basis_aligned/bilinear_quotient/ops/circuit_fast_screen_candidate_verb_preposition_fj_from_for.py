#!/usr/bin/env python3
"""verb_preposition_fj_from_for.prevented_vs_blamed -- does a site carry the prevented/blamed preposition selection in this frame?

`Last winter the pilot prevented, admittedly,` obliges ` from`. BREADTH cell: authored to widen an existing corpus entry's pool rather than to add an entry, after v375 showed frame copies merge and v377/v379 measured breadth per entry.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v381 breadth batch).
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
    task_id="verb_preposition_fj_from_for.prevented_vs_blamed",
    vocabulary=(' from', ' for'),
    generator_role="generate_linked_verb_preposition_fj_from_for_fit_panels",
    answer_role="score_jointly_tokenized_from_versus_for",
    a1=bs.Family("temporal_frame", "temporal_frame_verb_swap", lambda i, pos: f"Last winter the {_agent(i)} {'prevented' if pos else 'blamed'}, admittedly,"),
    a2=bs.Family("season_frame", "season_frame_verb_swap", lambda i, pos: f"During the {_adj(i)} winter the {_agent(i)} {'prevented' if pos else 'blamed'}, admittedly,"),
    p_donor=lambda i, pos: f"Last winter the {_adj2(i)} {_agent(i)} {'prevented' if pos else 'blamed'}, admittedly,",
    a1_suffix=lambda i: ", admittedly,",
    a2_suffix=lambda i: ", admittedly,",
    directions=('prevented_to_blamed', 'blamed_to_prevented'),
    kinds=('prevented', 'blamed'),
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
