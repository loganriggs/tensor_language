#!/usr/bin/env python3
"""case_he_him.knew_vs_thanked -- does a site carry a verb's COMPLEMENT TYPE (knew -> clause -> `he`, thanked -> object -> `him`) across a parenthetical to the pronoun's case?

`The pilot near the lantern knew, of course,` obliges ` he`; `The pilot near the lantern thanked, of course,` obliges ` him`. Case family (pronoun_case_coordination me/I failed rows 4/5 at v217), NEW readout token pair he/him keyed by the verb's subcategorization rather than by coordination.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v239 batch).
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
    task_id="case_he_him.knew_vs_thanked",
    vocabulary=(' he', ' him'),
    generator_role="generate_linked_case_he_him_fit_panels",
    answer_role="score_jointly_tokenized_he_versus_him",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} {'knew' if pos else 'thanked'}, of course,"),
    a2=bs.Family("alt_frame", "alt_frame_verb_swap", lambda i, pos: f"The {_adj(i)} {_alt(i)} {'hoped' if pos else 'praised'}, of course,"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} {'knew' if pos else 'thanked'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('clausal_to_transitive', 'transitive_to_clausal'),
    kinds=('clausal', 'transitive'),
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
