#!/usr/bin/env python3
"""verb_preposition_sub_in_to.resulted_vs_amounted -- the same cue pair with the SUBJECT SLOT broken.

`Near the lantern the delays across the region resulted, of course,` obliges ` in`; `... amounted, of course,`
obliges ` to`.

An adversarial audit reported that 115 of 119 cells in this family put a subject that is a SINGLE-TOKEN, SINGULAR,
DEFINITE, ANIMATE, HUMAN OCCUPATIONAL noun from one 32-item table IMMEDIATELY before the cue -- and that no P row and
no A2 row in the entire family ever perturbs a token between the subject and the readout. So the interchange
difference could be carried by an `the {occupation} + verb` bigram or an animacy-conditioned feature, and nothing in
the current design could detect it.
This cell breaks THREE of those constants at once: the subject head is plural, inanimate, and three tokens from the
cue, with ` region` rather than an occupation in the pre-cue slot. That is deliberate and it is a FIRST PASS: if the
incumbent direction still carries this cell, all three are ruled out together at one screen's cost. If it does not,
the failure CANNOT be attributed to any one of them and follow-ups are needed -- registered here so the compound is
not read as a clean result either way. ` across` was chosen as the internal preposition because it is not in this
cell's answer vocabulary.
Authored 2026-09-10 from the audit; capability screened on CPU first.
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
    task_id="verb_preposition_sub_in_to.resulted_vs_amounted",
    vocabulary=(' in', ' to'),
    generator_role="generate_linked_verb_preposition_sub_in_to_fit_panels",
    answer_role="score_jointly_tokenized_in_versus_to",
    a1=bs.Family("inanimate_frame", "inanimate_frame_verb_swap", lambda i, pos: f"Near the {_obj(i)} the delays across the region {'resulted' if pos else 'amounted'}, of course,"),
    a2=bs.Family("inanimate_storm_frame", "inanimate_storm_frame_verb_swap", lambda i, pos: f"After the {_adj(i)} storm the delays across the region {'resulted' if pos else 'amounted'}, of course,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the delays across the region {'resulted' if pos else 'amounted'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('resulted_to_amounted', 'amounted_to_resulted'),
    kinds=('resulted', 'amounted'),
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
