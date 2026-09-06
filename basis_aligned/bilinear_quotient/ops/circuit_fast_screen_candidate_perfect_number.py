"""perfect_number.have_vs_has -- head noun number read at the auxiliary, with DISJOINT vocabulary.

This exists to break a confound in this lane's own strongest result, and it is worth stating that
plainly rather than presenting it as another behaviour.

Three number behaviours now share one rank-1 direction at a matched auxiliary slot:
`lexical_number.pp_intervener` (copies number from a plural token), `coordination_agreement`
(composes it from two singulars) and `quantifier_number.each_vs_all` (overrides a plural).
Transfers were 1.053 and 0.600. But ALL THREE answer with the identical pair ` was` / ` were`.
A rank-1 direction that merely separates those two tokens would produce exactly those transfers
with no shared NUMBER feature at all. The comparison that would have caught it -- possessive
`their`/`his` -- differs in read slot too, and `das_slot_confound_calibration_v1` showed a slot
difference alone collapses transfer from 1.05 to 0.07, so that test could not discriminate.

This behaviour holds the slot fixed and changes the tokens. A fronted durative adjunct selects the
perfect, so the same head-noun number is read at an auxiliary in the same position, answering
` have` / ` has` instead:

    A1  "For years the leaders near the maple"  -> " have"  /  "... the leader ..."  -> " has"
    A2  "The reports say that for decades the leaders beside the tower" -> " have"
    P   swap the head noun lexically, keeping its number; the final token is untouched

Matched slot, disjoint vocabulary. A transfer against the was/were family is then interpretable as
evidence about number rather than about tokens -- the first such test this corpus can run.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
O = lex._OBJECTS
_head = lambda i: R[i][0]
_alt_head = lambda i: R[i][1]
_pp = lambda i: O[(i + 7) % len(O)]
_pp_b = lambda i: O[(i + 17) % len(O)]

SPEC = bs.BehaviourSpec(
    task_id="perfect_number.have_vs_has",
    vocabulary=(" have", " has"),
    generator_role="generate_linked_perfect_number_fit_panels",
    answer_role="score_jointly_tokenized_have_versus_has",
    a1=bs.Family("bare_frame", "bare_frame_head_number_swap",
                 lambda i, pos: f"For years the {_head(i)}{'s' if pos else ''} near the {_pp(i)}"),
    a2=bs.Family("report_frame", "report_frame_head_number_swap",
                 lambda i, pos: f"The reports say that for decades the {_head(i)}{'s' if pos else ''} beside the {_pp_b(i)}"),
    p_donor=lambda i, pos: f"For years the {_alt_head(i)}{'s' if pos else ''} near the {_pp(i)}",
    a1_suffix=lambda i: f" {_pp(i)}",
    a2_suffix=lambda i: f" {_pp_b(i)}",
    directions=("plural_to_singular", "singular_to_plural"),
    kinds=("plural_head", "singular_head"),
    p_generator_role="head_noun_lexical_rewrite",
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
