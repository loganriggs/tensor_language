"""quantifier_number.each_vs_all -- number set by the QUANTIFIER over a plural noun.

The noun is PLURAL in every row. `Each of the pilots was`, `All of the pilots were`. So the head
of the partitive is invariant and cannot be the cue; the quantifier three tokens back decides the
auxiliary. Like `coordination_agreement.and_vs_or`, there is no local morphological cue at the
prediction site -- but where coordination COMPOSES number from two singulars, this one OVERRIDES
the number a plural noun already carries.

    A1  "Each of the senior pilots"       -> " was"   /  "All of the senior pilots"   -> " were"
    A2  "In the report each of the guards"  -> " was"   /  "All of the guards"   -> " were"
    P   swap the adjective; the quantifier and the final noun token are untouched

Authored to extend the one cross-behaviour comparison in this corpus that is currently
interpretable. `das_slot_confound_calibration_v1` showed that DAS transfer between behaviours is
only readable when their answers sit at the SAME read position, and this answers at the same
auxiliary slot as `coordination_agreement.and_vs_or` and `lexical_number.pp_intervener`. Those two
transfer at 1.053, so number appears to be one direction reached by two routes. This adds a THIRD
route -- override rather than composition or copying -- at the same slot, where a transfer test is
interpretable rather than confounded.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
_noun = lambda i: R[i][0]
_alt_noun = lambda i: R[i][1]
_noun_b = lambda i: R[(i + 15) % len(R)][0]
A = lex._ADJECTIVES
_adj = lambda i: A[i]
_alt_adj = lambda i: A[(i + 9) % len(A)]
_adj_b = lambda i: A[(i + 5) % len(A)]

SPEC = bs.BehaviourSpec(
    task_id="quantifier_number.each_vs_all",
    vocabulary=(" was", " were"),
    generator_role="generate_linked_quantifier_number_fit_panels",
    answer_role="score_jointly_tokenized_was_versus_were",
    a1=bs.Family("bare_frame", "bare_frame_quantifier_swap",
                 lambda i, pos: f"{'Each' if pos else 'All'} of the {_adj(i)} {_noun(i)}s"),
    a2=bs.Family("report_frame", "report_frame_quantifier_swap",
                 lambda i, pos: f"In the report {'each' if pos else 'all'} of the {_adj_b(i)} {_noun_b(i)}s"),
    p_donor=lambda i, pos: f"{'Each' if pos else 'All'} of the {_alt_adj(i)} {_noun(i)}s",
    a1_suffix=lambda i: f" {_noun(i)}s",
    a2_suffix=lambda i: f" {_noun_b(i)}s",
    directions=("each_to_all", "all_to_each"),
    kinds=("each_singular", "all_plural"),
    p_generator_role="adjective_lexical_rewrite",
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
