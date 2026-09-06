"""lexical_number.pp_intervener -- number carried FROM A TOKEN, read at the same slot as coordination.

This exists to close one specific alternative explanation, and it is worth saying so plainly
rather than presenting it as a fresh discovery.

`das_coordination_number_transfer_v1` found that CONSTRUCTED number (built by `and`/`or` across
two singular nouns) and ANTECEDENT number (possessive `their`/`his`) occupy near-orthogonal
rank-1 directions at resid:18 -- cosine 0.072, transfer 0.018 / 0.030 both ways. A shared-token
artefact cannot explain that, because the vocabularies are disjoint. But a READ-POSITION
difference can: coordination is read at an auxiliary after the second noun, possessive at a
pronoun slot, and differing slots depress transfer in both directions exactly as distinct
features would. Symmetry does not discriminate them.

This behaviour matches the slot and varies only the cue's locality:

    A1  "The pilots near the harbor"  -> " were"  /  "The pilot near the harbor"  -> " was"
    A2  "In the report the pilots beside the tower" -> " were"
    P   swap the HEAD noun lexically, keeping its number; the final token is untouched

The number lives on a token (`pilots`), three tokens back, and the answer is the SAME auxiliary in
the SAME position as `coordination_agreement.and_vs_or`. So a transfer test between them holds
read position fixed and varies only whether the number was copied from a token or composed by a
connective. That is the comparison the coordination result actually needed.

Novelty is in the comparison, not the behaviour: agreement across a prepositional intervener is a
well-known behaviour class and `subject_verb.number_agreement` is already canonical here. It is
re-screened in this construction because the canonical module pins a drifted dossier hash and will
not build, so no usable direction can be obtained from it.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
O = lex._OBJECTS
_head = lambda i: R[i][0]
_pp = lambda i: O[(i + 7) % len(O)]
_pp_b = lambda i: O[(i + 17) % len(O)]
_alt_head = lambda i: R[i][1]

SPEC = bs.BehaviourSpec(
    task_id="lexical_number.pp_intervener",
    vocabulary=(" were", " was"),
    generator_role="generate_linked_lexical_number_pp_fit_panels",
    answer_role="score_jointly_tokenized_were_versus_was",
    a1=bs.Family("bare_frame", "bare_frame_head_number_swap",
                 lambda i, pos: f"The {_head(i)}{'s' if pos else ''} near the {_pp(i)}"),
    a2=bs.Family("report_frame", "report_frame_head_number_swap",
                 lambda i, pos: f"In the report the {_head(i)}{'s' if pos else ''} beside the {_pp_b(i)}"),
    p_donor=lambda i, pos: f"The {_alt_head(i)}{'s' if pos else ''} near the {_pp(i)}",
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
