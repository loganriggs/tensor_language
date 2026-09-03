# Rung 543: unique four-closer pending-opener counterfactual rows

**Frozen:** 2026-09-03 15:21 UTC, before building these rows and before any model outcome

## Why the old rows cannot be extended

R542 found no exact prompt leakage between FIT and SELECT, but the deterministic lexical cycle repeated prompt pairs
inside every split. The nominal 48 FIT examples contained 18 unique prompt pairs; each nominal 16-row held-out split
contained eight. R538--R540 keep their conclusions after exact-prompt deduplication, but their unopened FINAL_TEST and
OOD rows are retired. This rung creates a new authority rather than changing or silently reusing those rows.

The prior target also used only `)` versus `"`. R540 then learned a direction closely aligned with that binary output
logit contrast. The replacement uses four single-token opener/closer types: `()`; `[]`; `{}`; and quotes. A candidate
that merely points toward one closer token should not transfer across all ordered type pairs.

## Declared variable and five constructions

The proposed variable is the type of delimiter that is still open at the final prompt position.

Two constructions change that variable and the correct next token:

1. `direct_type_substitution`: replace one opener while leaving every other token fixed.
2. `completed_then_reopened_order`: complete one delimiter type and then open another; reverse their roles while
   preserving the ordinary-word multiset.

Three constructions preserve the pending type and correct closer:

1. `surface_paraphrase`: change the lexical scaffold and content.
2. `distance_shift`: move the opener-to-final distance by inserting neutral content.
3. `nonopener_punctuation_substitution`: change one comma to a colon before an unchanged opener.

Every semantic group contains all five constructions. `group_id` is the SHA-256 hash of its actual prefix, word tuple,
opener pair, and template choice, not a loop counter. FIT, SELECT, FINAL_TEST, and OOD use disjoint prefix and word
pools. Planned group counts are 96, 48, 48, and 48, for 1,200 rows total.

## Fail-closed row checks

The builder must stop unless all of these hold:

- every base and donor text round-trips through the tokenizer;
- all four answer tokens and all opener tokens are single GPT-2 tokens in their used positions;
- every answer-changing row changes the correct answer, and every answer-preserving row does not;
- direct substitution and non-opener punctuation rows differ at exactly one token and keep equal length;
- each family has exactly the planned split counts;
- every prompt pair is unique within the complete dataset;
- no base or donor token sequence appears in two rows or across splits;
- every `group_id` has exactly one row from each of the five families;
- the generated JSON and receipt contain no logits, losses, activations, or model outcomes.

The strict sequence-uniqueness requirement is intentionally stronger than necessary. It gives downstream bootstrap
and train/test code an unambiguous independent unit and prevents a future nominal group identifier from hiding
pseudoreplication.

## Future causal gates, not run here

The first model rung on these rows must repeat capability and complete-state site ceilings using FIT and SELECT only.
No learned subspace is allowed before both answer-changing families and all three answer-preserving families have live
complete-state effects under exactly the same patch semantics.

Every later projector must report its overlap with the endpoint-readout span

$$
\mathcal R = \operatorname{span}\{w_{)},w_{]},w_{\}},w_{\texttt{"}}\}
\cap \mathbf 1^\perp,
$$

equivalently any three independent pairwise differences among the four output weight vectors. The primary comparison
will be ordinary DAS versus DAS constrained to the orthogonal complement of $\mathcal R$. If only the ordinary fit
works, the result is an output-steering direction, not an identified pending-opener variable. A separate upstream
endpoint—such as a downstream head's attention to the unmatched opener—should also be measured so fitting and testing
are not the same closer-logit margin.

This rung uses CPU only, loads no model, and opens no old or new outcomes.
