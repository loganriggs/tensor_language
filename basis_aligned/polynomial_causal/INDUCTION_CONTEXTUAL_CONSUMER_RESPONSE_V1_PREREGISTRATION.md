# Induction contextual-consumer response screen V1

## Question and prior boundary

R594 and the live-clamp follow-up establish native selector/payload capability
at L5H5, L7H3, L8H3 and L8H4, but reject treating their complete joint write
as an answer-only circuit. A fixed post-result audit shows that three of four
answer-preserving joint cells retain the existing correct-answer CE and choice
bars while all four fail the existing full-vocabulary bar. This experiment asks
which downstream native sublayer first has a response that mediates that
non-answer collateral.

This is a response-restoration screen. It does not search or refit the four
producer sites, define a low-rank basis, discard output directions, identify a
unique consumer, extract a circuit, open SELECT/FINAL/OOD, or claim savings.

## Frozen rows and split

Load the immutable R585/R594 row authority without model outputs. Among FIT
directions with family `selector_payload_joint_answer_preserved`, sort the 72
group hashes and retain the first 24. Keep all four registered directed cells
per group, for 96 rows. The first 12 group hashes are DISCOVERY and the next 12
are CONFIRM. This rule is fixed before the new model run and is independent of
the prior outcome values. Each partition has 48 rows and 12 observations per
cell.

## Intervention semantics

For each endpoint, capture native equality terms and the complete native
attention and MLP writes at the registered final query. For a directed row,
install the donor's four equality terms with the existing live replacement
rule: at each producer site replace the current equality-supported contribution
with the frozen donor term, combining L8H3 and L8H4 in one FP32 transaction.

The `joint` arm lets every later module recompute. Each response-restoration arm
does the same joint edit, but at exactly one downstream sublayer replaces that
sublayer's complete final-query write with the native recipient write captured
on the unedited recipient. All other positions and modules remain live. The 19
fixed candidates are MLP8, then attention and MLP for layers 9 through 17.

Restoration uses a native oracle and removes the candidate's entire changed
response. A passing candidate is therefore a causal mediator screen, not an
independently generated or semantically pure consumer.

## Metrics and predictions

For every row and arm, relative to native recipient logits, record correct-token
CE damage, correct-minus-other margin change, correct-versus-other choice, and
full-vocabulary RMS change. The instrument must reproduce native and a self-term
live clamp within the existing `1e-3` maximum-absolute and `1e-5` relative
Frobenius bars; execute exactly the frozen row and forward counts; observe every
producer transaction; and show a nonzero joint edit and a nonzero restored
candidate response.

On DISCOVERY, a candidate is eligible only if every cell has mean CE damage at
most 0.10 nat and correct fraction at least 0.75. Among eligible candidates,
select the one maximizing the worst-cell fractional reduction of median
full-vocabulary RMS relative to `joint`. Ties within `1e-12` choose the earlier
candidate in the fixed order. If none is eligible, the scientific result is a
null without selecting a candidate.

Prediction B passes on CONFIRM only if the selected candidate independently:

1. has mean CE damage at most 0.10 nat and correct fraction at least 0.75 in
   every one of the four cells; and
2. reduces median full-vocabulary RMS by at least 25% relative to `joint` in
   every cell.

Prediction C, localization, passes only if B passes and both immediate adjacent
candidates (when present) fail B's complete conjunction on CONFIRM. This strict
condition distinguishes a localized response from a broad downstream erasure.
No alternate site, subset, gain, threshold, denominator, head, or rank is tried
after inspection.

Opposing predictions: a discrete downstream response mediates the contextual
collateral if one sublayer passes B and its neighbors fail C. If no candidate
passes B, the four-site signal is diffusely consumed or this restoration grammar
is inadequate. If many adjacent candidates pass B, mediation is not localized.

## Price and claim boundary

Maximum planned work is one native endpoint capture plus self-clamp, joint and
19 restore arms on 96 directed rows, batched at 32: 66 forwards and 2,112
sequence evaluations. The exact endpoint count and forward count are frozen in
the binding after row construction and checked at runtime.
There are zero fits, backwards, or weight updates. Save compact per-row metrics
and hashes; full logits need not persist after scoring.

All 545,902,902 model parameters, tokenizer, native row generator, donor terms,
native oracle writes, and the complete suffix remain charged. A pass advances
cross-boundary splitting and selective manipulation evidence only. It does not
establish stable identification, OOD prediction, extraction, reuse, or storage
and execution compression.
