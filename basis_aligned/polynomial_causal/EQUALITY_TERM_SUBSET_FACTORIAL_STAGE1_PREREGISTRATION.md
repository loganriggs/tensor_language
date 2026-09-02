# Equality-term subset factorial, stage 1: natural-text identification

Status: prospectively frozen design for rung 457. No subset outcome has been computed. This document authorizes
implementation and outcome-blind software tests, but not execution until the implementation is committed, audited,
and enqueued through the managed GPU runner.

## Research question

Four attention heads—L5H5, L7H3, L8H3, and L8H4—contain an exact term that follows the same token-equality relation.
Previous experiments changed all four terms together. This experiment asks:

1. which terms restore the same part of the copy/induction behavior and are therefore redundant;
2. which terms only work together because later layers combine their writes;
3. whether the answer changes with repeat distance or with the number of earlier matching tokens; and
4. whether a stable natural-text grouping is strong enough to justify a later interchange test on code.

This directly addresses cross-head grouping, extraction, interaction-aware removal, held-out document stability, and
the first half of OOD identification. It does not identify an MLP consumer yet. Later-module capture and causal patch
repair are a separately gated follow-up because adding them to the first factorial would complicate the instrument
before the basic subset effects are known.

Rank, parameter count, weight reconstruction, and aggregate CE preservation are not discovery criteria here. The
object being tested is a task-conditioned computation already defined algebraically.

## Prior exposure and honest claim level

The rows are the already-frozen `final_natural` role from
`induction_equality_tensor_final_ood_v2_rows_receipt.json`. Its all-four removal and extraction endpoints have already
been measured. The 14 non-endpoint subsets have not been measured. Consequently:

- stage 1 is an **identification/discovery** experiment on unopened subset outcomes, not a new one-shot final test;
- known all-four endpoint values are liveness references, not prospective predictions;
- no group is adopted from stage 1 alone; and
- the `ood_code` subset outcomes must remain unopened until the natural-text grouping rule is frozen. Stage 2 then
  becomes the OOD confirmation rather than a second dataset used to choose the grouping.

Historical whole-head mean ablations are background evidence only. They do not select a subset, a group, or a sign.

## Exact intervention

For query position `q`, source position `k`, and head `h`, the shared support is

`M[q,k] = 1[token[q] = token[k-1]] 1[1 <= k <= q]`.

The equality contribution from head `h` is

`e_h(q) = O_h sum_k M[q,k] score1_h(q,k) score2_h(q,k) value_h(k)`.

Let `H = {L5H5, L7H3, L8H3, L8H4}`. For every one of the 16 subsets `S` of `H`, execute:

- `remove:S`: replay the native selected heads and subtract `e_h` exactly for `h in S`;
- `extract:S`: set all four selected head writes to zero and restore `e_h` exactly for `h in S`.

The replay remains sequential: after a site is changed, every later layer receives the changed residual stream. This
is essential because the measured interactions are downstream composition, not interference in the local output
projection. Locally, the four writes are additive and all order-two-or-higher local write interactions are exactly
zero except floating-point rounding.

The implementation must enumerate the subsets in a fixed four-bit order and store every document-by-cell sufficient
statistic. It may not retain tokens, logits, or hidden states in the published result.

Bit 0 is L5H5, bit 1 is L7H3, bit 2 is L8H3, and bit 3 is L8H4. The exact arm order is `native`, then
`remove:0000` through `remove:1111`, then `extract:0000` through `extract:1111`. The SHA-256 of those 33 names joined
by a zero byte is `f8be9a80cc5451cda8c10ecbf1a025e856d9bc15b519c3284337d0a5d93d0b79`.

## Frozen rows and task conditions

Stage 1 uses exactly:

- role: `final_natural`;
- row file: `.rowcache_induction_equality_tensor_final_ood_v2/final_natural.pt`;
- file SHA-256: `5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759`;
- row-tensor SHA-256: `01d4fed403064b25112fdccdc5a2fed744e217ff071aeaa692c0429cdd027b0f`;
- 192 documents, one 257-token row per document; and
- scored query positions 64 through 255 inclusive.

The existing frozen conditions are:

| Condition | Meaning | Tokens | Documents |
|---|---|---:|---:|
| matched positive | nearest earlier occurrence of the query has the correct next token, balanced against a negative | 225 | 121 |
| matched negative | similar position/distance/frequency/repeat-count, but nearest earlier query has a different next token | 225 | 131 |
| all positive | every scored position satisfying the nearest-successor copy rule | 3,084 | 191 |
| off target | scored positions outside the positive relation and outside the matched negatives | 33,555 | 192 |
| all | every scored position | 36,864 | 192 |

Before any model execution, the implementation must deterministically partition `all positive` in two additional
ways using token IDs only:

| Condition | Exact rule | Tokens | Documents |
|---|---|---:|---:|
| near positive | distance to the nearest earlier equal token is at most 16 | 719 | 160 |
| far positive | that distance is greater than 16 | 2,365 | 191 |
| one-predecessor positive | exactly one earlier occurrence of the query token | 1,366 | 190 |
| multiple-predecessor positive | at least two earlier occurrences | 1,718 | 185 |

The two distance masks must be disjoint and union to `all positive`; the predecessor-count masks must do the same.
Counts, document supports, and mask hashes are integrity checks. These extra conditions test whether terms that look
redundant in the pooled effect actually specialize by context.

## Quantities computed

For each condition `c`, define extraction recovery in CE units as

`y_c(S) = CE(extract:empty, c) - CE(extract:S, c)`.

Positive values mean the restored equality terms improve prediction relative to deleting the four heads. Define
removal damage as

`r_c(S) = CE(remove:S, c) - CE(remove:empty, c)`.

Positive values mean removing those terms hurts prediction. CE is averaged over the exact tokens in condition `c`.
Native-to-arm KL and top-1 changes are reported as diagnostics, not identification gates.

For every nonempty term set `T`, compute the finite interaction

`d_c(T) = sum_{S subseteq T} (-1)^(|T|-|S|) y_c(S)`.

For a pair, negative `d` is redundancy, positive `d` is complementarity, and a value near zero is additivity. Shapley
values may summarize how the complete recovered effect is allocated, but native heads are not promoted to semantic
units merely because a Shapley value is large.

The primary prespecified comparisons are:

- the pair `P = {L8H3,L8H4}`, because both are in layer 8 and prior work isolates their correct successor edge;
- the early block `A = {L5H5,L7H3}` versus the layer-8 block `P`; and
- context specialization: changes in singleton recoveries and pair interaction between near/far and
  one/multiple-predecessor positives.

## Opposing predictions and decision rules

The experiment does not assume that same input relation means same circuit. It distinguishes these outcomes.

### Prediction A: instrument identity and liveness

All row, mask, checkpoint, source, and subset identities must match their frozen hashes and counts. `remove:empty`
must replay native logits with relative squared error at most `1e-12`. Every analytical arm must use replacement
dispatch at exactly sites 5, 7, and 8 and native attention everywhere else. The full extraction/removal endpoints must
have the same signs as the already-open v2 result. Any failure invalidates the run.

### Prediction B: redundancy versus complementarity of L8H3/L8H4

On `all positive`, classify the pair only if its interaction clears the measured simultaneous 95% interval and the
order-two numerical floor of `.006 nat` in magnitude:

- `d(P) <= -.006`: the two terms provide overlapping downstream benefit;
- `d(P) >= +.006`: later computation uses them complementarily; or
- `|d(P)| < .006` or the interval crosses zero: their pooled effects are additive/unresolved.

These are opposing outcomes, not a pass bar selected for one favored story.

### Prediction C: cross-layer composition

Define

`I_AP = y(A union P) - y(A) - y(P) + y(empty)`.

Classify `A` and `P` as redundant, complementary, or additive/unresolved with the same signed `.006 nat` rule and a
simultaneous 95% interval. This asks whether the early and layer-8 terms are alternative providers or stages whose
writes only become useful together downstream.

### Prediction D: context specialization

For each singleton and for pair `P`, compare near versus far and one versus multiple predecessors. A specialization
proposal requires both:

1. an absolute between-condition difference of at least `.012 nat` in recovered effect or pair interaction; and
2. the sign and head ordering to agree in both fixed 96-document halves wherever each half has nonzero support.

Otherwise the pooled grouping is retained only as a screen, with no context-specific split.

### Prediction E: stable identification

A natural-text grouping proposal is eligible for stage 2 only if:

- its redundancy/complementarity class agrees in the two fixed document halves;
- the relevant effect clears its numerical floor in both halves;
- the four singleton Shapley allocations have Spearman correlation at least `.70` between halves; and
- the proposal is not explained by a same-sized effect on matched-negative and off-target conditions.

The final clause is operational: report the proposed group's full signed response vector over every frozen condition.
It does not require copy-specificity, because prior evidence says equality copying is a broad service; it requires an
honest statement of which other conditions the proposed group affects.

## Uncertainty, numerical floors, and nulls

Use 20,000 document-cluster bootstrap draws shared across all subsets and conditions. The implementation must report
point estimates and simultaneous 95% intervals over all registered interactions and comparisons. Repeatability must
also be measured from an exact duplicate arm or an equivalent no-change replay. Unless the measured floor is larger,
minimum interaction magnitudes are `.006`, `.0085`, and `.012 nat` for orders 2, 3, and 4 respectively.

The strong null is any of:

- the replay, dispatch, row, mask, or endpoint liveness check fails;
- all singleton and pair differences are below the measured noise floor;
- the primary pair or cross-layer classification changes sign across document halves; or
- extraction behavior is dominated by matched-negative/off-target damage without a stable condition-dependent
  distinction.

For the last rule, “dominated” means that the absolute full-set recovered effect on `all positive` is no larger than
its absolute effect on either `matched negative` or `off target`, and no near/far or one/multiple comparison meets the
registered `.012 nat` stable-specialization rule. This is deliberately a broad-service diagnostic rather than a
copy-specificity requirement: prior results already show that equality copying is reused outside induction.

A null is informative: it would reject the four native equality terms as stable groupable units and redirect the
next decomposition below head-term grain, toward shared Q/K features, value features, or downstream-reader-defined
coordinates.

## Literal price and claim boundary

This is a diagnostic intervention experiment, not a deployed replacement. Its execution price is 32 analytical
configurations plus a native/replay integrity arm over 192 documents; the receipt must report actual forward count,
runtime, and peak GPU memory. It earns no parameter or storage saving.

Stage 1 can propose a grouping. It cannot adopt one. Before an adoption-level claim, the proposed grouping must be
frozen and then pass:

1. the same signed classification on the still-unopened `ood_code` subset outcomes;
2. within-group versus between-group interchange on natural occurrences; and
3. a downstream patch test identifying which later attention or MLP computation uses the grouped variable.

Only after those tests would it be meaningful to ask whether the identified circuit has a simpler executable
representation.
