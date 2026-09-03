# Rung 527: exact centered MLP0 context-source interaction quotient

Status: prospectively frozen after rung 526's valid downstream-gradient token-grouping null and before any rung 527
checkpoint outcome is computed.

## Question and circuit target

MLP0's fixed-gain context-only branch has a replicated Shapley-average CE benefit of `0.3506` nat on FIT and
`0.4177` nat on SELECT.  Rung 517 split attention0 into five source relations, but 47–52% of the context-branch
energy remained in an unnamed “closing” term because the constant FIT expectation of the quadratic context function
was not assigned to its source pairs.

This rung corrects that exact accounting and asks a circuit question: do two different source interactions have the
same finite downstream effect, up to one signed scale, across documents and circuit families?  A passing pair merges
parts of MLP0 by downstream use; failure leaves the exact terms as anatomical bookkeeping.  This is not a rank,
reconstruction, SAE, quantization, variance, or parameter-count experiment.

## The 20 exact semantic terms

Use the frozen five attention0 source relations from rung 517:

`SELF, PREVIOUS, NEAR, DISTANT_SAME, DISTANT_OTHER`.

Let `B(x,y)=Down[(Left x) elementwise-multiplied by (Right y)]`, let `mu=E_FIT[e+a]`, and let
`delta_s=a_s-E_FIT[a_s]` be one centered source write.  With `gbar` equal to the frozen FIT mean squared
RMS-normalization gain, define:

- five linear terms `L_s = gbar [B(delta_s,mu)+B(mu,delta_s)]`;
- five self terms `Q_ss = gbar [B(delta_s,delta_s)-E_FIT B(delta_s,delta_s)]`; and
- ten unordered cross terms
  `Q_st = gbar [B(delta_s,delta_t)+B(delta_t,delta_s)
                - E_FIT(B(delta_s,delta_t)+B(delta_t,delta_s))]`, `s<t`.

The 15 quadratic expectation vectors are estimated once on the same 96 FIT documents used by rungs 400/401/517 and
then frozen.  SELECT, discovery, confirmation, and circuit labels cannot change these means.  The 20 terms sum to
the centered context function of the five semantic writes.  The difference from rung 401's exact deployed
context-only branch is retained as a separate **numerical remainder**; it contains attention arithmetic/rounding and
is never offered as a semantic circuit.  This makes every finite intervention exact relative to the real MLP0 output
while exposing whether the old 47–52% closing energy was only the omitted expectation.

## Data boundaries

- Moment fitting: the immutable 96 FIT rows from rungs 400/401/517.
- Discovery: documents `0:248`, split into `D0=0:124` and `D1=124:248`, with the fixed 32 discovery circuits.
- Confirmation: documents `500:1000`, split at 750, with the other 30 circuit families.  These rows and tags remain
  unopened unless predictions A and B pass.
- The model checkpoint, circuit member masks, matched `slice_control` masks, source-relation vocabulary, and all
  thresholds are hash-frozen by the runner.

For term `u`, circuit `c`, and document half `h`, define its finite signed circuit effect as

`F_h(u,c) = mean(NLL_remove_u - NLL_native on member_c)
            - mean(NLL_remove_u - NLL_native on slice_control_c)`.

This baseline subtraction prevents a term from looking circuit-specific merely because its member tokens are
generally harder.  Every value comes from a real MLP0 term removal followed by the complete nonlinear model suffix;
no gradient or sum-of-singletons approximation is used.

## A — exact, live instrument

Before model loading, planted vector-valued quadratic tests must satisfy all of the following:

1. the 20 centered terms plus a planted numerical remainder reconstruct the planted context function with maximum
   absolute error at most `1e-10`;
2. the 15 separately accumulated quadratic expectations sum to the complete expected quadratic function with
   maximum absolute error at most `1e-10`; and
3. a planted pair detector recovers exactly its known proportional pairs while circuit-coordinate permutations
   destroy them.

On the real model:

1. the semantic terms plus retained numerical remainder reconstruct rung 401's context branch with relative squared
   error at most `1e-12`;
2. the numerical remainder has at most `0.01` of the context-branch squared energy in each discovery half, replacing
   rung 517's 47–52% unnamed fraction;
3. the unmodified dispatcher endpoint reproduces direct native logits exactly and all callback/circuit supports
   match their frozen counts; and
4. every one of the 20 requested term removals changes MLP0's deployed write by nonzero RMS.

If any clause fails, stop before scientific interpretation and repair only that clause.

## B — a small discovery equivalence relation

For every unordered pair of distinct terms `(u,v)`, fit one signed scalar using only the 32 D0 circuit coordinates:

`beta(u <- v) = <F_D0(v),F_D0(u)> / <F_D0(v),F_D0(v)>`.

A pair passes discovery only if:

1. both terms have pooled circuit-effect RMS at least `.0005` nat;
2. `.25 <= abs(beta) <= 4`;
3. on D0, predicting `u` as `beta*v` has cosine at least `.90` and relative residual at most `.35`;
4. with the same scalar and no refit, D1 cosine is at least `.80` and relative residual at most `.50`; and
5. the reciprocal prediction also satisfies the same materiality, direction, and residual clauses.

Keep every passer; do not rank pairs.  Exactly 1–8 pairs passes B.  Zero is a strong scientific null.  More than 8
means the 32-circuit observation basis does not identify a small quotient, and the best eight may not be selected.

As a descriptive control, independently permute each term's 32 circuit coordinates with 16 frozen seeds and rerun
the detector.  The real candidate count must exceed the higher 95th percentile of these counts.  This control cannot
change thresholds or promote a failed real pair.

## C — held-out circuits and documents

Only if A and B pass, freeze term identities and `beta`, open the other 30 circuit families on documents `500:1000`,
and require in each half and pooled:

- both terms' circuit RMS is at least `.0005` nat;
- cosine is at least `.75`; and
- relative residual is at most `.55` in both directions.

At least one frozen pair must pass.  The scalar is never refit.  Zero closes this source-term quotient as
document- or circuit-family-specific.

## D — bidirectional physical equivalence

For each C-passing pair, use the confirmation batches to compare the exact target removal against an intervention
made with the other term:

- target `u`: compare `native - u` with `native - beta*v`;
- target `v`: compare `native - v` with `native - v_pred`, where `v_pred=(1/beta)*u`.

Both are inserted at the real MLP0 output and run through layers 1–17.  In each direction, the resulting 30-circuit
effect vector must match the target removal with cosine at least `.75` and relative residual at most `.55` in both
document halves and pooled.  Every substitution must be nonzero.  At least one pair must pass both directions.

This is finite downstream interchangeability.  Similar term vectors or removal-effect correlations without this
test do not count.

## E — a nontrivial quotient group

A D-passing pair must use different source supports or different operation types (`linear`, `self`, or `cross`).
Connected components are reported as quotient groups only when every within-component pair passes bidirectional
substitution and products of fitted scales around every cycle differ from one by at most 25%.  Otherwise report only
the passing pairs.  At least one nontrivial pair is required.

`strong_null = not (A and B and C and D and E)`.

Frozen routes:

- A false: repair only the named exactness, support, replay, or edit-liveness clause.
- B false with zero pairs: close pairwise proportional grouping of the five-source context terms; next use a finite
  prefix-by-continuation predictive-state object or move to another module, not finer source bins or rank.
- B false with more than eight: add independently defined downstream readouts; do not select the best eight.
- C false: the apparent discovery grouping is not reusable; localize which first consumer separates it.
- D false: response correlation is not physical interchangeability; preserve the null.
- A–E true: test the frozen pair on the registered OOD code corpus, joint composition with other adopted circuits,
  and literal executable pricing.

## Literal price and compute

The diagnostic reference adds five `1152`-vectors of source means and fifteen `1152`-vectors of quadratic
expectations: `20*1152 = 23,040` floating-point values.  These are analysis values, not deployed parameters.  The
20 terms can reuse the five Left and Right source projections; each term needs one elementwise product and one Down
projection.  Rung 527 removes no native parameters and makes no compression claim.

Discovery uses 62 four-document batches and 21 complete model forwards per batch (`native + 20 removals`), plus one
attention/source decomposition per batch: 1,302 model forwards.  Conditional confirmation uses 125 batches and the
same 21 arms: 2,625 more forwards.  If `q<=8` pairs reach physical substitution, the final phase adds `250q` model
forwards.  There are no model backward passes and at most 190 CPU pair comparisons.
