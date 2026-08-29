# Block-3 native-gate subset v1 — validation implementation amendment

Frozen after the fit-stage artifacts at commit `90eb59a6` were sealed, but before any
`n192_skip7000` validation tensor, local metric, cut state, logit, or target was loaded
by this assay.

This amendment fixes a staged implementation of the already registered validation.
It does not change either budget, selected support, decoder, control, threshold, data
role, or final-replication rule.

## Why a stage-zero validation is necessary

The fit receipt reports the regression objective in which `uu`, `uv`, `vu`, and `vv`
are stacked as separate targets.  That statistic cannot determine the NRMSE of the
deployed all-term write, because typed target and error cross terms were not retained.
Validation stage V0 therefore evaluates the executable sum directly before spending a
full 16-mask cube.

## Frozen V0 role and arms

- role: canonical collision-separated `n192_skip7000`, containing 192 rows from
  79 source documents;
- positions: logits/states 64--255 with targets 65--256;
- unit of uncertainty: source document;
- bootstrap: 2,000 paired document-resampling draws, seed `2026082908`;
- batch size: 4;
- model dtype: float32.

The native prefix through attention 3 is computed once per physical batch.  Native
MLP3 is then called exactly once to define the teacher write.  Every intervention
replaces MLP3 on all positions 0--255; restricting replacement to the scored slice
would leak native writes into the causal context of later attention.  Every suffix arm
runs from cloned copies of the common post-attention residual, first-value state, and
initial residual:

1. native;
2. native bias only (`bias_only`), the registered all-four-term omission `z_Q`;
3. four singleton typed omissions, preserving the bias and the other three terms;
4. the activation-selected all-term program and four singleton typed replacements at
   K=256;
5. same-K all-term matched-random and permuted-label controls;
6. the activation-selected all-term mirror,
   `2 * native_write - activation_write`.

`bias_only`, not the literal zero tensor, is the registered all-term denominator: the
frozen formula is `z_Q=b`.  “Zero the bilinear write” must never be mislabeled as “zero
the entire biased MLP output”.  The native bias is included exactly once in every arm.

K=256 is evaluated first.  K=512 is opened only if K=256 is not validation-eligible;
it adds the K=512 all-term, four singleton replacements, two controls, and mirror while
reusing the native and omission stakes.  This prospective rule changes the original
phrase “smallest passing K” to **smallest validation-eligible K**: validation screens
the registered local, final, control, mirror, and singleton-materiality conditions;
the complete mask conditions remain promotion gates rather than selection criteria.

## Frozen measurements

For each fitted all-term program, directly measure:

- summed local write NRMSE, using native bias-free write energy in the denominator,
  and the q90 of per-document NRMSE;
- separate `uu`, `uv`, `vu`, and `vv` NRMSE;
- maximum direct-K-product versus four-term polarization replay error;
- residual-state NRMSE after blocks 3, 4, 8, and 17;
- propagation ratio `error_at_cut / error_at_cut3`;
- final native-to-arm KL;
- KL divided by the bias-preserved all-term-omission KL stake;
- centered-logit NRMSE and cosine relative to the native-minus-bias-only response;
- signed CE difference and its one-sided bootstrap q95;
- top-1 agreement.

All token numerators and denominators are first accumulated within source document,
including all repeated rows from that document, then aggregated or bootstrapped over
the 79 source documents.  No row or token bootstrap is allowed.  Logits and per-token
states are not retained after their document sufficient statistics are accumulated.

## Frozen V0 decision rule

An activation-selected candidate is **validation-eligible** for the full typed cube
only if all of the following hold on validation:

1. summed local NRMSE is at most 0.20;
2. all-term KL/bias-only-KL is at most 0.20 point and 0.35 bootstrap q95;
3. CE-difference bootstrap q95 is at most 0.01 nat;
4. its point KL/bias-only ratio is lower than both its same-K random and
   permuted-label controls;
5. every singleton whose omission KL is at least 5% of the all-term omission KL has
   positive candidate recovery;
6. its mirror KL/bias-only ratio is at most 0.35 point.

Validation eligibility does not open final replication.  The selected candidate must
next run the remaining replacement masks and omissions on validation, completing the
16/15 cube and the already registered interaction/recovery checks.  Only then can one
candidate open on final, where the complete 16/15 assay is mandatory again.

If no candidate is validation-eligible but K=512 has local NRMSE above 0.20 while both
candidate and mirror KL/bias-only q95 are at most 0.35, complete the validation cube at
K=512 only to test the registered downstream-null interpretation.  It remains
noncomposable and cannot receive local-interface credit.

Otherwise the activation-fitted family stops after V0.  No validation cube and no
final role are opened for it.  This rejects family A at the registered budgets, not the
native-subset grammar: family F remains required before a grammar-level conclusion.

## Integrity and physical accounting

The authority must bind exact committed/pushed source blobs, the seven fit artifact
hashes and their terminal joins, the canonical row receipt and validation row hashes,
and the full checkpoint hash before the first validation forward.  It is published
before validation rows or outcomes are loaded.

The receipt reports measured calls separately by wave, exact arm, and arm family;
separately counts native typed Down calls, candidate typed decoder calls, and direct
program calls; and explicitly records zero full-model outer forwards/returns and zero
native MLP3 calls on student arms.  Per batch the prefix uses attention 0--3 and MLP
0--2 once, native MLP3 once, and each listed suffix arm uses blocks 4--17 once.  Fitted
all-term programs make one direct K-product call.  Typed candidate terms and native
terms are each computed once and reused to construct singleton arms; mirrors reuse the
already measured activation write.  No partial mask is priced as a deployed program.

For every arm, the suffix starts from an autonomous fork and runs blocks 4--17; no
post-MLP3 teacher state is reused.  State errors are captured after blocks 3, 4, 8, and
17.  In addition to native-state NRMSE, report error relative to the bias-only stake at
the same cut.  A downstream-null “decay” requires the cut17/cut3 error ratio to be below
0.5; raw norms at different cuts are not treated as comparable evidence.

Publication is create-only and receipt-last.  Source, fit inputs, rows, checkpoint,
model tensor contents, authority, and payload hashes are replayed immediately before
the terminal receipt.  Any drift publishes a failure artifact and no success receipt.
