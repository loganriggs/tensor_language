# MLP0 finite-response head-service preregistration

**Registered:** 2026-09-01 18:23 UTC

**Owner / rung:** Codex / 417

**Parent:** rung402 exact centered interaction head carriers; rung416 geometric calibration

**Claim level:** finite-distribution mechanism identification only; no compression or adoption

## Question

Can different attention0 heads write different centered MLP0 interaction vectors that block1 treats as the same
functional service? The object is the downstream response, not raw head labels or output covariance.

## Frozen intervention and response tensor

Reuse rung402's exact split `I = I_numeric + sum_h I_h` with all `T/C/S`, explicit RMS residual, native attention0,
raw-token reinjection, MLP bias, and the first-value bus fixed. For each document-position and head define:

- `ZERO = native_mlp0 - I`;
- `NUMERIC = ZERO + I_numeric`;
- `SINGLE_h = NUMERIC + I_h`;
- `FULL = native_mlp0`;
- `DROP_h = FULL - I_h`.

Run the exact block1 prefix from each MLP0 write and record its 1,152-dimensional attention1 and MLP1 writes. Define
`R_single_h = consumer(SINGLE_h)-consumer(NUMERIC)` and
`R_drop_h = consumer(FULL)-consumer(DROP_h)`. Also retain the exact action `I_h`.

Use the frozen 96 FIT and 96 SELECT documents and positions 64:256. FINAL stays sealed. Head3 is fixed in advance as
the dominant rung402/416 producer. A seeded permutation of document-position rows independently within each other-head
column is the response negative control; it preserves marginal norms and head spectra while breaking shared examples.

## Computations

For action and for every response consumer/background, form the 9-by-9 head Gram matrix after flattening
document-position and residual output dimensions. Report normalized Gram correlations between FIT and SELECT and the
singular-value head rank: minimum `r` reaching 90% squared energy and top-two energy fraction.

Fit least-squares regression on FIT to predict head3's flattened vector from a linear combination of the other eight
heads, with one scalar coefficient per other head and an intercept. Evaluate SELECT squared-error `R2` against the
FIT head3 mean. Repeat for raw actions and the seeded shuffled-response control. Report coefficients and per-consumer,
per-background results; no pooling may hide a failed consumer/background.

## Frozen predictions

1. **A — exact and live.** The head terms plus numerical remainder reconstruct parent `I` at relative MSE `<=1e-8`;
   direct block1 native replay has attention1 and MLP1 maximum absolute errors `<=2e-5`; every response has nonzero
   RMS `>=1e-6`; FIT/SELECT documents and permutation indices are disjoint/auditable.
2. **B — downstream response has a shared head service.** In at least three of the four consumer/background cells,
   response head-rank90 is at least two smaller than action head-rank90 **or** response top-two head energy is at
   least `.15` larger than action; and each passing response Gram has FIT/SELECT upper-triangle correlation `>=.80`.
3. **C — redundant producer prediction.** In both singleton/removal backgrounds and for both consumers, held-out
   head3 response reconstruction from the other eight heads has `R2>=.50`, exceeds held-out action reconstruction by
   `>=.20`, and exceeds its matched shuffled-response control by `>=.30`.
4. **D — one equivalence survives the choice of consumer and operating point.** The largest absolute tail coefficient
   has the same head in all four response regressions, every pair of fitted response coefficient vectors has Spearman
   correlation `>=.75`, and the four head3 response reconstruction `R2` values span no more than `.20`.

**Strong null:** A fails; all response head-rank90 values are no smaller than action; every head3 response `R2<=.20`;
any shuffled control is within `.05` of its matched real response; or no response cell simultaneously beats action
and shuffle.

## Decision and literal price

- A+B+C+D with no null identifies a finite block1-defined redundant head service. Next fit its shared response basis,
  factor the dual input readers, and validate grouped physical CE before naming or adopting it.
- A+B with C/D failure means the response is low-dimensional collectively but not a head3-redundancy relation; move
  to sub-head residual directions without claiming heads compute the same thing.
- A holds but B/C fail: attention0 `I_h` paths are functionally distinct for block1. Move to the cross-head double-QK
  shared-half factorization and repeat response-defined tests at attention1.
- A failure repairs the instrument only.

This diagnostic retains every native parameter and deploys no new basis. The reported 9-by-9 Grams, spectra, and
eight regression coefficients per cell are analysis statistics, not a compressed model. No rank or regularization
sweep follows a miss.
