# Rung 581 preregistration: independent CPU audit of R580

**Frozen:** 2026-09-03, before any R580 scientific result exists or is opened

## Purpose and authority boundary

R581 independently audits the saved R580 native-capability result without loading the model, evaluating a prompt,
or opening FINAL_TEST/OOD. It cannot rescue a failed scientific gate, alter a threshold, or promote the induction
circuit. A held R581 says only that R580's saved evidence supports the verdict R580 reports under the frozen R580
rules.

The audit is bound to the reviewed R578 rows, receipt, builder, tests, and preregistration, and to the reviewed R580
preregistration, implementation, tests, and passing model-free dry-run by exact SHA-256. The future R580 result and
receipt are deliberately not available at freeze time. The result receipt must bind the result bytes when the audit
eventually runs.

## Independent reconstruction

Starting only from R578 authority plus R580's saved native sequence measurements, the audit must independently:

1. reconstruct the exact 108 FIT/SELECT semantic groups, 3,240 rows, and 3,024 unique prompt sequences;
2. verify that each saved sequence belongs to exactly one expected group and split and has the expected token-pair,
   registered answer, length, and final position;
3. verify both saved target cross-entropies from the saved log-normalizer and logits;
4. rebuild every row's base/donor sequence IDs, condition, correct-answer margins, cross-entropies, paired effect,
   answer-change flag, and stable identifiers;
5. rebuild all 108 four-cell factorial records and all 432 selected/neutral/contrast condition records;
6. rebuild every factorial, selector-by-payload interaction, relation-preserving control, selected-match necessity,
   selected-versus-neutral selectivity, and non-gated contrast diagnostic;
7. reproduce every one of the 2,000 SHA-defined group-cluster bootstrap draws for all 86 bootstrap cells, including
   the frozen lower/upper NumPy quantiles;
8. recompute the exact failed-clause list and terminal `held_capability_screen` or `scientific_null` verdict.

Numeric comparisons use absolute tolerance `1e-12` and zero relative tolerance. IDs, memberships, ordered group
lists, pass flags, failed clauses, and verdicts must match exactly. For each bootstrap cell, R581 records SHA-256 of
the complete integer draw-index matrix and the float64 bootstrap-statistic vector. It also records one combined hash
over all 86 cell records. These hashes make “all draws reproduced” checkable without copying millions of indices into
the audit JSON.

## Envelope checks and terminal rule

R581 requires the R580 result to report exactly FIT and SELECT, no forbidden split, 95 forwards, zero backwards,
zero weight updates, 3,024 unique sequences, the pinned checkpoint hash, and the frozen implementation/test/input
hashes. The R580 receipt must match the result SHA, checkpoint, verdict, prices, and split envelope. R580's
`instrument_passes` must be true; a malformed or incomplete artifact is an audit/integrity failure rather than a
scientific null.

The audit verdict is `held_independent_audit` only when every authority, membership, raw reconstruction, bootstrap,
aggregate, hash, price, split, and terminal-decision comparison holds. Any mismatch produces
`failed_independent_audit` with named failed checks; it never changes the independently recomputed scientific verdict.

## Pre-outcome tests and price

Before R580 runs, a model-free dry run must exercise the full 108-group authority with both a planted held fixture
and a planted scientific-null fixture. The held fixture must pass every gate. The null fixture must return a terminal
scientific null while retaining complete raw evidence. Focused tests must also independently verify the SHA draw
formula, group clustering, quantile conventions, all 86 expected bootstrap cell IDs, tamper detection, and the fact
that dry-run never reads a future R580 result.

R581 uses zero model forwards, zero backwards, zero fitted vectors, and zero weight updates. It is CPU-only.
