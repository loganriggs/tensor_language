# Preregistration — is the score-implementation mismatch shape gauge-covariant? (CPU probe; parallel lane)

Date: 2026-09-03 00:20 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any covariance outcome; zero model forwards, CPU only,
on rung513's published statistics.

## Question

Rung513 (§2646) found no cross-action factor circuit variable, but its signed-
mismatch decomposition stores, for each of 18 keys (6 branch-subsets ×
3 action-pairs) at sites a11/m11 and 2 halves, WHICH of the 31 exact factor
Moebius terms carry the difference between two score implementations
(equal-share signed inner-product fractions summing to 1). The a11 shape is
Q-dominant (A11{Q} .40, {V} .28, {Q2} .25). This probe asks whether that SHAPE
is INVARIANT across keys: do all 18 mismatch fingerprints point the same way?
A positive answer characterizes the descent's "source-dependent internal
realization" constructively — the four score implementations differ in a fixed
factor subspace, only the magnitude varying. Descriptive on published
statistics; NO physical or circuit claim. Codex's rung514 (sparse-signed-sum
search) is untouched.

## Computation (exact, deterministic)

Load rung513's receipt (sha 043dd563baa5ffe5bda57c7774dc76e4727add3b4351699cb997ecfd563179d5). For each site s in {a11, m11}, build
the fingerprint matrix F_s: rows = the 18 keys pooled over the two halves
(mean of half0/half1 share vectors), columns = that site's 31 factor terms.
L2-normalize each row. Report:

1. the mean pairwise cosine among the 18 rows (the covariance statistic);
2. the top-3 |share| coordinates per row and their agreement (how often the
   same 3 factors dominate);
3. a term-permutation control: independently permute the 31 columns within each
   row using 16 hash-fixed seeds (20260907+i) and report the permuted mean
   pairwise cosine q95;
4. an across-SITE check: cosine between a11's mean fingerprint and m11's.

## Frozen predictions

### pred_a — exact reproduction
Receipt sha matches; all 18 keys × 2 sites × 2 halves present with 31-term
share vectors each summing to 1.0 +/- 1e-6; the pooled a11 L::N-Z7 shares
reproduce the receipt to <= 1e-9.

### pred_b — the mismatch shape is gauge-covariant
For BOTH sites: mean pairwise cosine among the 18 key-fingerprints >= .70 AND
exceeds the term-permutation q95 by >= .10.

### pred_c — a stable dominant factor subspace
For BOTH sites: the same top-3 factor terms are shared by >= 12 of 18 keys
(two-thirds), naming the covariant subspace.

## Strong null
Fires if pred_a fails, or pred_b fails at either site (mismatch fingerprints
are key-idiosyncratic; the score-implementation difference has no fixed factor
shape and the sub-action source-dependence is genuinely unstructured). No bar
changes; the result is reported beside §2646 as a descriptive characterization
either way.

## Price
Zero model forwards; CPU < 15 s; one receipt JSON. Nothing deployed.
