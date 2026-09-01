# Attention0 Q/K normalizer common-congruence falsifier — rung 435

Date: 2026-09-01T21:33Z

## Question and exact computation

Rung 433b established that the 36 token/head/map RMS denominators are causally necessary, but rejected the registered
rank-8 token-table vocabulary. Before fitting a larger shared model, test whether their *weight-space quadratic forms*
have the algebraic structure required by a common set of squared features.

For each of four maps and nine heads, take the exact `128 x 1152` weight block `W_j` and form

`A_j = W_j^T W_j / 128`.

The squared denominator is `d_j(x)^2 = x^T A_j x + epsilon`. A common positive-square representation has

`A_j = U diag(c_j) U^T`, with `c_j >= 0`.

Let `M = mean_j A_j`. If `M` is positive definite, whiten each slice:

`C_j = M^(-1/2) A_j M^(-1/2)`.

Exact common congruence implies the symmetric `C_j` commute and are simultaneously orthogonally diagonalizable.
This implication is the falsifier; the converse is not claimed under approximate finite precision.

## Frozen metrics and controls

1. **Union range.** Stack all 36 `W_j` blocks. Report numerical rank at singular-value thresholds `1e-6`, `1e-8`,
   and `1e-10` relative to the largest, plus ranks retaining 90%, 95%, 99%, and 99.9% of squared singular mass.
2. **Whitened commutators.** For every unordered pair, report
   `||C_i C_j - C_j C_i||_F / (||C_i||_F ||C_j||_F)`.
3. **Observable joint-diagonalization residual.** For eight fixed positive coefficient vectors (seeds
   `43500..43507`), diagonalize `sum_j alpha_j C_j`. In each resulting basis, report the fraction of total slice
   Frobenius energy off the diagonal; retain the best of the eight. This is a diagnostic, not an optimized RJD solver.
4. **Matched null.** For seeds `435100..435107`, independently permute the 1,152 residual coordinates of each
   `W_j`, recompute that control's aggregate whitening, commutators, and eight-basis residual. This preserves each
   slice spectrum, PSD rank, and entry values while destroying cross-slice coordinate alignment. Comparisons use
   the empirical control distribution, never an absolute guessed window.

All eigendecompositions and accumulated diagnostics use float64. No token data, loss labels, or downstream results
enter this rung.

## Frozen predictions

### A — instrument

- `M` has minimum eigenvalue at least `1e-8` times its maximum;
- whitening residual `||mean(C)-I||_F/||I||_F <= 1e-8`;
- every factor-to-matrix reconstruction residual is at most `1e-10`;
- independently permuting coordinates preserves every slice trace and Frobenius norm to relative error `<=1e-10`;
- all 630 real pairs and all eight controls are finite.

Instrument failure withholds content.

### B — an exact undercomplete rank-256 common-square model is impossible

Both the `1e-8` numerical union rank and the 99%-energy union rank exceed 256. This is a weight-space lower bound,
not a claim about approximation on the finite token manifold.

### C — useful common-congruence structure

Both must hold:

- the real median normalized commutator is at most half the smallest of eight independently permuted-control medians;
- the real best joint-diagonalization residual is at most half the smallest control residual.

Passing licenses one common-positive-square physical arm, but does not license semantics or compression.

### D — weaker alignment above the matched null

Prediction C fails, but at least one of the two real metrics is at most 90% of the smallest matched-control value.
This records partial shared geometry without pretending it is the clean common basis needed by the proposed model.

## Strong null and routing

The strong null fires if the real median commutator and best diagonalization residual are each at least the median of
their eight matched controls. Then common congruence has no measured advantage over destroyed alignment and the
shared-positive-square family closes.

- A+B+C: run one physical common-positive-square arm, with complete price, against equal-price private forms and
  independently permuted coupling.
- A+B, C false, D true: shared geometry is weak. Do not fit R256. First test a block-structured basis by map family
  or the already observed head groups, with the partition frozen from independent results.
- A+B, C/D false or strong null: retain head/map-private normalizers and the successful shared score quotient; do
  not spend a rung tuning common-square rank.

This rung is algebraic identification only. It does not test OOD behavior, circuit extraction, selective removal,
composition, semantics, or a deployable complete artifact.
