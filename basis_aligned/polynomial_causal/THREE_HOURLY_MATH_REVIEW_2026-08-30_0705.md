# Three-hour mathematical review — 2026-08-30 07:05 UTC

## Bottom line

The most useful mathematical shift is to treat **downstream causal response as the
coordinate system in which simplicity must pay rent**.  Activation cosine, HOSVD
energy, and local reconstruction can suggest a model, but the newest six-component
result shows they cannot choose it: the relation between direction cosine and causal
cross-effect changes sign across components.

I implemented the prerequisite for a better test: a source-closed collector for the
signed, per-document response tensor.  Its non-degenerate tiny-transformer test suite
passes 13/13.  It uses typed synchronous dispatch rather than hooks, rejects document
leakage, retains additive signed statistics, and accounts for every physical model
call.  The exact real experiment is now frozen prospectively in
`CAUSAL_RESPONSE_TENSOR_V1_PREREGISTRATION.md`.  No bilin18 outcome has been opened by
this work.

The three highest-return mathematical moves are now:

1. held-out shared/private factorization of the signed causal-response tensor;
2. a suffix-induced Fisher/observability quotient for early-layer simplicity;
3. finite Hankel-rank tests for the terminal circuits that plausibly implement small
   state machines.

## Evidence and corrected current state

The strict project ledger remains:

- certified stored model content: **5.348245316%**;
- named causal CE: **10.923302467%**;
- unexplained CE: **4.72714 nat = 89.076697533%**;
- terminal actions passing extraction, selective removal, and OOD together: **0/68**.

The six-component census covers 49 of 62 localized circuits.  A dominant shared
activation direction appears in 3/6 components, not a stable majority.  More
importantly, geometry does not determine cross-circuit causal effects: full-direction
cosine versus causal concentration has Spearman correlations `+0.4212` at `a8`,
`+0.4198` at `a16`, and `-0.5411` at `m16` on the common all-circuit instrument.  The
earlier `a8` value `+0.6611` came from a selected five-circuit subset and does not
reproduce on all sixteen.  Therefore “factor what is parallel” is not a lawful general
selection rule.

The ratio-only census was still useful for falsifying that rule, but it cannot support
a composable factorization.  Its absolute concentration ratios erase sign and scale
and do not add across interventions.  The largest immediate missing interface is the
signed source-by-target-by-document response tensor.

The GPU is idle.  Launch is blocked only by experiment integrity: an independent audit
found that the historical `census_lib` loads hidden state, fits PCA and means using all
rows, does not clear its PCA cache, and cannot certify exception-safe hook cleanup or
physical calls.  The new backend avoids that library.  It still needs a create-only
authority/execution wrapper and independent source-closure audit before a real launch.

## Priority 1 — shared/private factorization in causal-response space

### Exact object

The frozen response array has entries

\[
R_{pstd}=\operatorname{mean}_{i\in M_t}\Delta\mathrm{CE}_{psdi}
-\operatorname{mean}_{i\in O_t}\Delta\mathrm{CE}_{psdi},
\]

indexed by phase (p\), source circuit (s\), target circuit (t\), and source document
(d\).  Each response is computed from additive signed sums and counts.  The full
stored object also retains absolute sums as diagnostics.

Fit a hierarchy such as

\[
R_{pstd}\approx
\sum_{k=1}^{K_0} a_{pk}b_{sk}c_{tk}h_{dk}
+\sum_{g}\sum_{k=1}^{K_g}
  \mathbf 1[s\in g],a^{(g)}_{pk}b^{(g)}_{sk}c^{(g)}_{tk}h^{(g)}_{dk},
\]

where the first sum is a global shared library and each group (g) is a component or
learned branch.  This is a block-term/shared-dictionary tensor program: a factor is
priced once, while sparse source and target loadings say which circuits use it.  Unlike
top-k activation routing, its multilinear contractions remain a tensor network.

### Operational definition and assumptions

“Simpler” means lower literal serialized/executable price at matched held-out signed
response error, after quotienting CP scale/permutation and any block-basis gauges.
Choose ranks, groups, and sparsity on FIT documents only.  EVAL documents measure
prediction.  The quotient-Jacobian instrument from the previous review must detect no
unexplained local null directions before atoms are called separately editable.

This assumes intervention effects are stable enough across documents and sufficiently
low multilinear rank.  CE is nonlinear, so finite deletion effects need not add;
document factors may instead absorb topics, and a low-rank fit may be predictive but
not causally separable.  Components such as `a8`, `a16`, and `m16` may require different
structures rather than one universal hierarchy.

### Prediction beyond reconstruction

A real shared factor must predict an unmeasured source-to-target cell or fresh
finite-amplitude intervention, not merely reconstruct the cells used to fit it.  A
private branch must enable selective removal with lower unrelated-target change than
deleting the whole shared factor.  Factors should also transport to a held-out domain
and remain identifiable under document doubling.

### Cheapest falsifier

Collect the preregistered lawful tensor once.  Before a large search, compare on held-
out EVAL documents: independent per-component CP, a shared-plus-private block model,
and a parameter-matched dense linear baseline.  Reject the hierarchy if it does not
improve held-out signed-response prediction per stored scalar, or if its factor names
change under exact regauging/document resampling.  This costs CPU after collection.

## Priority 2 — suffix-induced Fisher/observability quotient

### Exact object

For an early write (z\in\mathbb R^{1152}), let the frozen suffix map it to logits
(\ell(z)\).  On a data point, let (J=\partial\ell/\partial z), and let

\[
F_{\rm CE}=\operatorname{diag}(p)-pp^\top
\]

be the softmax Fisher matrix.  The pullback

\[
G=\mathbb E[J^\top F_{\rm CE}J]

\]

is a downstream observability seminorm: (\delta z^\top G\delta z) is the local
second-order predictive price of changing the write.  Directions in the numerical
kernel are locally invisible to next-token behavior.  This is the nonlinear,
distribution-relative analogue of an observability Gramian and gives an explicit
quotient of residual space by downstream-null directions.

Apply (G) to MLP0's exact token-token/token-context/context-context tensor terms and
to candidate MLP1/MLP2 factors.  Factorization error should be measured in this metric,
not unweighted Frobenius norm.  A sparse dictionary or HOSVD can then preserve many
Euclidean directions only when the suffix distinguishes them.

### Assumptions that may fail

The Fisher pullback is local and distribution-relative.  A direction invisible near
natural activations can matter under a large deletion or OOD input.  RMSNorm and later
bilinear layers make the suffix state-dependent; one global Gramian may average away
rare circuits.  Fisher curvature measures predictive distributions, not a named
selective-removal objective.

### Prediction beyond reconstruction

At matched parameter count, a (G)-weighted approximation should predict small held-
out CE changes and preserve terminal-circuit effects better than Frobenius/SVD
compression.  Its numerical nullspace should tolerate randomized perturbations with
little CE change; high-eigenvalue directions should have proportionally larger signed
responses.  Circuit-conditioned Gramians should predict which factors can be removed
selectively.

### Cheapest falsifier

On cached early-layer states, estimate only quadratic forms (v^\top Gv\) with JVPs for
the existing SVD/dictionary directions and equal-norm random controls.  Compare the
predicted local CE curvature with actual central finite differences at two amplitudes.
Reject the metric if rank ordering does not hold on held-out documents or collapses
when the sample doubles.  No new factor fit is required for this first test.

## Priority 3 — finite Hankel rank and minimal realizations for state-like circuits

### Exact object

Bracket closure, quote parity, ordered successor, and equality-copy control plausibly
compute small discrete states.  For one registered scalar behavior (f(uv)), form a
finite Hankel matrix

\[
H_{u,v}=f(uv),

\]

whose rows are prefixes (u) and columns are suffix probes (v).  For a rational
weighted language, the rank of the infinite Hankel operator equals the dimension of a
minimal linear representation.  Empirically stable finite rank therefore proposes a
small tensor-state update (h_{t+1}=A_{x_t}h_t), with a linear readout.

This applies to a circuit's extracted response/logit contribution, not the full model.
The tensor-native program stores transition tensors indexed by token classes.  Shared
transition factors across bracket and quote tasks become DAG parents; behavior-specific
readouts become leaves.

### Assumptions that may fail

The selected behavior may not be rational or finite-state.  Absolute position,
unbounded nesting, semantic context, RMSNorm, and attention to arbitrary content can
make rank grow with prefix/suffix length.  A finite matrix can look low rank because
the probe language is weak.  Minimal linear state need not be semantically interpretable
or selectively removable from the transformer.

### Prediction beyond reconstruction

A valid realization predicts unseen prefix-suffix compositions and longer lengths,
gives an explicit executable recurrence, and supplies state-specific interventions.
Its state dimension is an operational complexity measure with a minimality theorem,
not merely a compression ratio.  Shared transitions predict cross-task transfer;
separate readouts predict selective extraction/removal.

### Cheapest falsifier

Use synthetic registered strings already available for bracket, quote, successor, and
equality assays.  Build nested Hankel blocks as prefix/suffix length grows, cross-
validate missing entries, and track numerical rank with bootstrap uncertainty.  Reject
a small-state claim if rank keeps growing, held-out concatenations fail, or the inferred
state cannot replay the circuit's intervention effect.  This is mostly CPU once the
existing circuit outputs are cached.

## Pruned moves

- **Another Frobenius/HOSVD or weight SAE alone:** already contradicted as a selection
  rule by geometry-to-causality sign reversal; retain only as an initializer.
- **One universal hierarchy:** the six components exhibit several different
  arrangements, and `m14` changes class under a threshold shift of only `0.0004`.
- **Factorizing absolute concentration ratios:** mathematically nonadditive and missing
  sign, scale, counts, and document variation.
- **Whole-model polynomial expansion:** exact but combinatorially expensive across 18
  residual/RMSNorm interfaces; start with source-closed local maps and causal interfaces.
- **MDL without a use test:** code length is a useful price axis, but it is not evidence
  of prediction, extraction, selective removal, or OOD transport by itself.
- **Compressed simultaneous perturbation immediately:** still promising, but first
  collect a lawful individual-source pilot to learn whether the causal Jacobian is
  sparse and to provide a dense control.

## Executed action and claim boundary

Implemented:

- `causal_response_tensor_collection.py` — model-free per-document aggregation,
  validation, and create-only serialization;
- `causal_response_tensor_v1_backend.py` — source-closed typed-dispatch collection,
  FIT-only direction estimation, full/residual rank-one projection interventions, and
  an exact physical call ledger;
- two focused test files; **13/13 tests pass in 2.95 seconds**;
- `CAUSAL_RESPONSE_TENSOR_V1_PREREGISTRATION.md` — exact masks, signs, amplitudes,
  phases, lineage, gates, and analysis boundary frozen before outcomes.

This is CPU implementation and a non-degenerate tiny-transformer known-answer test.
It is not a bilin18 response result, factorization result, circuit promotion, strict-
ledger gain, or execution authority.  The next safe step is lifecycle/source-closure
audit; only then should the GPU collector run.
