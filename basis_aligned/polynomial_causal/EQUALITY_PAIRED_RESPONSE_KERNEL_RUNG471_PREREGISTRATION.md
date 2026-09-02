# Rung 471 preregistration: paired downstream-response kernel

Registered after rung470 and before opening any target-specific gradient or response-kernel outcome.

## Why this object

Rung469 found that covariance between downstream use and the equality-induced MLP state change carries most of the
local response. Rung470 found that repeat distance, predecessor count, and query position reproduce aggregate code
means but not individual effects or natural-text scale. This rung keeps the downstream reader and state change paired
at each sequence position.

## Exact kernel

For one selected loss position `q`, MLP layer `l`, and MLP write position `p`, define

`k[l,q,p] = - gradient(CE[q], MLP_write[l,p]) dot (MLP_write_source[l,p]-MLP_write_absent[l,p])`.

The second factor is computed exactly from all4,608 products. Summing `k` over positions is the complete local
first-order effect of removing that MLP contribution for target `q`. No native product index remains in the reported
kernel.

Split positions into four fixed regions relative to the target and its latest equal-token predecessor `r`:

1. `query`: `p=q`;
2. `latest_predecessor`: `p=r`;
3. `between`: `r<p<q`;
4. `earlier`: `p<r`.

Positions after `q` are a causal-leakage control and must sum to numerical zero.

## Fixed scope

- Use the first two `all_positive` query positions in every document, ordered by query index; documents with one use
  one and documents with none use none. This selection depends only on frozen masks.
- Windows, sources, and rows are exactly those in rungs469/470: code0:96 fit, code96:192 validation, and two natural
  waves0:96 and96:192.
- Exact causal targets come only from the already-frozen rung470 per-token bundle, matched by document/query.
- Fit on code discovery one least-squares scalar per source for each MLP and their sum, mapping total local response to
  exact removal CE. Freeze it before every other window.
- The primary control is rung470's already-frozen distance/count/position prediction evaluated on the same targets.
- Zero deployed saving/addition; no rank, product selection, new role, or SEALED attention0 result.

## Registered predictions

### A. Instrument

All hashes/coordinates match; native replay is at most `1e-12` relative-squared; MLP factor reconstruction is at most
`1e-10`; future-position leakage is at most `1e-9` times the total kernel norm plus `1e-12`; the four regions sum to
the full local response to `1e-9`; all expected forwards/backwards fire; SEALED remains closed.

### B. Held-out code causal prediction

For the sum of MLP8/9/12 under both sources, the frozen calibrated kernel predicts exact per-token removal CE on code
validation with Pearson at least `.55` and at least `15%` lower RMSE than rung470's context-only prediction on the
same targets. Its four aggregate context means have cosine at least `.90` and projection `.50--1.50`.

### C. Natural causal prediction

Under both sources in both natural waves, the code-frozen calibrated kernel has Pearson at least `.30`, at least `15%`
lower RMSE than the context-only predictor, and four-cell cosine at least `.80` with projection `.25--1.75`.

### D. Stable spatial computation

For at least two of MLP8/9/12, the four-region signed contribution vector and four-region absolute-contribution vector
each have code-discovery→validation cosine at least `.80` and code-discovery→each-natural-wave cosine at least `.70`
under both sources. The same region has the largest absolute contribution in every window/source for that MLP.

### E. Shared downstream use across matcher sources and MLPs

The complete per-target four-region kernel under N and H has cosine at least `.85` in every validation window. At least
two individual MLP calibrated kernels must beat their rung470 context controls in every window under both sources.

## Strong null and next branch

The strong null is true if A fails, if B fails under either source, or if no natural source/wave improves over the
context control. A full pass licenses exact interventions on the frozen position regions. If the total paired kernel
predicts causal effects but its regions do not transfer, retain it as a downstream-use variable but do not name a
spatial circuit. If it fails even on held-out code, first-order downstream response is insufficient; test an exact
target-specific region intervention or a nonlinear state/use variable, not rank or product-count variants.
