# Sustained full folded fit with eliminated output weights

The direct full-tensor one-step test passed all registered bars on both frozen
repaired dictionaries, with about1second per gradient and >2.3percentage-point
capture gain. Proceed to a sustained joint search rather than another L1 proxy
continuation. Use the same original OVERCOMPLETE_OLS_REENCODE_V1 learned starts,
not the one-step results or live L1 caches, to keep the initialization explicit.

Variables are the shared2304x1152 input dictionary and9216x128 signed sparse
coefficients on their existing supports. Normalize dictionary rows in the
objective; parameterize codes relative to fixed native reader lengths. Scale
both packed parameter blocks to unit Frobenius norm initially. At every
evaluation solve the unconstrained4608-product Down matrix exactly with the
existing spectral conditional solver, then use the existing full-U gradient
and sparse chain rule. This is variable projection, already established in
the07:51 review; the new application is the complete shared sparse graph.

Minimize full relative coefficient squared error, no L1/activation/CE penalty.
All native weight rows are in the target; historical reader splits are not
held-out evidence. No text is read. Native product pairing and support graph
remain fixed; output weights can change. This does not exhaust arbitrary new
products, overlapping support graphs or other structural assumptions.

Use SciPy L-BFGS-B, maxcor10, maxls25, ftol0, gtol1e-12, at most2500 accepted
iterations and3600 soft fit seconds per start. Save accepted states at least
every60seconds and on completion. Time is checked at accepted iterations;
report overruns. Whole managed job alarm9000seconds. The available Python
interface does not restore private L-BFGS history; saved parameters remain
usable but are not represented as a full solver-history checkpoint.

Use full parameter stationarity checks, not solver success: maximum of the
unit-feature tangent-gradient norm times sqrt(feature count), and code-gradient
norm times code norm, divided by max(relative loss,1e-12), must be<=1e-5 at two
successive accepted steps. Their relative objective change must be<=1e-8.
Report product Gram rank/conditioning at every step and exact writer normal
residual. The envelope derivative requires locally constant retained rank;
rank changes are reported and disqualify the registered smooth-fit instrument
verdict. A pseudoinverse cutoff is not a theorem about exact structural rank.
Also report gauge-invariant summed component energy, which can expose growing
cancellation despite falling residual.

Predictions:

- A: initial packed directional finite difference<=1e-6 relative to max(1,
  absolute derivative); writer normal<=1e-8; initial/final independent CP error
  replays and sparse execution<=1e-8; finite diagnostics, unit feature norm
  error<=1e-6; all accepted candidate Gram ranks remain4608; accepted objective
  increases<=1e-10.
- B: both starts meet the full joint convergence rule above.
- C: both gain>=.05 absolute coefficient capture beyond their own initial
  **already output-refitted** functions. A free writer alone cannot earn this.
- D: both final full coefficient captures>=.70.
- E: the two final complete functions have full-U cosine>=.9.

Save and report initial original-writer capture, initial optimal-writer capture,
final capture, every original/partial miss, solver reason, timings, parameter
scales, norms and literal price. A/C/D without B/E is useful approximation but
not stable identification; timeout is not absent structure. Rank changes or
large component-energy growth require numerical/degeneracy analysis before
interpreting convergence. No global recovery guarantee.

Each executable program stores9,142,272 matrix coefficients,1,179,648indices
and1152bias values, with runtime COO indices, U and native background charged.
Down is learned and stored explicitly, not silently treated as retained native
background. This is still only the last bilinear/unembedding contribution;
normalization and downstream behavioral properties remain separate validation.

Managed lane1, after the completed direct-gradient probe. Zero body forwards.
Sources: [O'Leary and Rust](https://www.cs.umd.edu/users/oleary/software/varpro.pdf),
[existing mathematical mapping](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_0751.md),
[new integration control](PROJECTED_SPARSE_DICTIONARY_V1_CONTROL.json).
