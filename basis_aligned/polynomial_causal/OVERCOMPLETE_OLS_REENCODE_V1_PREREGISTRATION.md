# Frozen overcomplete dictionaries with OLS support completion

Use the completed OVERCOMPLETE_L1_READER_V1 artifacts: initial and learned
dictionaries for seeds0/937. No dictionary learning, penalty change, output
refitting, corpus data or activation weighting. This is a different inference
rule applied equally to both sides of the existing comparison. Original parent
convergence and quality misses remain unchanged; historical weight holdout is
not fresh validation of this encoder choice.

Solve the same normalized-reader Lasso(.05) with the existing proximal core,
retain its nonzero support up to128 terms, fill remaining slots using the exact
one-step OLS improvement, and refit selected values by least squares. Use the
controlled batched Schur implementation, grouped by initial support size with
batch64. The rule is greedy, not the global sparse optimum. Rank failures are
explicit; no ridge or arbitrary zero-padding is silently added.

Native Down and bias are retained. Score all9216readers and the full-U folded
coefficient tensor through existing CP contractions. The native pairing and
normalized-reader objective remain restrictions. Save each new sparse program
before scoring and preserve parent artifact hashes. Numerical controls compare
batched and scalar inference on orthogonal and rectangular toy dictionaries;
the same comparison runs on the managed GPU before full recoding.

Predictions, registered before full recoding:

- A: GPU scalar/batched control passes1e-9 relative execution and normal-error
  bars with matching supports; all four native encoders converge to relative
  stationarity<=1e-7 and maximum code KKT<=1e-5; support normal residuals,
  parent full-capture replays and sparse/dense program execution<=1e-8; finite
  scores and dictionary norms<=1+1e-6.
- B: no arm loses more than1e-8absolute full coefficient capture versus its
  original encoded version. Greedy completion does not guarantee this globally.
- C: both learned arms gain>=.02absolute full coefficient capture versus their
  respective original encodings.
- D: both learned arms' historical held-out reader captures exceed the better
  newly encoded initial dictionary by>=.02absolute.
- E: both learned arms' full coefficient captures exceed the better newly
  encoded initial dictionary by>=.01absolute.
- F: complete learned functions across starts have full coefficient cosine>=.9.

Separate reconstruction, parent convergence and function stability verdicts.
Even all encoder-quality bars would not certify the unfinished parent fitting,
global feature recovery or the four circuit properties. Report training-reader
scores, historical test scores, code-support counts, time and every arm.

Each program keeps9,142,272matrix coefficients plus1,179,648stored support
indices and1152bias values. Runtime COO indices, U and native background remain
separately charged. No whole-model saving or runtime speedup is claimed from
this recoding. Zero body forwards/sequence evaluations, no text access.
Managed lane1 only, after the already queued FineWeb diagnostic. Whole-job
alarm1800seconds; save completed per-arm receipts before any later failure.
