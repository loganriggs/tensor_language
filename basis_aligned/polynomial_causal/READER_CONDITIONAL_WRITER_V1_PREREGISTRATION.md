# Frozen reader dictionaries: exact conditional output comparison

Claimed before native dictionary results. No new basis, support or text fit.
Compare identity/PCA top128-coordinate readers, orthogonal MSP seeds0/937,
and ordinary-covariance/Tyler oblique seeds0/937. Parent jobs define all reader
artifacts. Native Down is the paired baseline for each; replace it with the
unrestricted coefficient least-squares optimum conditional on those readers.

Use the existing product Gram algebra and a symmetric eigensolve with relative
cutoff1e-10, no ridge. Report retained rank/condition, normal-equation residual,
coefficient capture, old/new function cosine, elapsed solve time and writer norm.
Numerical truncation is explicit; a failed residual bar invalidates the
conditional-optimum claim. Do not retry with a hidden changed cutoff.

Predictions: A all conditional normal residuals, independent chunked loss
replays and sparse-program replay errors <=1e-8; B no arm loses more than1e-8
absolute capture against its frozen native-Down version; C some learned family
improves both starts by>=.01absolute capture; D some learned family beats the
better refitted identity/PCA baseline by>=.01in both starts. C/D misses mean
the tested relaxation did not provide that gain, not absent reader structure.
Report same-family C-and-D explicitly. Unconverged parent basis fits remain
unconverged; an exact output solve cannot repair that label.

Save each fitted Down before scoring to a new cache, with source artifact hash
and solver diagnostics. No optimizer or original artifact is overwritten.
Before native execution, compare the PSD solver with the prior pseudoinverse
implementation on full-rank and duplicate-product CPU controls (relative
writer and normal-equation errors <=1e-10).

Discovery uses full native weights, including all output writers. The parent
paired weight holdout tests basis learning only: this followup uses all
products to fit Down and is not a held-out full-tensor evaluation. No FineWeb,
Pile, activations or labels. Freeze weights before subsequent behavioral tests.

All arms store7,815,168 matrix coefficients plus1,179,648support indices and1152
bias entries. Dense Down has5,308,416entries in both versions. U, residual,
RMS/tanh and native background remain separately charged. Existing executable
SparseReaderProgram can consume the replacement Down, so this enables later
extraction and selective-feature interventions; it does not establish those
properties. GPU work uses managed lane1 after the oblique parent;0body forwards.
Budget1800seconds for eight conditional solves and scoring, no optimization
iterations. A timeout preserves completed per-arm receipts and is not a null.
