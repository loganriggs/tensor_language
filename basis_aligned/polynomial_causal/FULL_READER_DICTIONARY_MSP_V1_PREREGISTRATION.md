# Full-rank shared input dictionary, weights only

11 September 2026. Distinct from the earlier64-reader subspace and from
varimax in a truncated output space. Here all1152input directions remain.

Assumption: native Left/Right weight vectors can be expressed sparsely in a
common orthogonal basis. This is a restriction on the existing product
factorization, not the most general jointly refactored folded tensor.

Use Zhai et al., JMLR2020, L4 matching/stretching/projection (MSP), maximizing
mean summed fourth powers of coordinates over the orthogonal group. The update
is the polar factor of (AY)^3 Y^T. Complexity O(d^2 n+d^3) per iteration;
native d1152, training n6144. The paper's sparse random signal assumptions do
not hold by default for learned weights; finite-set convergence is not recovery.
Reference: https://www.jmlr.org/papers/volume21/19-755/19-755.pdf

Data means weight vectors only: no token corpus, activations or task labels.
Normalize all9216native Left/Right rows individually for discovery, retain their
norms in reconstruction. This removes reciprocal native product scaling from
the discovery objective. Freeze paired product split using CPU generator seed700:
first3072products training, last1536products validation. Both readers of a
product stay in the same split. Held-out weights never enter basis fitting.

Fit starts0/937, max20000MSP steps and900soft seconds each, preserving final
bases and histories. Local convergence requires tangent-gradient Frobenius
norm divided by objective <=1e-6 and five-step relative objective change<=1e-9.
Compare complete identity and PCA bases; PCA uses training weight vectors only.
No choice between starts using validation. Fixed sparsity128coordinates/reader.

Predictions:

- A: basis orthogonality and full-coordinate reader replay relative error<=1e-10.
  CPU planted recovery/control source bindings must hold.
- B: both learned bases locally converge by the unchanged criteria.
- C: both held-out normalized-reader top128energy captures exceed the better
  identity/PCA baseline by at least .05 absolute. A training-only gain fails.

Report signed-permutation atom alignment across restarts, training and held-out
energy capture, and exact full-U quadratic coefficient capture of each frozen
top128 reader reconstruction with native Down unchanged. This last metric
checks the full folded object; it is not the fitting objective and no writer
refit is allowed. Retain every original product and native background.

Literal component price: dense basis1152^2 + sparse reader values2*4608*128
+ native Down1152*4608 = 7,815,168 floating coefficients, plus1,179,648indices.
The original component has15,925,248matrix coefficients; bias, nativeU, residual,
normalization and the remaining model are separately retained. No comparison
as a matched-budget winner against the276480-coefficient structured-bank fit.
Execution: shared basis projection once, sparse read combinations,4608products,
native Down. This is a candidate organization of reuse, not circuit identification.

Positive controls: three24D planted starts recover meanatomcos.99856. Normalized
8d-sample sparse controls recover.99573. Dense null creates~12pptraining optimism
without held-out gain; one original2000step null missed convergence and the
unchanged-budget-extension2104step repair converged. Preserve that miss.

If C misses, investigate orthogonality/native-factor and joint-objective
restrictions before any claim of absent structure. If B misses, retain the
bases for continuation; no structural negative. If C holds, inspect stable
shared atom usage and downstream written functions before frozen FineWeb
interventions. No Pile adaptation or data-guided discovery in this experiment.
