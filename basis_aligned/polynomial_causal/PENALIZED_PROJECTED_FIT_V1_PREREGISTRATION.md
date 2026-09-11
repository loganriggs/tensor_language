# Joint component-penalized folded sparse fit

The output-only budget test reduces cancellation cheaply, but output-only changes
cannot resolve the observed cross-start function disagreement at <=.001 capture
loss each. Test joint reader movement with a component penalty. This changes the
objective and code parameterization, not the underlying structural family.

Start independently from the two final original PROJECTED_SPARSE_DICTIONARY_FIT_V1
parents, including their original fixed supports. Do not consume the queued support
exchange outputs. The parent numerical instrument must pass; parent convergence is
reported but not required. Both native weights and all U rows enter the objective.
No text is read and no historical weight split is held out.

Minimize relative full coefficient squared error plus lambda times summed individual
product-tensor energies, with **lambda=.0018823354889938328**. This is calibrated on
seed0's completed output-budget audit, not an untouched hyperparameter choice.
Normalize dictionary rows and code rows inside the objective; output weights absorb
reader scales. Keep2304features,9216readers,128connections per reader and4608pairs.
Eliminate outputs using the exact normalized-Gram Cholesky solve with positive
component penalty. This is explicit regularization, not an unreported numerical ridge.

Reuse projected_sparse_dictionary_fit_v1.fit via the thin PenalizedObjective adapter.
SciPy L-BFGS-B: maxcor10,maxls25,ftol0,gtol1e-12; two1200softfitseconds starts,
max2500acceptediterations, wholejob alarm3600seconds. Time is checked after accepted
steps and overruns are reported. Save actual accepted features, normalized code rows,
output weights and history every60seconds and at completion. Private L-BFGS history
is not restored or represented as available. Full history lives once per fit receipt,
not duplicated into every summary.

Joint convergence requires feature and code unit-row tangent-gradient norms times
sqrt(row count), divided by max(abs(penalized objective),1e-12), <=1e-5 at two
successive accepted steps, with relative objective change<=1e-8. These gauge-aware
criteria do not retroactively pass the old fit. Reader-span degeneracy and global
optimality remain unproven even though the conditional output solve is positive definite.

Predictions, scored independently:

- A: packed initial directional finite difference at h1e-6 <=1e-6; independent
  initial/final CP-plus-penalty objective and sparse executor replay<=1e-8;
  output normal residual<=1e-8; finite values; feature/code unit norms<=1e-6;
  accepted objective increases<=1e-10. Seed0 initial penalized objective must
  additionally replay its prior exact component-budget solution<=1e-8.
- B: both starts meet the joint convergence rule.
- C: both improve penalized objective by>=1e-4 beyond their own initial
  already penalty-refitted outputs.
- D: both retain capture>=own unpenalized parent capture-.001 and end with
  summed component energy<=1native tensor energy.
- E: final complete-function cosine>=.9 across starts.

The null is that regularizing cancellation/normalizing coordinates does not yield
convergence or stable shared computation under this fixed graph. A plateau or a
time limit is not absent structure. Every miss remains recorded; no automatic
identical extension is authorized by this protocol. Compare capture, cancellation
and stationarity separately; neither small component energy nor similar captures
identifies individual circuit units.

Literal price is unchanged:9,142,272 matrix floats,1,179,648stored indices and1152bias
values per program; runtime sparse indices, fullU and native background remain
charged. Down is learned explicitly. Validation of FineWeb/OOD behavior, extraction,
selective removal and composition is not part of this discovery run.

Managed lane1, appended after the already queued FULL_SUPPORT_EXCHANGE_V1.
Before enqueue require dense kernel, native smaller-step derivative and planted
controller controls to pass, bind reviewed dependencies, then hash-bound dry-run.
