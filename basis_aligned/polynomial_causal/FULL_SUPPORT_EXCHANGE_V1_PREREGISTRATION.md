# Full sparse-connection sweep after the continuous folded fit

The frozen iteration729 probe finds improving swaps for all32sampled readers,
with median normalized gain2.52e-7 versus6.25e-12 from refitting current
connections. Sequential application retains100.28% of the independent sum;
actual gain1.025e-5 is small. These are not additive predictions for the whole
layer. SequentialV1 had an execution-only duplicate-key failure; V2 retains
the same experiment and thresholds and passed the exact global CP check.

After PROJECTED_SPARSE_DICTIONARY_FIT_V1 completes, use its two final accepted
artifacts, verifying each hash and initial coefficient replay. Do not require
parent convergence or change its verdict. Abort on failed parent numerical
instrument. Freeze each shared dictionary and its fitted Down. Compare two
independent arms from each parent: (1) sequential same-support exact reader
refitting, (2) sequential best single drop/add per reader after that refit.
Process all4608Left then all4608Right readers, updating the full polynomial
residual after every change. Every reader retains128connections to2304features.
No new token data, parameter count, output refit, feature rotation or pairing.

For reader a with partner b and whitened writer w, its squared tensor error is

$$
 a^\top H a-2a^\top r+\mathrm{constant},\qquad
 H=\frac{\lVert w\rVert^2}{2}
 (\lVert b\rVert^2I+bb^\top).
$$

The rhs includes all other native and candidate products. In shared basis B,
use BHB^T and Br. Reuse the controlled full residual and Schur-complement
exchange implementation. This is a conditional combinatorial improvement,
not a converged graph optimizer; revisiting earlier readers can change their
best connections. Signed cancellation and cross-reader effects are included.

Measured predictions (both seeds must hold where relevant):

- A: initial/final independent full CP replay and sparse executor relative
  replay <=1e-8; finite states,128distinct connections per row; minimum
  normalized conditional gain>=-1e-10; full measured improvement agrees with
  summed sequential conditional gains within1e-8.
- B: exchange full coefficient capture improves>=.001 (0.1percentage point)
  over its own frozen parent.
- C: exchange improvement exceeds the complete same-support-refit control
  by>=.0005 (0.05percentage point).
- D: each complete9216reader sweep takes<=900seconds. Whole alarm5400seconds.

All misses remain misses. Report parent convergence, whole-function stability
before/after between seeds, and per-side gains/times; no predicted stability
improvement is registered. One sweep is a feasibility/gain screen, not absent
structure if it fails. Numerical conditioning or a no-change tie is recorded.

The program retains9,142,272matrix coefficients,1,179,648indices,1152bias
values, U and the native background. Parameter capacity is identical across
arms and to each parent; only sparse connections and values change. Exact
quadratic execution does not establish behavior, selective removal, OOD or
reusable semantic units. Successful connections become candidates for later
feature grouping and frozen intervention validation.

Managed lane1 only, queued after the live projected fit; zero model-body
forwards. Save four independent sparse artifacts and primary results. Never
overwrite live parent files or resume their private optimizer histories.
