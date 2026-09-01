# MLP0 rank-448 downstream CE-Fisher screen — preregistration

Date: 2026-09-01 16:20 UTC

## Question and scope

Rung 405 used exact Euclidean token/interaction branch derivatives. It improved those branch reconstruction errors
but not next-token cross-entropy. This rung changes the sensitivity object to the actual downstream CE gradient.
It asks whether a task-directed Fisher basis improves the same legal rank-448 MLP0 program.

This is a single-site equal-price screen. It cannot license compression/adoption, rank selection, FINAL evaluation,
separate branch storage, or a router.

## Frozen source and populations

- Pinned float32 bilin18 source checkpoint and `.rowcache/fineweb_n192_skip11000.pt[0:24,:257]`, exactly the source,
  dtype, and 24 program-fitting documents used by rungs 328/403--405.
- Fit-half stability uses documents `[0:12]` and `[12:24]`; the deployed candidate uses all 24.
- Physical evaluation is rung404's frozen 384-source-document population: one chunk per source document, four
  contiguous96-document waves, positions `[64:256)`, 73,728 positions total, under the rung401 BF16 model.
- The mechanism check is rung403's unchanged 96 SELECT documents, 32 T/C/I/S/A arms, FIT reference, and BF16
  arithmetic. FINAL remains unopened.

## Metric and legal program

For each fitting document, let `x_i in R^1152` be the input at position `i` to MLP0. Compute the summed real
next-token cross-entropy of the complete frozen model, and backpropagate only to the MLP0 input leaf:

`g_i = d CE_sum / d x_i`.

The directional downstream Fisher is

`F = (1/N) sum_i g_i g_i^T`.

This differs from rung353, which retained only the scalar norm `||g_i||` as an input-sample weight. It also differs
from rung405, whose output probe was isotropic and whose sensitivity target was Euclidean T/I branch error. Here the
direction and sign geometry of the actual full-suffix CE gradient are retained.

Let `C` and its eigenvalue floor be the unchanged rung328 normalized-input covariance. Define

`K_F = C^(1/2) F C^(1/2)`.

The Fisher basis `U_F` contains the top448 eigenvectors of `K_F`. The shared-input program is

`encoder=U_F^T C^(-1/2)`, `decoder=C^(1/2)U_F`,

`L_small=L decoder`, `R_small=R decoder`.

It therefore retains the exact rank448 shapes, native Down/bias, and `9,954,432` stored values, saving `5,971,968`
from native MLP0.

Controls:

- `covariance`: unchanged rung328 p448;
- `Fisher_shuffled`: seed406 permutes the 1,152 whitening-frame coordinates of `K_F`, preserving its complete
  eigenvalue spectrum while destroying eigenvector alignment, then builds the same p448 graph and price.

No interpolation, weight, clipping, rank, or seed is selected after evaluation.

## Measurements

1. Gradient norms/quantiles, finite/symmetry checks, full/half metric hashes, top448 Fisher energy, and normalized
   overlap of the two half-fit p448 subspaces.
2. Physical native/program CE damage for all three programs on each large-evaluation wave.
3. Exact T/I branch relative MSE for orientation only.
4. The complete rung403 SELECT branch-error factorial for the fixed Fisher program, including exact analytical
   identities, endpoints, live calls, Shapley values, and auxiliary closure.

## Frozen predictions

### A — exact instrument and equal price

- checkpoint, fit/evaluation rows, hashes, disjointness, wave sizes, rank, shapes, and literal price are exact;
- every fit document produces finite nonzero MLP0-input gradients; `F` symmetry relative error is at most `1e-6`,
  every basis orthogonality error is at most `1e-5`, and the shuffled whitened metric has the same eigenvalues as
  the real metric within relative `1e-5`;
- covariance p448 reproduces every rung404 wave damage within `1e-6`;
- all large-wave states/endpoints/calls and the Fisher SELECT factorial identities/endpoints/calls are exact; FINAL
  is unopened.

### B — directional Fisher structure is stable and causal rather than spectral luck

- normalized overlap of the two independently estimated half-fit top448 Fisher subspaces is at least `0.50`;
- pooled Fisher CE damage is at least `0.001 nat` lower than Fisher-shuffled damage.

### C — Fisher improves held-out prediction robustly

- pooled Fisher CE damage is at most `0.85` times covariance damage;
- Fisher improves on covariance by at least `0.0002 nat` in at least three of four waves;
- no wave regresses by more than `0.001 nat`.

### D — the predictive gain repairs the registered token-grammar error

Let `G=T+I` denote the sum of SELECT Shapley damage from those two factors. Relative to the unchanged rung403 SELECT
covariance baseline:

- Fisher `G` is at most `0.75` times baseline `G`;
- Fisher total compact CE damage is no larger than baseline total damage;
- the absolute change in combined `C+S+A` Shapley damage is no larger than the positive reduction in `G`.

## Strong null

The strong null fires if A fails; pooled Fisher improvement over covariance is below `0.0002 nat`; Fisher-shuffled
comes within `0.0002 nat` of Fisher; Fisher loses to covariance in at least three waves; or half-fit overlap is at
most `0.25`.

## Decision

- A+B+C+D with no null licenses one established whole-model census/certificate/intervention gate for this fixed
  Fisher p448, not adoption by this receipt.
- Stable Fisher geometry without CE improvement identifies another objective mismatch; do not tune the metric.
- Strong null closes fixed quadratic/first-order p448 metrics. Next compare direct nonlinear CE fitting against the
  heldout headroom of a small observable-state router before constructing either.
