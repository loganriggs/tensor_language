# Block-3 typed reduced-rank regression v1: pre-outcome control specification

**Status:** superseded as the primary experiment before any block-3 covariance or
evaluation outcome was collected.  It is retained as an equal-cost linear-factor
control for the native-gate-subset experiment.  The mathematical red-team correctly
observed that this grammar reduces factor-matrix storage but still computes all 4,608
products and retains all `Down` columns; it is therefore weaker executable simplicity
than removing shared native gates.

Frozen before collecting block-3 covariance or evaluating any fitted program.

## Purpose

The exact attention--MLP polarization and the failed raw coefficient screen leave one
specific question: is the part of MLP3's dense coefficient map that natural trajectories
use and downstream computation cares about smaller than the full map?

This experiment distinguishes three possibilities:

1. **Empirically low-dimensional typed interface:** the residual-derived and
   attention-derived factor activations are jointly predictable through a small shared
   encoder, with small local and downstream error.
2. **Downstream tolerance or cancellation only:** local typed error remains large but
   final CE/logit consequences are small.
3. **No useful grouped compression at this cut:** both local typed and downstream errors
   remain large at useful ranks.

The experiment is a block-3 pilot.  It cannot earn whole-model storage or causal-ledger
credit until a passing construction composes with adjacent blocks.

## Exact typed variables

Let (h) be the residual after block 3's learned residual mixing and before attention,
(a) its native attention write, and

\[
\gamma=\left(\frac{1}{d}\lVert h+a\rVert_2^2+\epsilon\right)^{-1/2}.
\]

Define

\[
u=\gamma h,
\qquad
v=\gamma a,
\qquad
z=u+v=\operatorname{RMSNorm}(h+a).
\]

For the gauge-balanced product factors \(\widetilde L,\widetilde R\), the four exact
typed writes are

\[
T_{uu}=D[(\widetilde Lu)\odot(\widetilde Ru)],
\]

\[
T_{uv}=D[(\widetilde Lu)\odot(\widetilde Rv)],
\]

\[
T_{vu}=D[(\widetilde Lv)\odot(\widetilde Ru)],
\]

\[
T_{vv}=D[(\widetilde Lv)\odot(\widetilde Rv)].
\]

Their sum plus the native bias is exactly MLP3's write.

## Programs compared at identical rank and execution form

Every candidate replaces the stacked factor map

\[
\begin{bmatrix}\widetilde L\\\widetilde R\end{bmatrix}
\]

by a shared rank-(r) encoder and two decoders.  The real stored-factor and multiply
count is

\[
r(d+2m)
\]

instead of (2md).  `Down` and the bias remain native.  Ranks are frozen to

\[
r\in\{64,128,256,512\}.
\]

The two primary arms are:

- **z-only RRR:** reduced-rank regression minimizes factor error on the native
  (z=u+v) distribution;
- **typed RRR:** the same regression minimizes the combined factor error on (u) and
  (v) separately.

They have exactly the same parameterization and rank.  Therefore any typed-edit
advantage cannot be attributed to a larger model.  The already-measured identity-
covariance coefficient factorization is retained as a descriptive raw-weight control,
not a new discovery arm.

Each gate row is first placed in the exact positive minimum-norm product gauge.  Rows
are weighted by their `Down` column norm during regression.  This weighting is a fixed
linear proxy for downstream importance; final nonlinear consequences remain the judge.

## Frozen data roles

- fit: all 480 cached FineWeb rows from `fineweb_n480_skip80.pt`;
- validation: 192 cached rows from `fineweb_n192_skip7000.pt`;
- fixed replication: 192 cached rows from `fineweb_n192_skip11000.pt`.

Only positions 64 through 255, corresponding to targets 65 through 256, enter fit or
score.  Fit uses 92,160 token positions.  Validation and replication each use 36,864.
Rows are reused project infrastructure, so the evidence is held-out for this frozen
fit but not a globally pristine dataset.  A successful pilot must later transport to
the frozen code corpus or newly frozen collision-free FineWeb rows.

## Evaluation hierarchy

### Stage A: local factor and typed-output screen

For every arm and rank, report on validation and replication:

- downstream-weighted factor NRE;
- NRE of each (T_{uu},T_{uv},T_{vu},T_{vv});
- NRE of their natural all-term sum;
- real parameter and multiply counts.

NRE is the Euclidean norm of candidate-minus-native values divided by the native
centered norm.  No arm can be promoted from these local metrics alone.

### Stage B: natural downstream consequence

Continue the model from the block-3 residual using the candidate natural all-term
write.  Report candidate-versus-native:

- CE difference;
- mean next-token KL divergence;
- centered-logit relative error;
- top-1 agreement;
- document-bootstrap intervals.

The validation role selects the smallest typed rank satisfying all of:

\[
|\Delta\mathrm{CE}|\le 0.01,
\qquad
\mathrm{KL}\le 0.01,
\qquad
\mathrm{centered\ logit\ NRE}\le 0.10,
\qquad
\mathrm{top1\ agreement}\ge 0.95.
\]

The selected rank is then frozen.  No rank may be changed after replication opens.

### Stage C: typed edit transport

For only the selected typed program and its equal-rank z-only control, compare exact
and compressed continuations under these five masks:

| mask | retained terms | interpretation |
|---|---|---|
| all | uu, uv, vu, vv | natural MLP3 write |
| no-vv | uu, uv, vu | remove pure attention-quadratic write |
| no-cross | uu, vv | remove both residual--attention cross writes |
| cross-only | uv, vu | retain only coupling writes |
| uu-only | uu | residual-derived quadratic write only |

For each nonnatural mask, the truth arm and compressed arm receive the same mask.  The
key response is the change relative to their own all-term baseline.  Report CE-effect
error, centered-logit response NRE/cosine, KL, top-1 agreement, and document-bootstrap
intervals.  This evaluates functional editing, not similarity of hidden coordinates.

## Registered predictions and decisions

1. At rank 256, typed RRR reduces validation typed-factor NRE by at least 20% relative
   to the raw identity-covariance rank-256 control.
2. At equal selected rank, typed RRR is within 0.01 CE of z-only RRR on the natural
   arm and improves mean edit-response NRE by at least 0.05.
3. A typed rank at most 512 passes all natural thresholds on both validation and fixed
   replication, and has mean edit-response NRE at most 0.20 with every individual mask
   at most 0.30.

If prediction 1 fails, activation weighting did not solve the dense coefficient
interface.  If natural thresholds pass but edit transport fails, the compression is
functionally faithful only on-distribution and is not a composable circuit.  If no
rank at most 512 passes natural replication, grouped factor compression at block 3 is
pruned in this form.  If all three pass, extend the same frozen construction to block
4 before attempting a two-block composition.

## Integrity and publication

Collection must be source-closed and receipt-last.  It records the exact checkpoint,
row hashes, block, positions, RMS epsilon, covariance counts, call census, and exact
replay error (u+v-z).  Fit sees only the fit covariance.  Validation selects once;
replication is loaded only after selection is sealed.  Final artifacts retain
document-level sufficient statistics, never full logits.
