# Three-hour mathematical review: choose measurements by tensor cross

**Review time:** 2026-08-28 22:30 UTC

**Code:** `cut_cross_interpolation_diagnostic.py`

**Result:** `cut_cross_interpolation_diagnostic_results.json`
**Scope:** discovery and experiment design only. The completed 64-mask grid has
already been revealed, so none of the numbers below is new held-out evidence.

## Executive conclusion

The latest cut experiment did not show that the intervention function has tensor
rank one or two. It showed two more specific things:

1. an arbitrary partially observed grid was not enough to predict the untouched
   interactions at rank at most two; and
2. after seeing the full grid, its singular values nevertheless decay.

Those facts are compatible. A matrix can be approximately low-rank while an
ill-chosen or insufficient set of entries is poor for interpolation. The next
mathematical move should therefore change **which experiments we measure**, not
just rerun another factorization on the same entries.

The highest-priority move is **maximum-volume tensor-cross interpolation** of the
whole-program consequence function. It adaptively selects informative program
masks and, if the function really has low tensor-train rank, reconstructs it from
only a number of calls linear in network depth rather than enumerating all masks.
Unlike local MSE or weight rank, success directly means predicting executions of
new compiled programs.

A CPU diagnostic completed during this review. On the present 7 by 7 anchored
interaction matrices, rank four maximum-volume crosses retrospectively attain:

| target | cross normalized error | cross RMSE | best rank-4 normalized error |
|---|---:|---:|---:|
| top-1 cost | 0.0823 | 0.1142 percentage points | 0.0460 |
| CE cost | 0.1606 | 0.0586 nats | 0.1162 |

This is promising enough to justify a **new prospective cut**, but not to claim a
rank-four law: the pivots and ranks were examined after all cells were visible.

The tempting alternative, “rank two plus a few sparse exceptions,” was tested and
is not supported. After the best rank-two approximation, the residual has an
effective support of 16.2 cells for top-1 and 21.4 cells for CE; the four largest
cells contain only 38.9% and 29.3% of residual energy. The error is too diffuse for
generic robust-PCA-style sparse outliers to be the explanation.

## Current project state used in this review

The whole-model ledgers have not moved:

- all 36 attention/MLP sites have structural replacements;
- the certified complete-program storage reduction is 5.3481%;
- the older human-semantic behavior ledger explains about 32.1% plus or minus
  6.4%;
- strict named causal CE recovery is 10.923%, leaving 4.72714 nats, or 89.077%,
  unexplained by that standard;
- the final 68-action causal tensor has 0 of 68 final scored rows, because its
  semantic replay reducer and comparator/uncertainty contract are not yet closed.

Recent results narrow the target:

- The full-rank length-one program exactly reaches the model's own per-token,
  length-one ceiling on covered positions. The old roughly 0.55-nat “headroom” was
  a rank-64 comparison error, not a full-rank defect.
- Roughly 2.74 nats therefore remain between that ceiling and the live model. That
  is mainly the cost of deleting contextual attention, not a missing table row.
- Independent table choices and a scalar depth rule fail when composed. MLP2 also
  changes sign depending on whether MLP0 and MLP1 have been restored. Whole-program
  interaction is therefore the central unsolved object.
- The layer-5 8 by 8 mask grid has low spectral tails but failed the registered
  rank-at-most-two interaction prediction. Total cost can look predictable because
  large additive row and column effects dominate; that does not explain the joint
  interaction.

At review time the GPU is occupied by the independently owned
`rank_to_ceiling.py` job. This review used CPU only and did not modify its queue,
logs, or source.

## The object we should compile

Let each site action be a symbol such as native, length-one table, fitted table, or
a compressed native module. For an ordered action string

\[
    w = a_1 a_2 \cdots a_{36},
\]

let

\[
    F(w) =
    \bigl(\Delta \mathrm{CE},\;\Delta \mathrm{top1},\;
          \text{copy response},\;\text{frequency response},\;
          \text{consumer responses}\bigr).
\]

Every evaluation of \(F\) is a real forward pass of a physical compiled program.
This avoids the earlier token-prefix Hankel problem: that audit created unnatural
token splices and was severely out of distribution. Here the prefixes and suffixes
are **program interventions**, not text fragments.

The scalar 8 by 8 grid was one tiny matrix slice of \(F\). The final 68-action bank
is intended to provide several output coordinates of \(F\), but its final reducer
is presently blocked.

## Priority 1: maximum-volume tensor-cross interpolation

### Exact object

For a physical cut after site \(k\), form the matrix

\[
    H_k(p,s) = F(ps),
\]

where \(p\) is a legal prefix program through site \(k\), and \(s\) is a legal
suffix program after it. Across all 36 sites these matrices are unfoldings of one
high-order consequence tensor.

### Mathematical definition and guarantee

For an ordinary matrix \(A\), choose row indices \(I\) and column indices \(J\),
with square intersection \(U=A[I,J]\). Cross interpolation uses

\[
    \widehat A = A[:,J] U^{-1} A[I,:].
\]

If \(A\) has exact rank \(r\) and \(U\) is a nonsingular \(r\) by \(r\) cross,
this recovers \(A\) exactly. A maximum-volume cross maximizes
\(|\det U|\), which controls interpolation conditioning and gives an entrywise
error bound in terms of the next singular value. Tensor-train cross applies these
crosses successively to tensor unfoldings. For mode size \(q\), depth \(d\), and
TT rank \(r\), its queried-entry and arithmetic scales are roughly
\(O(d q r^2)\), not \(q^d\). See
[Oseledets and Tyrtyshnikov, 2010](https://doi.org/10.1016/j.laa.2009.07.024).

This is an experimental-design theorem, not merely a compression score: it says
which new program masks are maximally informative if low cut rank is real.

### Assumptions that may fail

- The needed TT ranks may be large or may grow sharply near RMSNorm/residual
  interfaces.
- CE is nonlinear in logits, so a low-rank internal program need not give a
  low-rank scalar CE tensor. Vector logit or consumer responses may be the more
  closed output.
- Near-singular pivots amplify document noise. Pivot condition number and bootstrap
  stability must be gates, not diagnostics hidden after selection.
- The action alphabet is site-dependent. Standard homogeneous TT notation must not
  be mistaken for identical transitions at every layer.

### Consequence beyond reconstruction

A passing cross predicts the CE and causal-response vector of **unqueried compiled
programs**. Its TT cores are executable, compose in site order, have explicit
multiply/storage cost, and expose which cut carries interaction state. That directly
supports prediction and extraction. Editing a core can support selective removal,
but only after collateral-response tests.

### Cheapest prospective falsifier

Choose a fresh physical cut and fresh legal prefix/suffix masks. Before measuring
their outcomes:

1. use a small discovery set only to choose candidate maximum-volume pivots;
2. freeze ranks 1 through 4, pivot conditioning thresholds, and document bootstrap;
3. reserve complete rows, complete columns, and ordinary cells that play no role in
   pivot selection;
4. require the frozen cross to beat additive and fixed-rank ALS baselines on CE and
   on a response vector, not only top-1;
5. reject the rank if its upper confidence bound misses the registered error target.

### CPU result executed now

The completed layer-5 grid was reanalyzed exhaustively. For each rank 1 through 4,
the code enumerated every square submatrix, selected the largest absolute
determinant, and formed the skeleton interpolation. Four tests pass, including exact
recovery of a synthetic rank-two matrix.

Detailed values:

| target | rank | best-SVD NRE | max-volume-cross NRE | cross RMSE |
|---|---:|---:|---:|---:|
| top-1 cost | 1 | 0.2987 | 0.4125 | 0.5724 pp |
| top-1 cost | 2 | 0.1808 | 0.2850 | 0.3954 pp |
| top-1 cost | 3 | 0.0948 | 0.1378 | 0.1912 pp |
| top-1 cost | 4 | 0.0460 | 0.0823 | 0.1142 pp |
| CE cost | 1 | 0.6019 | 0.6894 | 0.2515 nats |
| CE cost | 2 | 0.3413 | 0.4736 | 0.1728 nats |
| CE cost | 3 | 0.1884 | 0.3079 | 0.1124 nats |
| CE cost | 4 | 0.1162 | 0.1606 | 0.0586 nats |

At rank two, both targets independently choose the same maximum-volume pivot rows
6 and 7 and columns 4 and 7. This is weak evidence that the two targets share an
interaction skeleton. It is not a prospective result, and ranks 3 and 4 choose
different pivots.

The result also clarifies the earlier failure. Rank two is simply too restrictive
for CE here, even with an ideal pivot. Rank four is a much better retrospective
description, but it must earn prediction credit on a new grid.

## Priority 2: vector-valued shared-cross factorization

### Exact object

Do not separately factor a CE table, a top-1 table, 18 consumer tables, and the
copy/frequency tables. Treat the consequence coordinate as an additional tensor
mode:

\[
    \mathcal R(a_1,\ldots,a_{36},m),
\]

where \(m\) indexes the measured consequence. Seek shared prefix and suffix bases
with a small metric-specific core:

\[
    H^{(m)}_k \approx U_k G^{(m)}_k V_k^\top.
\]

This is simultaneous factorization/shared-dictionary learning with physical program
prefixes and suffixes as dictionary elements. It is not an SAE on weight columns.

### Operational definition

The representation is simpler only if one shared \((U_k,V_k)\) predicts held-out
coordinates and held-out masks at lower total storage and multiply count than the
sum of independently fitted bases. Gauge changes
\(U_k\mapsto U_k Q, V_k\mapsto V_k Q^{-\top}\) are quotiented out when pricing the
state dimension; an arbitrary rotation is not a new explanation.

The connection to multilinear sequence models is exact in the linear case:
vector-valued weighted automata/linear second-order RNNs can be recovered from
low-rank Hankel tensors; see
[Rabusseau, Li, and Precup, 2019](https://proceedings.mlr.press/v89/rabusseau19a.html).

### Assumptions that may fail

- Different consequences may require incompatible subspaces; the common rank can
  become as large as the sum of individual ranks.
- Ratios in the consumer contract can be unstable if the native denominator is
  small.
- A shared basis may predict scalar losses while failing selective-removal
  responses. Each consequence family therefore needs a separate held-out gate.

### Measurable consequence beyond reconstruction

If this passes, a single small latent program state predicts multiple causal
consequences. Components can then be named by their response signatures, extracted
once, and edited while bounding collateral coordinates. It also tells us whether a
“simple program” is objective-specific or genuinely shared.

### Cheapest falsifier

Once the final response reducer closes, fit the shared basis without one complete
consequence coordinate and without a registered set of masks. Reject shared
factorization if it is worse than independent bases at matched total floats, or if
held-out copy/frequency/selective-removal error is not controlled.

The current two-output diagnostic is only suggestive: top-1 and CE share the
rank-two maximum-volume pivot, but not the rank-three or rank-four pivots.

## Priority 3: an action-Hankel minimal realization, conditional on priorities 1–2

### Exact object

For legal action words \(u,v\), define a block Hankel matrix

\[
    \mathsf H(u,v)=F(uv).
\]

Rows are physical prefix programs, columns are physical suffix programs, and entries
are consequence vectors. This differs from the rejected token-splice Hankel matrix:
every concatenation here is an executable intervention on the same model.

### Theorem and operational definition

For a rational series, the rank of the infinite Hankel matrix equals the number of
states in its minimal weighted automaton. Given a complete finite basis and rank
factorization \(H=PS\), shifted Hankel blocks recover the transition matrices. See
[Arrivault et al., 2017](https://proceedings.mlr.press/v57/arrivault16.html).

Operationally, the minimal state is the smallest dimension whose frozen transition
maps predict new action words at several adjacent physical cuts. This state dimension
is invariant to invertible changes of coordinates, which is exactly the gauge issue
that makes raw factors hard to interpret.

### Assumptions that may fail

- Bilin18 plus RMSNorm need not define a finite-dimensional linear realization in
  the selected output coordinates.
- Site transitions are not homogeneous. We may need layer-indexed transitions,
  weakening the ordinary automaton theorem.
- A single scalar such as CE is not a sufficient state observation.
- A low-rank block at one cut is not a complete Hankel basis and does not certify a
  global realization.

### Measurable consequence beyond reconstruction

A passing realization is an executable smaller program: its state updates compose,
predict arbitrary unseen intervention words, and identify behaviorally equivalent
internal states. That is a direct route from tensor compression to a manipulable
program rather than a collection of per-site approximations.

### Cheapest falsifier

After priority 1 supplies stable bases, construct shifted blocks at two adjacent
cuts. Freeze one state basis and transition map, then predict a held-out set of words
and the next cut's response block. Reject if rank changes wildly, the transferred
basis is ill-conditioned, or an additive baseline wins. Do not run a larger version
of the old token-splice experiment.

## Candidate mathematics considered and pruned or subordinated

### Robust PCA: pruned by the new CPU diagnostic

Principal-component pursuit decomposes \(M=L+S\) by minimizing nuclear norm plus an
\(\ell_1\) penalty. Under incoherence of \(L\) and sufficiently sparse, suitably
distributed support of \(S\), it can exactly recover both components
([Candès et al., 2009](https://arxiv.org/abs/0912.3599)). Our matrix is only 7 by 7,
its masks are highly structured, and—decisively—the rank-two residual is diffuse.
Generic low-rank-plus-sparse exceptions are not the next experiment.

### Hierarchical Möbius/polynomial sparsity: retained only as a residual fallback

A mask-cost function has an exact discrete Möbius expansion into singleton, pair,
and higher interactions. Downward-closed or strong-heredity support can reduce sample
complexity when high-order terms occur only with their lower-order ancestors; related
weighted compressed-sensing guarantees exist for polynomial lower sets
([Chkifa et al., 2016](https://arxiv.org/abs/1602.05823)).

This remains useful if TT-cross leaves a structured residual, because its terms map
to explicit interaction edits. It is not ranked above cross interpolation because:

- earlier additive and independent-choice rules already failed;
- the present masks combine several sites, so a clean per-site Möbius basis has not
  yet been measured; and
- arbitrary sparse residuals have just been rejected.

The fair comparison on a new mask bank is: spend one extra TT rank versus spend the
same number of coefficients on a preregistered hereditary residual. Whichever better
predicts untouched masks at matched cost wins.

### Approximate causal bisimulation: validation layer, not current generator

Two program states may be merged only when every allowed action gives the same
immediate consequence and transitions to equivalent states. Quantitative versions
use a fixed-point pseudometric combining immediate discrepancy and a Wasserstein
distance between next-state distributions
([Panangaden et al., 2024](https://www.jmlr.org/papers/volume25/23-1415/23-1415.pdf)).

This is the right eventual definition for safe removal and edit locality, but it
cannot generate a quotient until the final response tensor has real rows. It remains
the admission test after priorities 1–3, not a reason to cluster current activations.

### Gauge norm minimization followed by HOSVD: closed, not reopened

The exact toy result already established the boundary. Scalar balancing can reduce
factor norm by orders of magnitude while leaving the contracted tensor and every
HOSVD spectrum unchanged. On a genuine shared general-linear edge, norm balancing
can improve conditioning before canonicalization, but it does not create a lower
tensor rank. It becomes relevant after a shared response factorization exists.

### SAE/dictionary learning on weights: deferred

Sparse coding weight columns can find a short description of the chosen coordinates,
but it does not by itself predict program compositions, OOD transport, or collateral
damage. A dictionary becomes relevant only as a factorization of the shared physical
response bases in priority 2 and must beat the TT cores at matched executable cost.

### Arithmetic-circuit rank and invariant theory: local certificate only

Partial-derivative ranks and tensor ranks can lower-bound bilinear/polynomial circuit
size, while contracted maps and cycle quantities remove gauge fiction. They are
valuable once the exact polynomial subgraph and its input/output interface are
fixed. Applying them directly to CE would mix polynomial computation with RMSNorm
and the nonlinear loss and would not predict behavior.

### Information bottleneck and MDL: evaluation, not discovery

Mutual information is not automatically meaningful for deterministic continuous
states, and parameter count ignores precision and gauge. Prequential MDL is useful
after two runnable families exist: a lower-rank TT program should need fewer bits and
stabilize with less data. Neither measure replaces untouched-mask, OOD, extraction,
or selective-removal tests.

### Approximation certificates

The cross theorem supplies a structural interpolation bound in terms of neglected
singular values and pivot quality. It does **not** certify final CE through the full
nonlinear model. Any whole-model certificate still needs empirical held-out response
bounds or a downstream-weighted local bound whose assumptions are checked. Global
Lipschitz products remain too loose to prioritize.

## Ranked next actions

1. **Prospectively preregister a maximum-volume TT-cross mask assay at a fresh cut.**
   Highest information gain, direct whole-program prediction, explicit
   falsification, and moderate GPU cost.
2. **Close the 68-action semantic reducer and make the cross output vector-valued.**
   This tests whether one latent tensor program predicts CE, extraction, and
   collateral consequences rather than overfitting a scalar objective.
3. **If two adjacent crosses pass, recover and test a small action-Hankel
   realization.** This is the route from low cut ranks to a composable executable
   state machine.
4. **Fit a hereditary Möbius residual against one additional TT rank at matched
   coefficient cost.** Run only if cross residuals show stable feature structure.
5. **Apply causal-bisimulation partition refinement and prequential/MDL pricing to
   admitted programs.** These validate editability and simplicity after a predictive
   object exists.

## What this review changed

- It rejected generic sparse exceptions as the explanation for the cut-rank
  discrepancy.
- It replaced arbitrary missing-cell matrix completion with an active,
  theorem-backed choice of program executions.
- It identified rank three to four—not rank one to two—as the retrospective range
  worth falsifying prospectively at this cut.
- It organized the mathematical path as: **cross-selected measurements -> shared
  consequence factors -> minimal action realization -> causal quotient/MDL**.

No explained-fraction ledger is incremented by this work. The only new empirical
claim is the retrospective CPU diagnostic, with its discovery-only role stated in
the result artifact.
