# Three-hour mathematical review — 2026-08-29 19:18 UTC

## Evidence that changes the mathematics

The valid MLP2 finite receipt is now the central constraint.  Every native-product
K=512 program fails, and deleting MLP2 is less harmful than retaining any tested
subset.  For SUFFIX, final dCE/KL/logit-NRMSE/top-1 agreement are
`0.28920 / 0.29711 / 0.33150 / 72.60%`; ZERO is
`0.16235 / 0.16906 / 0.24458 / 77.74%`.  The joint/singleton distortion ratio is
`1.83497`, and small/full signed-effect cosine is `0.26118`.

Therefore the mathematical object cannot be “4,608 mostly independent useful
channels.”  Any next basis must represent coordinated mixtures, downstream
observability, or higher-order interaction terms.  The strict whole-model ledger
remains 10.923302467% named causal CE, 4.72714 nat unnamed, and 0/68 complete
terminal actions.  The GPU is occupied by the quotient lane, so this review executes
only CPU analysis of the already-authorized validation ledger.

## Ranked move 1 — finite-response balanced realization

### Exact bilin18 object

Treat the MLP2 product/write interface as a controlled nonlinear system.  The
“reachable” directions are mixtures of product deviations induced by real residual
states and explicit residual/product interventions.  The “observable” directions
are mixtures whose finite interventions change final logits or a verified consumer
bank.  Unlike the failed selector, the state coordinates are arbitrary mixtures, not
native product channels.

Let (P) be an empirical covariance/controllability Gramian of product responses and
(Q) an empirical observability Gramian from finite downstream effects.  A balanced
basis comes from the singular vectors of (L_Q^\top L_P), where
(P=L_PL_P^\top) and (Q=L_QL_Q^\top).  Small singular directions are simultaneously
hard to reach and hard to observe.

This is the empirical analogue of nonlinear balanced reduction.  Scherpen defines
nonlinear balancing through input/output energy functions; Lall, Marsden, and
Glavaški give an empirical construction that reduces to ordinary balanced truncation
for linear systems.  Quadratic-bilinear work constructs algebraic Gramians from
Volterra kernels and uses them to remove states that are both weakly reachable and
observable.

Primary sources:

- [Scherpen, *Balancing for nonlinear systems* (1993)](https://doi.org/10.1016/0167-6911(93)90117-O)
- [Lall, Marsden & Glavaški, *Empirical Model Reduction of Controlled Nonlinear Systems*](https://authors.library.caltech.edu/records/zz5nw-nm878)
- [Benner & Goyal, *Balanced Truncation Model Order Reduction for Quadratic-Bilinear Control Systems*](https://arxiv.org/abs/1705.00160)
- [Benner, Goyal & Gugercin, quadratic-bilinear \(\mathcal H_2\) quasi-optimal reduction](https://doi.org/10.1137/16M1098280)

### Assumptions that may fail

Classical error bounds assume a stable dynamical system and suitable Gramians.  MLP2
is a static bilinear node inside a feed-forward transformer; the suffix includes
RMSNorm, attention softmax, and other bilinear blocks.  Empirical Gramians are
distribution- and intervention-dependent.  A local (Q) from Jacobians is already
insufficient, as the signed finite test showed.

Therefore “balanced” here is an operational definition: the basis must be estimated
from multiple finite radii and must pass held-out final CE/KL, signed edits,
composition with MLP0 C512, and OOD transport.  No classical bound will be claimed
without checking its assumptions.

### Prediction beyond reconstruction

If a small balanced realization exists, document-by-program causal harm should have
one or a few stable response modes; mixtures learned from finite effects should beat
ZERO and all native K512 arms at equal executable price; and their ordering should
remain stable across 48/96/192 documents and consumer cells.

### Cheapest falsifier, preregistered here

From the existing ledger, form the document-by-arm dCE matrix for
`SUFFIX/LOCAL/RMS/MASS/DERANGED/HASH_RANDOM`.  Remove each arm's mean and scale to
unit RMS.  Report singular-value energy and the cosine between first right-singular
vectors on documents 0--95 and 96--191.  A **single shared imbalance mode** is
promising only if rank-1 energy is at least 80% and split cosine at least 0.90.  A
two-mode block is promising if rank-2 energy is at least 90%.  Otherwise the cheap
one/two-mode version is falsified, though higher-rank balancing remains possible.

## Ranked move 2 — rooted-tree/Volterra response compiler

### Exact bilin18 object

The tensor transformer is an arithmetic DAG: bilinear MLP products and attention
interact through residual addition, while RMSNorm and softmax are smooth scalar/vector
maps on the observed domain.  Around an intervention path (t), the final response
can be organized by rooted interaction trees: linear terms, pairwise curvature,
third-order interactions, and so on.  Quadratic-bilinear model reduction uses the
same Volterra-kernel organization; recent work makes the binary-tree structure
explicit and derives Gramian/error consequences for genuine QB systems.

Primary sources:

- [Benner, Goyal & Gugercin (2018), Volterra kernels and QB optimality conditions](https://doi.org/10.1137/16M1098280)
- [Redmann, *Tree-based solution representations for quadratic bilinear systems* (2026)](https://arxiv.org/abs/2607.07841)

### Assumptions that may fail

The transformer suffix is not a stable continuous-time QB system, and softmax/RMSNorm
make the exact expansion non-polynomial.  Taylor/Volterra expansions may have a small
radius; the observed tangent failure may reflect large even-order curvature or may
mean no low order extrapolates to full deletion.

### Prediction beyond reconstruction

A degree-2 or degree-3 response law fitted only at small signed edits should predict
the full (t=1) held-out per-document CE effect.  If it does, interaction-tree terms
provide a composable correction language and a route to finite certificates.  If it
does not, local polynomial compilation must not be used for deletion/removal claims.

### Cheapest falsifier, preregistered here

For every supported validation document, fit zero-intercept degree 1, 2, and 3
polynomials to dCE at (t=-.25,-.10,+.10,+.25), then predict the physical SUFFIX
effect at (t=1).  Report pooled error, document-level NRMSE/correlation/sign
agreement, and 48/96/192 stability.  Degree at most 3 is promising only if the
192-document pooled absolute error is at most 0.02 nat and document NRMSE at most
0.25.  These bars are frozen before running the calculation.

## Ranked move 3 — causal-state/Hankel minimal realization

### Exact bilin18 object

Construct a matrix (H) whose rows are early histories or controlled early-state
interventions and whose columns are future “tests”: final-token distributions and
verified copy, capitalization, numeric, syntax, and entity-continuation circuit
responses.  Two residual states are equivalent if they give the same row of future
test predictions under allowed continuations/interventions.  The rank of (H) is an
operational predictive-state dimension.

For weighted automata, finite Hankel rank equals the number of states in a minimal
linear realization.  Predictive-state representations use multi-step conditional
predictions directly as state and can be no larger than a minimal POMDP realization.

Primary sources:

- [Carlyle & Paz, *Realizations by stochastic finite automata* (1971)](https://doi.org/10.1016/S0022-0000(71)80005-3)
- [Littman, Sutton & Singh, *Predictive Representations of State* (2001)](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)
- [Singh, James & Rudary, *Predictive State Representations*](https://arxiv.org/abs/1207.4167)

### Assumptions that may fail

Exact Hankel minimality applies to rational/linear sequential realizations with a
sufficient family of histories and tests.  Natural language has effectively infinite
alphabet/state, our test bank is incomplete, and low empirical rank can be sampling
or probe rank rather than true minimality.

### Prediction beyond reconstruction

A stable finite rank predicts held-out and OOD consumer responses, supplies a
canonical interface shared across layers, and makes states in the same row-equivalence
class interchangeable.  This directly supports extraction, selective removal, and
composition.  Interchange interventions provide the causal-abstraction check rather
than mere probe accuracy; see [Geiger et al., causal abstractions of neural networks](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html).

### Cheapest falsifier

After adding at least three non-copy verified consumers, build a held-out
history-by-consumer intervention matrix and test whether its numerical rank and
subspace stabilize across document halves and new continuation templates.  The
current copy-only bank is too narrow; running Hankel SVD now would merely rediscover
probe count.

## Ideas pruned in this review

- **Ordinary HOSVD/CP/Tucker on coefficient Frobenius norm:** already contradicted by
  Family F and the finite MLP2 result.  HOSVD solves a multilinear least-squares
  problem, not the downstream causal metric; its guarantees do not cross RMSNorm and
  the residual suffix.  [The original HOSVD approximation paper](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/delathauwer-2000-rank-approx.pdf)
  supports the coefficient objective, not our causal claim.
- **Norm minimization before HOSVD:** useful only to choose a gauge or whiten a metric.
  It cannot change the represented function, repair missing finite observability, or
  turn a Frobenius optimum into a CE optimum.  It becomes relevant inside balanced
  coordinates, not as a standalone entry point.
- **Tensor-CUR/native fiber selection:** interpretability through actual fibers is
  attractive, but K512 native fiber selection just failed.  Arbitrary mixtures are
  now required before reconsidering sparse fibers.
- **Information bottleneck alone:** predictive sufficiency is useful, but mutual
  information does not establish intervention equivalence or executable tensor cost.
  Use it only after defining the causal-state tests.
- **MDL/prequential coding alone:** a good selector among executable programs, not a
  generator of the missing balanced basis.  Keep it as a validation currency.
- **Sparse program synthesis:** premature until balanced/Hankel coordinates expose a
  small dictionary.  Searching native channels now repeats a closed branch.
- **Margin/Lipschitz certification alone:** the 2% native margin quantile is zero and
  every K512 margin bound collapsed.  Distributional CE/KL and cellwise certificates
  must precede any worst-case claim.

## Execution status

The two frozen CPU falsifiers above are the highest-priority safe action.  Their
results will be appended below without changing the thresholds.

